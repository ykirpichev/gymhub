"""C3 in small: a two-stage pipeline across machines with compressed activations (M7, #91).

A byte-level transformer is cut in two: rank 0 holds the embedding and the first half of the
blocks, rank 1 the second half, the head and the loss. Every step rank 0 sends its activations to
rank 1, and rank 1 sends back the gradient with respect to them. These two messages are all the
traffic, and they are what a slow link between machines has to carry; this script compresses both:

    none      fp32, as computed
    bf16      2 bytes per value
    int8      per-token absmax scale (fp32) + int8 values: ~1 byte per value
    topk:F    the largest fraction F of each token's values (bf16 values + int16 indices), with error
              feedback for the gradients (what was left out is added to the next step's message)

Run as a pool gang (ranks meet through RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT):

    pool run --gang 2 --network peers --gpus 1 --image nvcr.io/nvidia/pytorch:25.01-py3 \
        -- python3 pipeline.py --compress int8

or on one machine, two processes over loopback: `python3 pipeline.py --compress int8`. The last
line printed is a JSON summary: held-out loss, bytes per step and measured seconds per step, and
the step time a 100 Mbit/s or 1 Gbit/s link would give (compute plus transfer, no overlap).
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


# ---- data and model ----------------------------------------------------------------------------------


def corpus(limit: int = 6 << 20) -> torch.Tensor:
    """Bytes of the Python standard library's source: the same on every machine with Python."""
    root = os.path.dirname(os.__file__)
    data = bytearray()
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("site-packages", "test", "tests", "__pycache__"))
        for name in sorted(files):
            if name.endswith(".py"):
                with open(os.path.join(folder, name), "rb") as handle:
                    data += handle.read()
            if len(data) >= limit:
                return torch.frombuffer(data[:limit], dtype=torch.uint8).long()
    return torch.frombuffer(data, dtype=torch.uint8).long()


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d))
        return x + self.ffn(self.ln2(x))


class First(nn.Module):
    def __init__(self, d, heads, layers, seq):
        super().__init__()
        self.emb, self.pos = nn.Embedding(256, d), nn.Parameter(torch.zeros(seq, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))

    def forward(self, tokens):
        x = self.emb(tokens) + self.pos[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return x


class Second(nn.Module):
    def __init__(self, d, heads, layers):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, 256)

    def forward(self, x, targets):
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        return F.cross_entropy(logits.reshape(-1, 256).float(), targets.reshape(-1))


# ---- compression: a tensor <-> one uint8 message --------------------------------------------------------


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


class Codec:
    """encode(x) -> uint8 message of a fixed size for this shape; decode(message) -> x' (float32)."""

    def __init__(self, kind: str, shape: tuple, feedback: bool = False):
        self.kind, self.shape, self.feedback = kind, shape, feedback
        self.rows, self.cols = math.prod(shape[:-1]), shape[-1]
        self.k = max(1, int(round(float(kind.split(":")[1]) * self.cols))) if kind.startswith("topk") else None
        self.residual = None  # error feedback for top-k

    def size(self) -> int:
        n = self.rows * self.cols
        if self.k is not None:  # bf16 value + int16 index per kept value
            return self.rows * self.k * 4
        return {"none": 4 * n, "bf16": 2 * n, "int8": n + 4 * self.rows}[self.kind]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().reshape(self.rows, self.cols)
        if self.kind == "none":
            return _bytes(x)
        if self.kind == "bf16":
            return _bytes(x.to(torch.bfloat16))
        if self.kind == "int8":
            scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
            q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
            # the scale stays fp32: gradients are ~1e-6, and 1e-6 / 127 underflows in fp16
            return torch.cat([_bytes(q), _bytes(scale)])
        if self.residual is not None:
            x = x + self.residual
        values, index = x.abs().topk(self.k, dim=1)
        kept = torch.gather(x, 1, index)
        sparse = torch.zeros_like(x).scatter_(1, index, kept)
        if self.feedback:
            self.residual = x - sparse
        del values
        return torch.cat([_bytes(kept.to(torch.bfloat16)), _bytes(index.to(torch.int16))])

    def decode(self, message: torch.Tensor) -> torch.Tensor:
        n = self.rows * self.cols
        if self.kind == "none":
            x = message.view(torch.float32)
        elif self.kind == "bf16":
            x = message.view(torch.bfloat16).float()
        elif self.kind == "int8":
            q = message[:n].view(torch.int8).float().reshape(self.rows, self.cols)
            x = q * message[n:].view(torch.float32).reshape(self.rows, 1)
        else:
            m = self.rows * self.k
            kept = message[: 2 * m].view(torch.bfloat16).float().reshape(self.rows, self.k)
            index = message[2 * m:].view(torch.int16).long().reshape(self.rows, self.k)
            x = torch.zeros(self.rows, self.cols, device=message.device).scatter_(1, index, kept)
        return x.reshape(self.shape)


# ---- the two ranks -----------------------------------------------------------------------------------------


def run(rank: int, world: int, args) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(0)
    data = corpus()
    split = int(len(data) * 0.95)
    train, held = data[:split], data[split:]
    shape = (args.batch, args.seq, args.d)
    model = (First(args.d, args.heads, args.layers, args.seq) if rank == 0 else Second(args.d, args.heads, args.layers)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # error feedback only for gradients: an activation left out cannot be made up for at the next step
    forward, backward = Codec(args.compress, shape), Codec(args.compress_grad or args.compress, shape, feedback=True)
    generator = torch.Generator().manual_seed(1)

    def batch(source, gen):
        start = torch.randint(0, len(source) - args.seq - 1, (args.batch,), generator=gen)
        rows = torch.stack([source[s: s + args.seq + 1] for s in start.tolist()])
        return rows[:, :-1].to(device), rows[:, 1:].to(device)

    sent = 0
    compute = comm = 0.0
    losses = []
    for step in range(args.steps + args.eval_batches):
        evaluating = step >= args.steps
        tokens, targets = batch(held if evaluating else train, generator)  # both ranks draw the same batch
        lr = args.lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
        for group in opt.param_groups:
            group["lr"] = lr
        if rank == 0:
            t0 = time.perf_counter()
            with torch.set_grad_enabled(not evaluating):
                out = model(tokens)
            message = forward.encode(out).cpu()
            t1 = time.perf_counter()
            dist.send(message, 1)
            sent += message.numel()
            if not evaluating:
                grad_message = torch.empty(backward.size(), dtype=torch.uint8)
                dist.recv(grad_message, 1)
                t2 = time.perf_counter()
                grad = backward.decode(grad_message).to(device)
                opt.zero_grad(set_to_none=True)
                out.backward(grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                compute += (t1 - t0) + (t3 - t2)
                comm += t2 - t1
        else:
            message = torch.empty(forward.size(), dtype=torch.uint8)
            t0 = time.perf_counter()
            dist.recv(message, 0)
            t1 = time.perf_counter()
            x = forward.decode(message).to(device).requires_grad_(not evaluating)
            if evaluating:
                with torch.no_grad():
                    losses.append(float(model(x, targets)))
                continue
            loss = model(x, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            grad_message = backward.encode(x.grad).cpu()
            t2 = time.perf_counter()
            dist.send(grad_message, 0)
            sent += grad_message.numel()
            compute += t2 - t1
            if step % 100 == 0:
                print(f"[rank 1] step {step} loss {loss.item():.3f}", flush=True)
    # gather the summary on rank 1
    totals = torch.tensor([sent, compute, comm], dtype=torch.float64)
    other = torch.zeros(3, dtype=torch.float64)
    if rank == 0:
        dist.send(totals, 1)
    else:
        dist.recv(other, 0)
        steps = args.steps
        bytes_per_step = (other[0].item() - args.eval_batches * forward.size() + totals[0].item()) / steps
        assert bytes_per_step == forward.size() + backward.size()
        compute_s = (other[1].item() + totals[1].item()) / steps  # the two stages run one after the other
        summary = {"compress": args.compress, "compress_grad": args.compress_grad or args.compress, "held_out_loss": round(sum(losses) / len(losses), 4),
                   "bytes_per_step": int(bytes_per_step), "compute_s_per_step": round(compute_s, 4),
                   "loopback_s_per_step": round(compute_s + other[2].item() / steps, 4),
                   "s_per_step_at_100mbps": round(compute_s + bytes_per_step * 8 / 100e6, 4),
                   "s_per_step_at_1gbps": round(compute_s + bytes_per_step * 8 / 1e9, 4),
                   "steps": steps, "batch": args.batch, "seq": args.seq, "d": args.d, "layers_per_stage": args.layers,
                   "device": device.type}
        print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compress", default="none", help="none, bf16, int8 or topk:F (e.g. topk:0.1)")
    parser.add_argument("--compress-grad", help="for the gradients sent back (default: as --compress)")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2, help="blocks per stage")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--port", type=int, default=0, help="on one machine: the rendezvous port (default: a free one)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    for kind in (args.compress, args.compress_grad or "none"):
        if kind not in ("none", "bf16", "int8") and not (kind.startswith("topk:") and 0 < float(kind[5:]) <= 1):
            parser.error("--compress: none, bf16, int8 or topk:F with 0 < F <= 1")
    if "RANK" in os.environ:  # a pool gang: one rank per machine
        run(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
    else:
        if not args.port:
            import socket

            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                args.port = probe.getsockname()[1]
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(args.port)
        mp.spawn(run, args=(2, args), nprocs=2, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
