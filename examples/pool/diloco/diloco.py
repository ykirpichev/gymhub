"""DiLoCo across unequal machines on a slow link: a pool gang job (issue #45, recipe C1 in small).

Every rank trains its own copy of a small byte-level transformer for a fixed stretch of wall time
(the inner loop, AdamW), then the ranks average how far each moved from the shared weights (the
pseudo-gradient), weighted by how many tokens each processed, and apply it with an outer Nesterov
step. Only one model-sized message per rank crosses the network per round, so a Wi-Fi link between
a DGX Spark and a Jetson is enough. Rounds are timed rather than counted, so the fast machine never
waits for the slow one: it simply does more steps, and its update weighs more.

    pool run --gang 2 --network peers --gpus 1 --mem 8 \
        --image nvcr.io/nvidia/vllm:26.04-py3 --image-for gpu=tegra=gpu-gym-jetson:local-r36.4-cu126 \
        --data corpus=./corpus -- python diloco.py --rounds 10 --round-seconds 30

Without a gang (WORLD_SIZE unset) it trains alone: plain AdamW for the same wall time, or
`--outer` to run the DiLoCo outer loop with one worker. Metrics go to $POOL_OUTPUTS/metrics.json.
The outer step is SGD with optional Nesterov momentum (--outer-lr 0.7, --outer-momentum 0 by default).

--robust (the checked tier, M6) all-gathers the updates and leaves out outliers and excluded machines
(see robust_sync); --poison makes one rank send a poisoned update, to test that (poison.sh).

Reference: Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models" (2023).
"""

import argparse
import datetime
import json
import math
import os
import platform
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

if os.environ.get("POOL_LIB"):  # in a pool job: the job library and the built-in reducers sit there
    sys.path.insert(0, os.environ["POOL_LIB"])
    import pool_reducers as reducers
    import pool_job
else:  # from a checkout of the repository
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from pool import reducers
    pool_job = None


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class ByteGPT(nn.Module):
    def __init__(self, layers=4, d=256, heads=4, ctx=256):
        super().__init__()
        self.tok, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, 256, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def batches(data: torch.Tensor, batch: int, ctx: int, generator: torch.Generator, device):
    while True:
        starts = torch.randint(0, len(data) - ctx - 1, (batch,), generator=generator)
        window = torch.stack([data[s: s + ctx + 1] for s in starts.tolist()]).long().to(device, non_blocking=True)
        yield window[:, :-1], window[:, 1:]


def loss_of(model, x, y):
    return F.cross_entropy(model(x).float().view(-1, 256), y.reshape(-1))


@torch.no_grad()
def evaluate(model, val: torch.Tensor, ctx: int, device, n: int = 16, batch: int = 32) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(1234)  # the same held-out windows every time
    it = batches(val, batch, ctx, generator, device)
    total = 0.0
    for _ in range(n):
        x, y = next(it)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total += loss_of(model, x, y).item()
    model.train()
    return total / n


def flat(params) -> torch.Tensor:
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in params])


def unflat(vector: torch.Tensor, params) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vector[offset: offset + n].view_as(p).to(p.device, p.dtype))
        offset += n


def poison(delta: torch.Tensor, how: str, scale: float, round_no: int) -> torch.Tensor:
    """What a malicious participant sends instead of its update (the M6 demo)."""
    if how == "scale":
        return delta * scale
    if how == "flip":
        return -delta
    if how == "random":  # noise of the same norm: no larger than an honest update, pointing nowhere
        noise = torch.randn(delta.shape, generator=torch.Generator().manual_seed(round_no))
        return noise * (delta.norm() / noise.norm())
    return delta


def robust_sync(delta, tokens, world, rank, machines, excluded, sync_dtype, args):
    """Update sanity (M6): every rank gathers every update, and all of them leave out the same ones: the
    outliers (pool_reducers.outliers, on the same inputs everywhere) and the ranks on machines the pool
    has excluded. Rank 0 reports each machine's update to the pool (pool_job.report), so failed checks
    add up in its reputation; once the pool excludes a machine, rank 0 tells the others."""
    parts = [torch.zeros(delta.numel(), dtype=sync_dtype) for _ in range(world)]
    dist.all_gather(parts, delta.to(sync_dtype).contiguous())
    counts = [torch.zeros(1) for _ in range(world)]
    dist.all_gather(counts, torch.tensor([float(tokens)]))
    weights = [int(c.item()) for c in counts]
    active = [r for r in range(world) if r not in excluded]
    changes = [parts[r].float().numpy() for r in active]
    stats = dict(zip(active, reducers.compare(changes)))
    outlying = {active[i] for i in reducers.outliers(changes, args.norm_factor, args.min_cosine)}
    if rank == 0 and pool_job is not None:
        for r in active:
            if r not in machines:
                continue
            ratio, cosine = stats[r]
            standing = pool_job.report(machines[r], r not in outlying, "update",
                                       f"DiLoCo update: norm {ratio:.2f}x the others' median, cosine {cosine:.2f}")
            if standing.get("excluded"):
                excluded.add(r)
    flags = torch.tensor([1.0 if r in excluded else 0.0 for r in range(world)])
    dist.broadcast(flags, 0)
    excluded.update(r for r in range(world) if flags[r] > 0)
    left_out = sorted(outlying | excluded)
    kept = [r for r in range(world) if r not in left_out]
    total = sum(weights[r] for r in kept)
    merged = sum(parts[r].float() * (weights[r] / total) for r in kept) if kept else torch.zeros_like(delta)
    checks = {"left_out": left_out, "outlying": sorted(outlying), "excluded": sorted(excluded),
              "norm_ratio": {r: round(s[0], 2) for r, s in stats.items()}, "cosine": {r: round(s[1], 3) for r, s in stats.items()}}
    return merged, weights, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--round-seconds", type=float, default=30)
    parser.add_argument("--round-steps", type=int, default=0,
                        help="a fixed number of inner steps per round instead of a fixed time (for comparisons between runs)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.0,
                        help="Nesterov momentum of the outer step. DiLoCo uses 0.9 for long runs from a warm start; "
                             "from random weights and a few rounds it overshoots (see README.md), so the default is 0")
    parser.add_argument("--outer", action="store_true", help="alone: still run the outer loop (DiLoCo with one worker)")
    parser.add_argument("--sync-dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--corpus", default=os.path.join(os.environ.get("POOL_INPUTS", "."), "corpus", "corpus.txt"))
    parser.add_argument("--robust", action="store_true",
                        help="update sanity (M6): all-gather the updates, drop outliers (pool_reducers.outliers), report "
                             "each rank's machine (pool_job.report), and leave out machines the pool excludes")
    parser.add_argument("--norm-factor", type=float, default=3.0, help="--robust: drop updates this many times the median's norm")
    parser.add_argument("--min-cosine", type=float, default=0.1, help="--robust: drop updates less aligned with the median")
    parser.add_argument("--poison", choices=["none", "scale", "flip", "random"], default="none",
                        help="a malicious participant, for the M6 demo: its update scaled up, sign-flipped, or replaced "
                             "by noise of the same norm")
    parser.add_argument("--poison-machine", help="the machine (name, from POOL_PEERS) whose rank poisons; default the last rank")
    parser.add_argument("--poison-scale", type=float, default=10.0)
    args = parser.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    distributed = world > 1
    if distributed:
        dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                                rank=rank, world_size=world, timeout=datetime.timedelta(minutes=10))
    outer = distributed or args.outer
    machines = {p["rank"]: p["machine"] for p in json.loads(os.environ.get("POOL_PEERS") or "[]")}  # set by the pool
    poisoner = -1
    if args.poison != "none":
        poisoner = (next((r for r, m in machines.items() if m == args.poison_machine), -1)
                    if args.poison_machine else world - 1)
    excluded: set[int] = set()  # ranks whose machines the pool has excluded (--robust)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    who = f"rank {rank}/{world} on {torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.machine()}"
    print(f"{who}: torch {torch.__version__}, {'DiLoCo' if outer else 'plain AdamW'}", flush=True)

    raw = torch.frombuffer(bytearray(open(args.corpus, "rb").read()), dtype=torch.uint8)
    split = int(len(raw) * 0.95)
    train, val = raw[:split], raw[split:]
    torch.manual_seed(0)  # the same initial weights on every rank
    model = ByteGPT(args.layers, args.dim, max(1, args.dim // 64), args.ctx).to(device)
    params = [p for p in model.parameters()]
    count = sum(p.numel() for p in params)
    inner = torch.optim.AdamW(params, lr=args.inner_lr, weight_decay=0.1, betas=(0.9, 0.95))
    shared = flat(params)  # the outer (global) weights, on the CPU, fp32
    momentum = torch.zeros_like(shared)
    it = batches(train, args.batch, args.ctx, torch.Generator().manual_seed(1000 + rank), device)
    sync_dtype = torch.float16 if args.sync_dtype == "fp16" else torch.float32
    if rank == 0:
        print(f"model {count / 1e6:.2f}M parameters; corpus {len(raw) / 1e6:.1f} MB; "
              f"{count * (2 if sync_dtype == torch.float16 else 4) / 1e6:.1f} MB per rank per sync", flush=True)

    history = []
    started = time.time()
    first = evaluate(model, val, args.ctx, device)
    if rank == 0:
        history.append({"round": 0, "elapsed_s": 0.0, "val_loss": round(first, 4)})
        print(f"round 0: val loss {first:.4f}", flush=True)
    for round_no in range(1, args.rounds + 1):
        steps, losses, stop = 0, [], time.time() + args.round_seconds
        while (steps < args.round_steps) if args.round_steps else (time.time() < stop):
            x, y = next(it)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_of(model, x, y)
            inner.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            inner.step()
            steps += 1
            if steps % 20 == 0:
                losses.append(loss.item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        tokens = steps * args.batch * args.ctx
        sync_s, weights, checks = 0.0, [tokens], None
        if outer:
            delta = shared - flat(params)  # how far this rank moved: its pseudo-gradient
            if rank == poisoner:
                delta = poison(delta, args.poison, args.poison_scale, round_no)
            sync_start = time.time()
            if distributed and args.robust:
                delta, weights, checks = robust_sync(delta, tokens, world, rank, machines, excluded, sync_dtype, args)
            elif distributed:
                counts = torch.tensor([float(tokens)])
                dist.all_reduce(counts)  # tokens of every rank, for the weights
                share = tokens / counts.item()
                message = (delta * share).to(sync_dtype)
                dist.all_reduce(message)
                delta = message.float()
                gathered = [torch.zeros(1) for _ in range(world)]
                dist.all_gather(gathered, torch.tensor([float(tokens)]))
                weights = [int(t.item()) for t in gathered]
            sync_s = time.time() - sync_start
            momentum.mul_(args.outer_momentum).add_(delta)  # Nesterov SGD on the averaged pseudo-gradient
            shared -= args.outer_lr * (delta + args.outer_momentum * momentum)
            unflat(shared, params)
        val_loss = evaluate(model, val, args.ctx, device) if rank == 0 else None
        line = {"round": round_no, "elapsed_s": round(time.time() - started, 1), "steps": steps, "tokens_by_rank": weights,
                "train_loss": round(sum(losses) / len(losses), 4) if losses else None, "sync_s": round(sync_s, 2)}
        if checks:
            line["checks"] = checks
        if rank == 0:
            line["val_loss"] = round(val_loss, 4)
            history.append(line)
        print(f"round {round_no}: {steps} steps here, tokens by rank {weights}, sync {sync_s:.2f} s"
              + (f", val loss {val_loss:.4f}" if val_loss is not None else "")
              + (f", left out rank(s) {checks['left_out']}" if checks and checks["left_out"] else ""), flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        out = os.environ.get("POOL_OUTPUTS", ".")
        summary = {"world_size": world, "mode": "diloco" if outer else "adamw", "params": count, "args": vars(args),
                   "device": who, "history": history, "final_val_loss": history[-1]["val_loss"],
                   "total_s": round(time.time() - started, 1),
                   "bits_per_byte": round(history[-1]["val_loss"] / math.log(2), 4),
                   "poisoner": machines.get(poisoner, poisoner) if poisoner >= 0 else None,
                   "excluded": [machines.get(r, r) for r in sorted(excluded)]}
        with open(os.path.join(out, "metrics.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"final val loss {summary['final_val_loss']:.4f} ({summary['bits_per_byte']:.3f} bits/byte) "
              f"after {summary['total_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
