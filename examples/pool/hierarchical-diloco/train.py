"""Hierarchical DiLoCo of OLMoE with torchft, islands as pool jobs (issue #89, Demo C1 in small).

Each island is one pool job (a gang; here one rank, since the demo machine has one GPU). Inside an
island the ranks would sync every step (DDP); between islands torchft's DiLoCo syncs every
`--inner-steps` steps over the slow link. The pool round on `--ref` (pool_reducers.TorchftRounds)
carries the lighthouse address, the quorum and the deadline, and records each outer step.

    # two islands (two pool jobs, network: peers), each processing --batch sequences per step
    python3 train.py --mode island --island 0 --islands 2 --ref c1/run1 ...
    python3 train.py --mode island --island 1 --islands 2 --ref c1/run1 ...
    # the single-island baseline at equal tokens: one replica, --islands x --batch per step
    python3 train.py --mode single --islands 2 ...
    # step time of the full OLMoE-1B-7B (random weights, forward and backward only)
    python3 train.py --mode bench --model full

`--bind 127.0.0.N` makes this island's gloo pairs and torchft manager use that loopback address,
so that two islands on one machine talk over a link that tc can shape (see shape.sh).
Metrics go to $POOL_OUTPUTS/metrics.json.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OlmoeConfig, OlmoeForCausalLM

# OLMoE-1B-7B-0924 (config.json on the hub) and the same architecture, scaled down to fit two
# islands and a baseline on one GB10 in minutes: 64 experts, 8 active, its tokenizer and vocab.
FULL = dict(hidden_size=2048, intermediate_size=1024, num_hidden_layers=16, num_attention_heads=16,
            num_key_value_heads=16)
SMALL = dict(hidden_size=512, intermediate_size=256, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=8)
TINY = dict(hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)  # smoke tests
COMMON = dict(vocab_size=50304, num_experts=64, num_experts_per_tok=8, norm_topk_prob=False, max_position_embeddings=4096,
              rope_theta=10000.0, router_aux_loss_coef=0.01, tie_word_embeddings=False, pad_token_id=1, eos_token_id=50279)


def model_config(name: str) -> OlmoeConfig:
    return OlmoeConfig(**COMMON, **{"full": FULL, "small": SMALL, "tiny": TINY}[name], output_router_logits=True)


def bind_gloo(addr: str) -> None:
    """torchft's gloo groups bind this address (gloo otherwise picks the hostname's address)."""
    import torchft.process_group as tpg

    base = tpg.BaseProcessGroupGloo

    def gloo(store, rank, size, timeout=timedelta(seconds=1800)):
        options = base._Options()
        options._timeout = timeout
        options._devices = [base.create_device(hostname=addr)]
        return base(store, rank, size, options)
    tpg.BaseProcessGroupGloo = gloo


class Batches:
    def __init__(self, path: str, batch: int, seq: int, seed: int, device):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch, self.seq, self.device = batch, seq, device
        self.rng = np.random.default_rng(seed)

    def next(self):
        starts = self.rng.integers(0, len(self.data) - self.seq - 1, self.batch)
        window = torch.from_numpy(np.stack([self.data[s: s + self.seq + 1].astype(np.int64) for s in starts]))
        window = window.to(self.device, non_blocking=True)
        return window[:, :-1], window[:, 1:]


def loss_of(model, x, y, aux: bool = True):
    out = model(input_ids=x)
    loss = F.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), y.reshape(-1))
    if aux and out.aux_loss is not None:
        return loss, loss + model.config.router_aux_loss_coef * out.aux_loss
    return loss, loss


@torch.no_grad()
def evaluate(model, path: str, seq: int, device, n: int = 24, batch: int = 8) -> float:
    model.eval()
    it = Batches(path, batch, seq, 1234, device)  # the same held-out windows every time
    total = 0.0
    for _ in range(n):
        x, y = it.next()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total += loss_of(model, x, y, aux=False)[0].item()
    model.train()
    return total / n


def fragments(model: OlmoeForCausalLM, k: int) -> list[nn.Module]:
    """Split the model into k fragments for streaming DiLoCo (each synced at its own step)."""
    if k == 1:
        return [model]
    layers = list(model.model.layers)
    size = math.ceil(len(layers) / k)
    groups = [nn.ModuleList(layers[i * size:(i + 1) * size]) for i in range(k)]
    groups[0].append(model.model.embed_tokens)
    groups[-1].extend([model.model.norm, model.lm_head])
    return groups


def bench(args, device) -> dict:
    """Forward and backward time of one step (bf16 weights, no optimizer): for projecting the full model."""
    torch.manual_seed(0)
    model = OlmoeForCausalLM(model_config(args.model)).to(device, torch.bfloat16)
    params = sum(p.numel() for p in model.parameters())
    it = Batches(os.path.join(args.data, "train.bin"), args.batch, args.seq, 0, device)
    times = []
    for step in range(args.bench_steps):
        x, y = it.next()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = loss_of(model, x, y)[1]
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        model.zero_grad(set_to_none=True)
    step_s = statistics.median(times[2:])
    return {"mode": "bench", "model": args.model, "params": params, "batch": args.batch, "seq": args.seq,
            "step_s": round(step_s, 3), "tokens_per_s": round(args.batch * args.seq / step_s),
            "sync_bytes_bf16": params * 2, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["island", "single", "bench"], required=True)
    parser.add_argument("--model", choices=["small", "full", "tiny"], default="small")
    parser.add_argument("--island", type=int, default=0)
    parser.add_argument("--islands", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10, help="outer steps")
    parser.add_argument("--inner-steps", type=int, default=50, help="inner steps per outer step (H)")
    parser.add_argument("--batch", type=int, default=8, help="sequences per step per island")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.0)
    parser.add_argument("--fragments", type=int, default=1, help="streaming DiLoCo: fragments synced in turn")
    parser.add_argument("--sync-delay", type=int, default=0, help="streaming DiLoCo: inner steps a fragment's sync overlaps")
    parser.add_argument("--quantize", action="store_true", help="torchft's fp8 all-reduce of the pseudo-gradients")
    parser.add_argument("--ref", default="c1/olmoe", help="the pool ref whose rounds record the outer steps")
    parser.add_argument("--bind", help="this island's address for gloo and the torchft manager (127.0.0.N)")
    parser.add_argument("--lighthouse-port", type=int, default=29510)
    parser.add_argument("--deadline", type=float, default=1800)
    parser.add_argument("--link", default="", help="a label for the link, for the metrics (100mbit, say)")
    parser.add_argument("--bench-steps", type=int, default=8)
    parser.add_argument("--data", default=os.path.join(os.environ.get("POOL_INPUTS", "."), "data"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":  # a CPU smoke test: one thread per CPU the container gets
        torch.set_num_threads(max(1, min(4, len(os.sched_getaffinity(0)))))
    out_dir = os.environ.get("POOL_OUTPUTS", ".")
    if args.mode == "bench":
        result = bench(args, device)
        print(json.dumps(result), flush=True)
        json.dump(result, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
        return

    torch.manual_seed(0)  # the same initial weights on every island
    model = OlmoeForCausalLM(model_config(args.model)).to(device)
    params = [p for p in model.parameters()]
    count = sum(p.numel() for p in params)
    inner = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.rounds * args.inner_steps
    schedule = torch.optim.lr_scheduler.LambdaLR(inner, lambda s: min(1.0, (s + 1) / args.warmup))
    batch = args.batch * (args.islands if args.mode == "single" else 1)
    it = Batches(os.path.join(args.data, "train.bin"), batch, args.seq, 1000 + args.island, device)
    val = os.path.join(args.data, "val.bin")
    who = f"island {args.island}" if args.mode == "island" else "single island"
    print(f"{who}: OLMoE-{args.model} {count / 1e6:.0f}M parameters, batch {batch}x{args.seq}, {total_steps} steps, "
          f"torch {torch.__version__}", flush=True)

    rounds = manager = diloco = None
    if args.mode == "island":
        sys.path.insert(0, os.environ.get("POOL_LIB", "/run/pool"))
        import pool_reducers
        from torchft import ProcessGroupGloo
        from torchft.local_sgd import DiLoCo

        if args.bind:
            bind_gloo(args.bind)
        rounds = pool_reducers.TorchftRounds(args.ref, quorum=args.islands, deadline_s=args.deadline, grace_s=1,
                                             host=args.bind, port=args.lighthouse_port, replica_id=f"island{args.island}")
        print(f"{who}: round on {args.ref} v{rounds.version}, lighthouse {rounds.lighthouse}"
              + (" (hosted here)" if rounds.hosting else ""), flush=True)
        outer = [torch.optim.SGD(f.parameters(), lr=args.outer_lr, momentum=args.outer_momentum,
                                 nesterov=args.outer_momentum > 0) for f in fragments(model, args.fragments)]

        def state_dict():
            return {"model": model.state_dict(), "inner": inner.state_dict()}

        def load_state_dict(state):
            model.load_state_dict(state["model"])
            inner.load_state_dict(state["inner"])
        manager = rounds.manager(ProcessGroupGloo(timeout=timedelta(seconds=args.deadline)), state_dict=state_dict,
                                 load_state_dict=load_state_dict, init_sync=False)
        diloco = DiLoCo(manager, fragments(model, args.fragments), inner, outer if args.fragments > 1 else outer[0],
                        backup_device=device, sync_every=args.inner_steps, fragment_sync_delay=args.sync_delay,
                        should_quantize=args.quantize)
        diloco.__enter__()

    history, step_times, sync_steps, eval_s = [], [], [], 0.0
    tokens_here, last_tokens = 0, 0
    first = evaluate(model, val, args.seq, device)
    history.append({"round": 0, "tokens": 0, "val_loss": round(first, 4)})
    print(f"round 0: val loss {first:.4f}", flush=True)
    started = time.perf_counter()
    losses = []
    for step in range(1, total_steps + 1):
        t0 = time.perf_counter()
        x, y = it.next()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lm, loss = loss_of(model, x, y)
        inner.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        inner.step()  # with DiLoCo, every --inner-steps steps this also runs the outer step
        schedule.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        tokens_here += batch * args.seq
        if step % 10 == 0:
            losses.append(lm.item())
        synced = rounds.synced(manager) if rounds else False
        if synced:
            sync_steps.append(step)
            rounds.record(manager, weight=tokens_here - last_tokens, meta={"train_loss": losses[-1] if losses else None})
            last_tokens = tokens_here
        if step % args.inner_steps == 0:
            e0 = time.perf_counter()
            val_loss = evaluate(model, val, args.seq, device)
            eval_s += time.perf_counter() - e0
            line = {"round": step // args.inner_steps, "step": step, "tokens": tokens_here * (args.islands if rounds else 1),
                    "elapsed_s": round(time.perf_counter() - started, 1), "val_loss": round(val_loss, 4),
                    "train_loss": round(statistics.mean(losses[-5:]), 4) if losses else None,
                    "participants": manager.num_participants() if manager else 1}
            history.append(line)
            print(json.dumps(line), flush=True)

    wall = time.perf_counter() - started - eval_s
    if diloco is not None:
        diloco.__exit__(None, None, None)
    normal = [t for i, t in enumerate(step_times, 1) if i not in set(sync_steps)]
    median = statistics.median(normal)
    exposed = sum(step_times[i - 1] - median for i in sync_steps)
    result = {"mode": args.mode, "island": args.island, "islands": args.islands, "link": args.link, "params": count,
              "args": vars(args), "history": history, "final_val_loss": history[-1]["val_loss"],
              "train_s": round(wall, 1), "eval_s": round(eval_s, 1), "step_s": round(median, 4),
              "sync_s": [round(step_times[i - 1] - median, 2) for i in sync_steps],
              "utilisation": round(1 - exposed / wall, 4) if wall > 0 else None,
              "tokens_total": history[-1]["tokens"], "tokens_per_s_here": round(tokens_here / wall),
              "sync_mb_fp32": round(count * 4 / 1e6, 1)}
    if rounds:
        rounds.close()
        result["ledger_errors"] = rounds.errors
    print(f"{who}: final val loss {result['final_val_loss']:.4f}, utilisation {result['utilisation']}, "
          f"outer syncs {result['sync_s']} s", flush=True)
    json.dump(result, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
