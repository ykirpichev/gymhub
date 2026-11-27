"""Demo C2 (issue #91, FlexOlmo-style expert merging) in small: one script, one mode per pool job.

    anchor  train the whole dense model on both domains                    -> anchor.pt
    expert  from the anchor, train only the FFNs on one domain (--domain)  -> expert-<domain>.pt
    merge   an MoE: shared attention from the anchor, experts [anchor FFN, code FFN, legal FFN],
            a router trained on both domains with everything else frozen  -> merged.pt
    dense   the baseline: a dense model with the merged model's parameter count (FFN 3x as wide),
            trained on both domains for as many steps as anchor + experts + router together
    Every mode ends by evaluating on held-out bytes of each domain, into <mode>.json.

Inputs (pool inputs, or paths): --data (code.bin, legal.bin), --anchor, --experts (folders with
expert-*.pt). Outputs go to --out (POOL_OUTPUTS).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ByteLM, params

DOMAINS = ("code", "legal")
WIDTH = 1024


def load_data(folder: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out = {}
    for name in DOMAINS:
        data = torch.frombuffer(bytearray((folder / f"{name}.bin").read_bytes()), dtype=torch.uint8)
        cut = int(len(data) * 0.95)
        out[name] = (data[:cut], data[cut:])
    return out


def batch(data: dict, domains: list[str], size: int, ctx: int, generator, device, split: int = 0):
    rows = []
    for i in range(size):
        series = data[domains[i % len(domains)]][split]
        start = int(torch.randint(0, len(series) - ctx - 1, (1,), generator=generator))
        rows.append(series[start: start + ctx + 1])
    window = torch.stack(rows).long().to(device)
    return window[:, :-1], window[:, 1:]


def loss_of(model, x, y):
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        logits = model(x)
    return F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))


@torch.no_grad()
def evaluate(model, data, ctx, device, n=24, size=32) -> dict:
    model.eval()
    out = {}
    for name in DOMAINS:
        generator = torch.Generator().manual_seed(7)  # the same held-out windows for every model
        out[name] = round(sum(loss_of(model, *batch(data, [name], size, ctx, generator, device, split=1)).item()
                              for _ in range(n)) / n, 4)
    out["mean"] = round(sum(out[name] for name in DOMAINS) / len(DOMAINS), 4)
    model.train()
    return out


def train(model, trainable, data, domains, steps, lr, ctx, device, seed, label):
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0 if lr > 1e-3 else 0.01)
    generator = torch.Generator().manual_seed(seed)
    started = time.time()
    for step in range(1, steps + 1):
        x, y = batch(data, domains, 32, ctx, generator, device)
        loss = loss_of(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 250 == 0 or step == steps:
            print(f"{label} step {step}/{steps}: loss {loss.item():.4f} ({time.time() - started:.0f} s)", flush=True)


def ffn_params(model):
    return [p for block in model.blocks for p in block.experts.parameters()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["anchor", "expert", "merge", "dense"])
    parser.add_argument("--data", default=os.path.join(os.environ.get("POOL_INPUTS", "."), "data"))
    parser.add_argument("--anchor", help="folder with anchor.pt")
    parser.add_argument("--experts", nargs="*", default=[], help="folders with expert-<domain>.pt")
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--steps", type=int, default=1500, help="anchor and expert steps")
    parser.add_argument("--router-steps", type=int, default=300)
    parser.add_argument("--out", default=os.environ.get("POOL_OUTPUTS", "out"))
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(Path(args.data))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = 256
    report = {"mode": args.mode}

    if args.mode == "anchor":
        model = ByteLM(width=WIDTH).to(device)
        train(model, list(model.parameters()), data, list(DOMAINS), args.steps, 3e-4, ctx, device, 1, "anchor")
        torch.save(model.state_dict(), out / "anchor.pt")
    elif args.mode == "expert":
        model = ByteLM(width=WIDTH).to(device)
        model.load_state_dict(torch.load(Path(args.anchor) / "anchor.pt", map_location=device))
        for p in model.parameters():
            p.requires_grad_(False)
        trainable = ffn_params(model)
        for p in trainable:
            p.requires_grad_(True)
        train(model, trainable, data, [args.domain], args.steps, 3e-4, ctx, device, 2 + DOMAINS.index(args.domain), f"expert {args.domain}")
        torch.save({k: v for k, v in model.state_dict().items() if ".experts." in k}, out / f"expert-{args.domain}.pt")
        report["domain"] = args.domain
    elif args.mode == "merge":
        anchor = torch.load(Path(args.anchor) / "anchor.pt", map_location=device)
        experts = [anchor] + [torch.load(next(Path(folder).glob("expert-*.pt")), map_location=device) for folder in args.experts]
        model = ByteLM(width=WIDTH, experts=len(experts), topk=2).to(device)
        state = {k: v for k, v in anchor.items() if ".experts." not in k}
        for e, source in enumerate(experts):  # expert 0 is the anchor's own FFN
            for key, value in source.items():
                if ".experts.0." in key:
                    state[key.replace(".experts.0.", f".experts.{e}.")] = value
        missing = model.load_state_dict(state, strict=False).missing_keys
        assert all("router" in key for key in missing), missing
        for p in model.parameters():
            p.requires_grad_(False)
        router = [block.router.weight for block in model.blocks]
        for p in router:
            p.requires_grad_(True)
        report["before_router"] = evaluate(model, data, ctx, device)
        train(model, router, data, list(DOMAINS), args.router_steps, 1e-2, ctx, device, 9, "router")
        torch.save(model.state_dict(), out / "merged.pt")
    else:  # dense baseline with the merged model's size and the whole pipeline's step count
        model = ByteLM(width=WIDTH * 3).to(device)
        steps = 3 * args.steps + args.router_steps
        train(model, list(model.parameters()), data, list(DOMAINS), steps, 3e-4, ctx, device, 1, "dense")
        report["steps"] = steps
    report["params"] = params(model)
    report["eval"] = evaluate(model, data, ctx, device)
    print(json.dumps(report), flush=True)
    (out / f"{args.mode}{'-' + args.domain if args.domain else ''}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
