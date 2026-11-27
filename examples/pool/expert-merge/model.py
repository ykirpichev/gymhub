"""A small byte-level transformer whose feed-forward layer may be a mixture of experts.

Dense: one FFN of width `ffn` per block. MoE: `experts` FFNs per block and a router (a linear map
from the block's input to one logit per expert); the top `topk` experts by router weight run, mixed
by their softmax weights. Attention, embeddings and norms are shared, as in FlexOlmo, where experts
are trained separately against a frozen anchor and only their FFNs differ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ffn(d: int, width: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, d))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, width: int, experts: int = 1, topk: int = 1):
        super().__init__()
        self.heads, self.topk = heads, topk
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.experts = nn.ModuleList(ffn(d, width) for _ in range(experts))
        self.router = nn.Linear(d, experts, bias=False) if experts > 1 else None
        if self.router is not None:
            nn.init.zeros_(self.router.weight)  # start from an even mix

    def mix(self, h):
        if self.router is None:
            return self.experts[0](h)
        weights = F.softmax(self.router(h).float(), dim=-1)
        top, index = weights.topk(self.topk, dim=-1)
        top = (top / top.sum(-1, keepdim=True)).to(h.dtype)
        out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):  # small models: run every expert, keep the chosen
            chosen = (index == e)
            if chosen.any():
                gate = (top * chosen).sum(-1, keepdim=True)
                out = out + gate * expert(h)
        return out

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.mix(self.ln2(x))


class ByteLM(nn.Module):
    def __init__(self, layers=4, d=256, heads=4, ctx=256, width=1024, experts=1, topk=1):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, width, experts, topk) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, 256, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
