import torch


def solve(X: torch.Tensor, Y: torch.Tensor, N: int) -> None:
    torch.mul(X, 2, out=Y)
