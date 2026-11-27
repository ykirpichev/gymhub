"""Synthetic, checkable arithmetic (no download): "What is 37 * 6 + 145?" -> 367.
The answer must follow "####", so rewards rise as the model learns the format and the arithmetic."""
import random
import sys
from pathlib import Path

import pandas as pd

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def rows(n, seed, split):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b, c = rng.randint(10, 99), rng.randint(2, 9), rng.randint(100, 999)
        question = f"What is {a} * {b} + {c}?"
        out.append({
            "data_source": "arith",
            "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(a * b + c)},
            "extra_info": {"split": split, "index": i},
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows(2048, 1, "train").to_parquet(out / "train.parquet")
    rows(128, 2, "test").to_parquet(out / "test.parquet")
    print("data in", out)
