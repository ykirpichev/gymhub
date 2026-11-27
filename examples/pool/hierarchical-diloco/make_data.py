"""Tokenize a slice of WikiText-103 with OLMoE's tokenizer into data/{train,val}.bin (uint16 token ids).

    python3 make_data.py data/ --tokens 12000000

Downloads two parquet files from the Hugging Face hub (157 MB and 0.7 MB) and OLMoE-1B-7B's
tokenizer.json (2 MB). Needs pyarrow, numpy and tokenizers (the Dockerfile here adds pyarrow to the vLLM image).
"""

import argparse
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

BASE = "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-103-raw-v1"
FILES = {"train": f"{BASE}/train/0.parquet", "val": f"{BASE}/validation/0.parquet"}
TOKENIZER = "https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/tokenizer.json"


def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".part")
        os.rename(path + ".part", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--tokens", type=int, default=12_000_000, help="at most this many training tokens")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, ".download")
    os.makedirs(cache, exist_ok=True)
    tokenizer = Tokenizer.from_file(fetch(TOKENIZER, os.path.join(cache, "tokenizer.json")))
    eos = 50279  # OLMoE-1B-7B eos_token_id (config.json)
    for split, url in FILES.items():
        lines = pq.read_table(fetch(url, os.path.join(cache, f"{split}.parquet"))).column("text").to_pylist()
        ids, limit = [], args.tokens if split == "train" else 10 ** 9
        chunk = 2000
        for start in range(0, len(lines), chunk):
            text = "".join(lines[start: start + chunk])
            for encoding in tokenizer.encode_batch([t for t in text.split("\n = ") if t.strip()]):
                ids.extend(encoding.ids)
                ids.append(eos)
            if len(ids) >= limit:
                break
        array = np.array(ids[:limit], dtype=np.uint16)
        array.tofile(os.path.join(args.out, f"{split}.bin"))
        print(f"{split}: {len(array):,} tokens")


if __name__ == "__main__":
    main()
