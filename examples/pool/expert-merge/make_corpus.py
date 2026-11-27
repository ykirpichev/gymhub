"""Two byte-level domains from what any Linux machine has, so nothing is downloaded:

    code   Python source of the standard library
    legal  license and copyright texts under /usr/share/doc (deduplicated: many packages share a license)

    python3 make_corpus.py data/        -> data/code.bin, data/legal.bin (about 8 MB each)
"""

import hashlib
import sys
import sysconfig
from pathlib import Path

LIMIT = 8 * 1024 * 1024


def gather(paths, limit=LIMIT) -> bytes:
    seen, out, size = set(), [], 0
    for path in sorted(paths):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen or not data.strip():
            continue
        seen.add(digest)
        out.append(data)
        size += len(data)
        if size >= limit:
            break
    return b"\n".join(out)[:limit]


out = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
out.mkdir(parents=True, exist_ok=True)
stdlib = Path(sysconfig.get_paths()["stdlib"])
(out / "code.bin").write_bytes(gather(p for p in stdlib.rglob("*.py") if "test" not in p.parts))
(out / "legal.bin").write_bytes(gather(p for p in Path("/usr/share/doc").rglob("*") if p.is_file() and p.name == "copyright"))
for name in ("code", "legal"):
    print(name, (out / f"{name}.bin").stat().st_size, "bytes")
