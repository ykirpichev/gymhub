"""Write corpus.txt: about 12 MB of the Python standard library's source, a byte-level corpus that
needs no download.

    python3 make_corpus.py corpus/
"""

import pathlib
import sys
import sysconfig

LIMIT = 20_000_000


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = pathlib.Path(sysconfig.get_paths()["stdlib"])
    data = bytearray()
    for path in sorted(lib.rglob("*.py")):
        if "site-packages" in path.parts or "test" in path.parts:
            continue
        data += path.read_bytes() + b"\n"
        if len(data) > LIMIT:
            break
    (out_dir / "corpus.txt").write_bytes(bytes(data))
    print(f"{len(data):,} bytes from {lib} into {out_dir / 'corpus.txt'}")


if __name__ == "__main__":
    main()
