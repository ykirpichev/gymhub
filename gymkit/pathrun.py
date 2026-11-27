"""Run a gym command for whatever exercise file is open in the editor.

    python -m gymkit.pathrun run   path/to/solutions/<exercise>/<file>     (every case + outputs, not recorded)
    python -m gymkit.pathrun test  path/to/solutions/<exercise>/<file>     (submit: recorded)
    python -m gymkit.pathrun bench path/to/gyms/algo/solutions/<exercise>/solution.cpp

Works for every gym. gyms/gpu is the GPU gym; gyms/<id>/solutions/... is a gymkit gym.
A file still at solutions/ or problems/ in the repo root is treated as the GPU gym too.
The language comes from the file name. VS Code tasks call this with ${file}.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GPU_FILES = {"solution.cu": "cuda", "solution.py": "pytorch", "triton_solution.py": "triton"}


def resolve(file: Path) -> tuple[list[str], str, str]:
    """Return (command prefix, exercise, language)."""
    rel = file.resolve().relative_to(REPO)
    parts = rel.parts
    if len(parts) >= 3 and parts[0] in {"solutions", "problems"}:
        lang = GPU_FILES.get(parts[-1])
        if lang is None:
            raise SystemExit(f"{parts[-1]} is not a GPU gym solution file")
        return [sys.executable, "-m", "gpu_gym"], parts[1], lang
    if len(parts) >= 5 and parts[0] == "gyms" and parts[2] in {"solutions", "problems"}:
        gym = parts[1]
        config = json.loads((REPO / "gyms" / gym / "gym.json").read_text())
        if config.get("runtime") == "gpu_gym":
            lang = GPU_FILES.get(parts[-1])
            if lang is None:
                raise SystemExit(f"{parts[-1]} is not a GPU gym solution file")
            return [sys.executable, "-m", "gpu_gym"], parts[3], lang
        langs = {item["file"]: item["id"] for item in config["langs"]}
        lang = langs.get(parts[-1])
        if lang is None:
            raise SystemExit(f"{parts[-1]} is not a {gym} gym solution file. Expected one of {', '.join(langs)}")
        return [sys.executable, "-m", "gymkit", gym], parts[3], lang
    raise SystemExit(f"{rel} is not inside an exercise. Open a file under solutions/ or gyms/<gym>/solutions/.")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    action, file = sys.argv[1], Path(sys.argv[2])
    prefix, slug, lang = resolve(file)
    if action == "hint":
        command = prefix + ["hint", slug]
    elif action in {"run", "test", "bench"}:
        command = prefix + [action, slug, "--lang", lang]
    else:
        raise SystemExit(f"unknown action {action!r}: use run, test, bench, or hint")
    return subprocess.call(command, cwd=REPO)


if __name__ == "__main__":
    sys.exit(main())
