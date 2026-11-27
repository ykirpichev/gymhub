"""Practise in your own editor: sync exercises between the board and a local folder.

    python -m gymhub.cli login --board https://<board> --token gp_...    once (stored in ~/.config/gymhub/cli.json)
    python -m gymhub.cli pull <gym> [--dir DIR] [--force]                exercises into DIR/<gym>/<exercise>/
    python -m gymhub.cli push [DIR] [--force]                            save changed files to the board
    python -m gymhub.cli watch [DIR] [--force]                           push on every save
    python -m gymhub.cli run <file> | submit <file>                      run or submit one file, output here

The folder keeps .gymhub.json: which file is which exercise and language, and the content last synced,
so push only sends real changes and pull never overwrites edits you have not pushed. What you save
here shows on the board, and the board's code comes back with the next pull.

Each file also remembers the revision the board gave it. A push names that revision; if the file was
changed on the board since (the browser, another machine), the board refuses the save, nothing is
overwritten on either side, and --force replaces the board's copy on purpose.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CONFIG = Path.home() / ".config" / "gymhub" / "cli.json"
META = ".gymhub.json"


def credentials(board: str | None = None, token: str | None = None) -> tuple[str, str]:
    saved = json.loads(CONFIG.read_text()) if CONFIG.is_file() else {}
    board = board or os.environ.get("GYMHUB_BOARD") or saved.get("board")
    token = token or os.environ.get("GYMHUB_TOKEN") or saved.get("token")
    if not board or not token:
        raise SystemExit("Not logged in: python -m gymhub.cli login --board https://<board> --token gp_... "
                         "(create the token on the board's Gyms page)")
    return board.rstrip("/"), token


class Conflict(SystemExit):
    """The board refused a save because the file changed there since it was pulled."""


def request(board: str, token: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(board + path, data=data, method=method, headers={
        "Authorization": f"Bearer {token}", "X-Gymhub": "1", "Content-Type": "application/json", "User-Agent": "gymhub-cli"})
    try:
        with urllib.request.urlopen(req, timeout=60) as reply:
            return json.loads(reply.read() or b"{}")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        try:
            detail = json.loads(text).get("detail")
        except json.JSONDecodeError:
            detail = text[:300]
        if isinstance(detail, dict):
            detail = detail.get("message")
        raise (Conflict if exc.code == 409 else SystemExit)(f"HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {board}: {exc.reason}") from None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_meta(root: Path) -> dict:
    return json.loads((root / META).read_text())


def _save_meta(root: Path, meta: dict) -> None:
    (root / META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _find_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / META).is_file():
            return candidate
    raise SystemExit(f"{path} is not inside a pulled gym folder (no {META}); run pull first")


# ---- commands ------------------------------------------------------------------------------------


def login(args) -> int:
    board, token = args.board.rstrip("/"), args.token
    request(board, token, "GET", "/api/gym-versions")  # proves the token works
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"board": board, "token": token}) + "\n")
    os.chmod(CONFIG, 0o600)
    print(f"Logged in to {board}. The token is stored in {CONFIG}.")
    return 0


def pull(args) -> int:
    board, token = credentials(args.board, args.token)
    gym = request(board, token, "GET", f"/api/gyms/{urllib.parse.quote(args.gym)}")
    root = Path(args.dir or ".").resolve() / args.gym
    root.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(root) if (root / META).is_file() else {"board": board, "gym": args.gym, "files": {}}
    files = {lang["id"]: lang["file"] for lang in gym["gym"].get("langs") or []}
    written = kept = 0
    for item in gym["problems"]:
        if (item.get("kind") or "code") != "code":
            continue
        detail = request(board, token, "GET", f"/api/gyms/{urllib.parse.quote(args.gym)}/problems/{urllib.parse.quote(item['slug'])}")
        problem = detail["problem"]
        folder = root / problem["slug"]
        folder.mkdir(exist_ok=True)
        (folder / "PROBLEM.md").write_text(problem.get("statement") or "")
        for lang, code in (problem.get("current") or {}).items():
            if lang not in files:
                continue
            relative = f"{problem['slug']}/{files[lang]}"
            path = root / relative
            known = meta["files"].get(relative)
            if path.is_file() and not args.force and known and _digest(path.read_text()) != known["hash"]:
                kept += 1
                print(f"  kept {relative}: it has changes you have not pushed (pull --force replaces them)")
                continue
            path.write_text(code)
            meta["files"][relative] = {"slug": problem["slug"], "lang": lang, "hash": _digest(code),
                                       "rev": (problem.get("revisions") or {}).get(lang)}
            written += 1
    _save_meta(root, meta)
    print(f"{args.gym}: {written} files in {root}" + (f", {kept} kept" if kept else ""))
    return 0


def _changed(root: Path, meta: dict) -> list[tuple[str, dict, str]]:
    changed = []
    for relative, info in sorted(meta["files"].items()):
        path = root / relative
        if path.is_file():
            text = path.read_text()
            if _digest(text) != info["hash"]:
                changed.append((relative, info, text))
    return changed


def _push_root(root: Path, board: str, token: str, quiet: bool = False, force: bool = False,
               conflicts: dict[str, str] | None = None) -> int:
    """Save every changed file; returns how many were saved.

    Each save names the revision the file was pulled at, so the board refuses it if the file
    changed there since: both copies are left alone and the file is listed in `conflicts`
    (relative path -> local digest, so a watcher reports it once, not every second). `force`
    saves without the check, replacing what the board has.
    """
    meta = _load_meta(root)
    changed = _changed(root, meta)
    pushed = 0
    for relative, info, text in changed:
        if conflicts is not None and conflicts.get(relative) == _digest(text):
            continue  # already reported; waits for another edit, a pull, or --force
        body = {"gym": meta["gym"], "slug": info["slug"], "lang": info["lang"], "action": "save", "code": text}
        if info.get("rev") and not force:
            body["base"] = info["rev"]
        try:
            reply = request(board, token, "POST", "/api/jobs", body)
        except Conflict as exc:
            print(f"  conflict {relative}: {exc.code}")
            print(f"    your file is unchanged; the board's copy is too. `pull --force` takes the board's, `push --force` takes yours.")
            if conflicts is not None:
                conflicts[relative] = _digest(text)
            else:
                conflicts = {relative: _digest(text)}
            continue
        info["hash"] = _digest(text)
        info["rev"] = reply.get("revision")
        pushed += 1
        print(f"  pushed {relative}")
    if pushed:
        _save_meta(root, meta)
    elif not changed and not quiet:
        print("Nothing to push.")
    return pushed


def push(args) -> int:
    board, token = credentials(args.board, args.token)
    conflicts: dict[str, str] = {}
    _push_root(_find_root(Path(args.dir or ".").resolve()), board, token, force=args.force, conflicts=conflicts)
    return 1 if conflicts else 0


def watch(args) -> int:
    board, token = credentials(args.board, args.token)
    root = _find_root(Path(args.dir or ".").resolve())
    print(f"Watching {root}: every save is pushed to the board. Ctrl+C stops.")
    conflicts: dict[str, str] = {}
    try:
        while True:
            try:
                _push_root(root, board, token, quiet=True, force=args.force, conflicts=conflicts)
            except SystemExit as exc:  # a failed push is reported, and retried on the next change
                print(f"  {exc}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def _run_file(args, action: str) -> int:
    board, token = credentials(args.board, args.token)
    path = Path(args.file).resolve()
    root = _find_root(path.parent)
    meta = _load_meta(root)
    relative = str(path.relative_to(root))
    info = meta["files"].get(relative)
    if info is None:
        raise SystemExit(f"{relative} is not an exercise file of {meta['gym']}")
    code = path.read_text()
    body = {"gym": meta["gym"], "slug": info["slug"], "lang": info["lang"], "action": action, "code": code}
    if action == "test" and info.get("rev") and not getattr(args, "force", False):
        body["base"] = info["rev"]  # a submission also saves the file, so it must not replace a newer one
    job = request(board, token, "POST", "/api/jobs", body)
    if job.get("warning"):
        print(f"note: {job['warning']}", file=sys.stderr)
    printed = 0
    while True:
        state = request(board, token, "GET", f"/api/jobs/{job['id']}")
        output = state.get("output") or ""
        if len(output) > printed:
            sys.stdout.write(output[printed:])
            sys.stdout.flush()
            printed = len(output)
        if state.get("status") in {"done", "error", "cancelled"}:
            break
        time.sleep(1)
    if action == "test":  # a submission also saves the file on the board
        info["hash"] = _digest(code)
        info["rev"] = job.get("revision")
        _save_meta(root, meta)
    return 0 if state.get("ok") else 1


def run(args) -> int:
    return _run_file(args, "run")


def submit(args) -> int:
    return _run_file(args, "test")


def register(commands) -> None:
    """Add the sync commands to gymhub.cli's argument parser."""
    def common(parser):
        parser.add_argument("--board")
        parser.add_argument("--token")
        return parser

    login_parser = commands.add_parser("login", help="store the board URL and your API token")
    login_parser.add_argument("--board", required=True)
    login_parser.add_argument("--token", required=True)
    login_parser.set_defaults(func=login)
    pull_parser = common(commands.add_parser("pull", help="exercises of a gym into a local folder"))
    pull_parser.add_argument("gym")
    pull_parser.add_argument("--dir")
    pull_parser.add_argument("--force", action="store_true", help="replace local changes")
    pull_parser.set_defaults(func=pull)
    push_parser = common(commands.add_parser("push", help="save changed files to the board"))
    push_parser.add_argument("dir", nargs="?")
    push_parser.add_argument("--force", action="store_true", help="replace the board's copy even if it changed there")
    push_parser.set_defaults(func=push)
    watch_parser = common(commands.add_parser("watch", help="push on every save"))
    watch_parser.add_argument("dir", nargs="?")
    watch_parser.add_argument("--force", action="store_true", help="replace the board's copy even if it changed there")
    watch_parser.add_argument("--interval", type=float, default=1.0)
    watch_parser.set_defaults(func=watch)
    for name, func, text in (("run", run, "run a file on every case and show the output; nothing recorded"),
                             ("submit", submit, "submit a file: full tests, progress recorded")):
        parser = common(commands.add_parser(name, help=text))
        parser.add_argument("file")
        if name == "submit":
            parser.add_argument("--force", action="store_true", help="save over a newer copy on the board")
        parser.set_defaults(func=func)
