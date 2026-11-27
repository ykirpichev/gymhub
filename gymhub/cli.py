"""Gym authoring from a terminal or CI.

    python -m gymhub.cli gym validate <dir>                 check the package, then run the same checks a
                                                            machine runs (graders load, starters fail,
                                                            references pass) on this computer
    python -m gymhub.cli gym validate <dir> --quick         only the manifest and files
    python -m gymhub.cli gym push <dir> [--wait]            upload a new version to the board
    python -m gymhub.cli gym seed [<id> ...]                publish the repository's courses to the store
                                                            GYMHUB_STORE names (a local board; the runner
                                                            does this at start, see gymhub/courses.py)

Gym Markdown: a whole gym in one file (gymhub/GYM_MARKDOWN.md), easy for people and agents to write:
    python -m gymhub.cli gym guide                          print the language reference
    python -m gymhub.cli gym new <file.gym.md> [--id ID]    start one from a template
    python -m gymhub.cli gym check <file.gym.md> [--json]   compile and check it; errors carry line numbers
    python -m gymhub.cli gym build <file.gym.md> -o <dir>   write the package it compiles to
    python -m gymhub.cli gym export <dir> [-o file.gym.md]  any package as Gym Markdown
validate and push accept a .gym.md as well as a directory.

push needs the board's URL and a personal API token (the board's Compute page, "API tokens"):
    GYMHUB_BOARD=https://<board>  GYMHUB_TOKEN=gp_...      or --board / --token
A board behind IAP cannot be reached from outside a browser; such boards accept uploads on the
Gyms page only.

Practising from a local editor (pull, push, watch, run, submit): see gymhub/sync.py.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from gymhub import gymmd, packages


def _is_markdown(path: str) -> bool:
    return Path(path).is_file() and path.endswith(".md")


def _compiled(path: str, tmp: str) -> packages.Package:
    """The package at `path`: a directory, or a .gym.md compiled into `tmp`. Exits with its errors."""
    if not _is_markdown(path):
        return packages.load(path)
    try:
        return gymmd.build(Path(path).read_text(), Path(tmp) / "package")
    except gymmd.GymMarkdownError as exc:
        _print_errors(path, exc.errors)
        raise SystemExit(1) from None


def _print_errors(path: str, errors: list[dict]) -> None:
    """file:line: exercise: message, the form editors and agents jump to."""
    for error in errors:
        where = f"{error['exercise']}: " if error.get("exercise") else ""
        print(f"{path}:{error.get('line') or 1}: {where}{error['message']}")


def _validate(args) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        package = _compiled(args.dir, tmp)
        errors = packages.validate_upload(package)
        print(f"{package.id}: {len(package.slugs())} exercises")
        for error in errors:
            print(f"  - {error}")
        if errors:
            return 1
        if args.quick:
            print("files ok")
            return 0
        command = [sys.executable, "-m", "gymhub.runner.validate", str(package.root), "--out", str(Path(tmp) / "result.json")]
        return subprocess.run(command).returncode


def _check(args) -> int:
    """Compile and check a .gym.md without running anything. --json for agents and editors."""
    text = Path(args.file).read_text()
    gym, errors = gymmd.check(text)
    if args.json:
        exercises = [{"id": e.slug, "kind": e.meta["kind"], "line": e.line} for e in gym.exercises] if gym else []
        print(json.dumps({"ok": not errors, "gym": gym.manifest["id"] if gym else None, "exercises": exercises,
                          "errors": errors}, indent=2))
    elif errors:
        _print_errors(args.file, errors)
    else:
        kinds: dict[str, int] = {}
        for exercise in gym.exercises:
            kinds[exercise.meta["kind"]] = kinds.get(exercise.meta["kind"], 0) + 1
        summary = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
        print(f"{gym.manifest['id']}: {len(gym.exercises)} exercises ({summary}) in {len(gym.manifest['tracks'])} tracks: ok")
        if any(exercise.meta["kind"] == "code" for exercise in gym.exercises):
            print(f"code exercises: run `python -m gymhub.cli gym validate {args.file}` to run their tests on this computer")
    return 1 if errors else 0


def _build(args) -> int:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty")
        return 1
    try:
        package = gymmd.build(Path(args.file).read_text(), out)
    except gymmd.GymMarkdownError as exc:
        _print_errors(args.file, exc.errors)
        return 1
    print(f"{out}: {package.id}, {len(package.slugs())} exercises")
    return 0


def _export(args) -> int:
    try:
        text = gymmd.export(packages.load(args.dir))
    except gymmd.ExportError as exc:
        print(exc)
        return 1
    if args.out:
        Path(args.out).write_text(text)
        print(f"{args.out}: {len(text.splitlines())} lines")
    else:
        sys.stdout.write(text)
    return 0


def _new(args) -> int:
    path = Path(args.file)
    if path.exists():
        print(f"{path} exists")
        return 1
    gym_id = args.id or path.name.split(".")[0].lower().replace("_", "-")
    if not packages.GYM_ID.match(gym_id):
        print("give --id: 3-40 characters, lowercase letters, digits and -")
        return 1
    title = gym_id.replace("-", " ").capitalize()
    path.write_text(gymmd.TEMPLATE.format(id=gym_id, title=title))
    print(f"{path}: a lesson, a choice and a Python exercise to start from. The language: python -m gymhub.cli gym guide")
    return 0


def _guide(args) -> int:
    del args
    sys.stdout.write(gymmd.guide())
    return 0


def _request(board: str, token: str, method: str, path: str, data: bytes | None = None, content_type: str = "application/json"):
    request = urllib.request.Request(
        board.rstrip("/") + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "X-Gymhub": "1", "Content-Type": content_type, "User-Agent": "gymhub-cli"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            return json.loads(reply.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code in (301, 302, 303, 307):
            raise SystemExit("The board redirected to a sign-in page (IAP): upload on its Gyms page instead.") from None
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            raise SystemExit(f"HTTP {exc.code}: {body[:300]}") from None
        message = detail.get("detail")
        errors = detail.get("errors") or []
        if isinstance(message, dict):
            errors, message = message.get("errors") or [], message.get("message")
        raise SystemExit(f"HTTP {exc.code}: {message}" + "".join(f"\n  - {error}" for error in errors)) from None


def _push(args) -> int:
    from gymhub.sync import credentials

    board, token = credentials(args.board, args.token)
    with tempfile.TemporaryDirectory() as tmp:
        package = _compiled(args.dir, tmp)
        errors = packages.validate_upload(package)
        if errors:
            print("\n".join(f"  - {error}" for error in errors))
            return 1
        data = packages.pack(package.root)
    reply = _request(board, token, "POST", "/api/gym-versions", data, "application/gzip")
    version = reply["version"]
    if version.get("duplicate"):
        print(f"{package.id}: this exact content is already v{version['version']} ({version['status']})")
    else:
        print(f"{package.id}: uploaded v{version['version']}; a machine is validating it")
    if reply.get("warning"):
        print(f"note: {reply['warning']}")
    if not args.wait:
        return 0
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        versions = _request(board, token, "GET", f"/api/gym-versions?gym={package.id}")["versions"]
        current = next((item for item in versions if item["version"] == version["version"]), None)
        if current and current["status"] != "validating":
            checks = current.get("checks") or []
            print(f"v{current['version']} {current['status']}: {sum(c['ok'] for c in checks)}/{len(checks)} checks passed")
            for error in current.get("errors") or []:
                print(f"  - {error}")
            for warning in current.get("warnings") or []:
                print(f"  note: {warning}")
            return 0 if current["status"] == "published" else 1
        time.sleep(3)
    print("still validating; check the board")
    return 2


def _seed(args) -> int:
    from gymhub import courses
    from gymhub.blobs import open_blobs
    from gymhub.store import open_store

    if not os.environ.get("GYMHUB_STORE"):
        print("Set GYMHUB_STORE to the board's store, for example GYMHUB_STORE=sqlite:.cache/gymhub/dev.db")
        return 2
    known = [package.id for package in courses.courses()]
    unknown = sorted(set(args.ids) - set(known))
    if unknown:
        print(f"Not a course: {', '.join(unknown)}. The courses are {', '.join(known)}.")
        return 2
    outcomes = courses.seed(open_store(), open_blobs(), only=args.ids or None)
    for gym_id, outcome in outcomes.items():
        print(f"{gym_id}: {'already up to date' if outcome == 'skipped' else outcome}")
    return 1 if "failed" in outcomes.values() else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gymhub.cli", description="Practise from your editor; create and publish gyms.")
    groups = parser.add_subparsers(dest="group", required=True)
    gym = groups.add_parser("gym").add_subparsers(dest="command", required=True)
    validate = gym.add_parser("validate", help="check a gym package (a directory or a .gym.md) on this computer")
    validate.add_argument("dir")
    validate.add_argument("--quick", action="store_true", help="only check files, do not run graders")
    validate.set_defaults(func=_validate)
    push = gym.add_parser("push", help="upload a new version (a directory or a .gym.md) to the board")
    push.add_argument("dir")
    push.add_argument("--board")
    push.add_argument("--token")
    push.add_argument("--wait", action="store_true", help="wait for validation and exit 1 if it fails")
    push.add_argument("--timeout", type=int, default=900)
    push.set_defaults(func=_push)
    seed = gym.add_parser("seed", help="publish the repository's courses to a local board's store (GYMHUB_STORE)")
    seed.add_argument("ids", nargs="*", help="course ids (default: every course in gyms/)")
    seed.set_defaults(func=_seed)
    check = gym.add_parser("check", help="compile and check a .gym.md (nothing runs)")
    check.add_argument("file")
    check.add_argument("--json", action="store_true", help="machine-readable result, errors with line numbers")
    check.set_defaults(func=_check)
    build = gym.add_parser("build", help="write the package a .gym.md compiles to")
    build.add_argument("file")
    build.add_argument("-o", "--out", required=True)
    build.set_defaults(func=_build)
    export = gym.add_parser("export", help="write a package as Gym Markdown")
    export.add_argument("dir")
    export.add_argument("-o", "--out")
    export.set_defaults(func=_export)
    new = gym.add_parser("new", help="start a .gym.md from a template")
    new.add_argument("file")
    new.add_argument("--id")
    new.set_defaults(func=_new)
    gym.add_parser("guide", help="print the Gym Markdown reference").set_defaults(func=_guide)
    from gymhub import gymsfetch

    fetch = gym.add_parser("fetch", help="put the gyms repository in gyms/ at the commit gyms.lock pins")
    gymsfetch.add_arguments(fetch)
    fetch.set_defaults(func=gymsfetch.run)
    from gymhub import sync

    sync.register(groups)  # login, pull, push, watch, run, submit: practise in a local editor
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
