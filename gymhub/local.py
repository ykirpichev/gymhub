"""The board and a runner on this machine, in one command: `gymhub-board` (or `python -m gymhub.local`).

    gymhub-board                  # http://localhost:8080, jobs graded on this machine's GPU
    gymhub-board --port 9000
    gymhub-board --check          # only print what this machine can grade, then exit
    gymhub-board --lan            # for a group: name-and-password sign-in, listening on every address
    gymhub-board --lan --owner-link   # a new link for the owner's account (first sign-in, or a lost password)

It is the "Run it all locally" recipe from docs/self-hosting.md with its defaults filled in: a SQLite
store under .cache/gymhub/, no sign-in (GYMHUB_AUTH=dev), and a runner in direct mode that grades your
jobs here (GYMHUB_TRUST_JOBS=1). Variables you set yourself win. With no sign-in anyone who can reach
the port is the admin, so the board listens on 127.0.0.1 only.

With --lan (or GYMHUB_AUTH=local) people sign in with a name and a password the board keeps, so it may
listen on a LAN or Tailscale address (--lan listens on 0.0.0.0). The owner (--owner, default your login
name) gets a one-time link that creates their admin account; everyone else joins by an invite link
from the Admin page. The session key is kept beside the SQLite store, in session.key. Other members' code is not
run on this machine unless GYMHUB_TRUST_JOBS=1 or a compute pool grades it (docs/self-hosting.md).

Before starting it says what this machine can grade: the GPU, whether PyTorch can use it, the CUDA
toolkit (nvcc) and Triton, each with the fix when it is missing.
"""

import argparse
import getpass
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

# Runs in a subprocess so this process never imports torch (a broken CUDA install cannot take it down).
_TORCH_PROBE = r"""
import json
out = {}
try:
    import torch
except ImportError:
    out["torch"] = None
else:
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    try:
        out["sees_gpu"] = bool(torch.cuda.is_available())
        if out["sees_gpu"]:
            out["device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        out["sees_gpu"], out["error"] = False, f"{type(exc).__name__}: {exc}"
try:
    import triton
    out["triton"] = triton.__version__
except Exception:
    out["triton"] = None
print(json.dumps(out))
"""


def _torch_facts() -> dict:
    import json

    try:
        done = subprocess.run([sys.executable, "-c", _TORCH_PROBE], capture_output=True, text=True, timeout=120)
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"torch": None, "triton": None, "error": "could not run the PyTorch check"}


def readiness(machine: dict | None = None, torch: dict | None = None) -> tuple[list[str], bool]:
    """Lines saying what this machine can grade, and whether every GPU exercise language runs here."""
    if machine is None:
        from gymhub.runner.capabilities import probe

        machine = probe()
    torch = _torch_facts() if torch is None else torch
    lines, ready = [], True
    gpus = machine.get("gpus") or []
    if gpus:
        names = ", ".join(gpu["name"] for gpu in gpus)
        lines.append(f"  ok    GPU          {names} (driver {machine.get('driver') or '?'})")
    else:
        lines.append("  --    GPU          none found (nvidia-smi): the GPU and LLM gyms list their exercises, "
                     "runs wait for a machine with an NVIDIA GPU")
        ready = False
    if torch.get("torch") is None:
        lines.append("  --    PyTorch      not installed: run ./setup.sh (the GPU and LLM gyms need it)")
        ready = False
    elif gpus and not torch.get("sees_gpu"):
        build = f"CUDA {torch['torch_cuda']} build" if torch.get("torch_cuda") else "CPU-only build"
        lines.append(f"  !!    PyTorch      {torch['torch']} ({build}) cannot use the GPU"
                     + (f": {torch['error']}" if torch.get("error") else "")
                     + ". Run ./setup.sh again, or install the build for your driver (https://pytorch.org/get-started/locally/)")
        ready = False
    else:
        where = f"on {torch['device']}" if torch.get("sees_gpu") else "CPU only"
        lines.append(f"  ok    PyTorch      {torch['torch']} ({where})")
    if machine.get("cuda"):
        lines.append(f"  ok    CUDA toolkit nvcc {machine['cuda']}")
    elif gpus:
        lines.append("  --    CUDA toolkit no nvcc: CUDA solutions wait, PyTorch and Triton run. "
                     "Install it from https://developer.nvidia.com/cuda-downloads")
        ready = False
    if gpus:
        if torch.get("triton"):
            lines.append(f"  ok    Triton       {torch['triton']}")
        else:
            lines.append("  --    Triton       not installed: Triton solutions fail (it comes with PyTorch's CUDA build on Linux)")
            ready = False
    return lines, ready


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def default_owner() -> str:
    """The owner's account name for local sign-in: the login name, made a valid board name."""
    try:
        name = getpass.getuser()
    except Exception:  # no login name (some containers)
        name = ""
    name = re.sub(r"[^a-z0-9._-]", "", name.lower())[:32].lstrip("._-")
    return name if len(name) >= 2 else "admin"


def _session_key(path: Path) -> str:
    """The key that signs session cookies, made once and kept, so sign-ins survive a restart."""
    if path.is_file() and len(path.read_text().strip()) >= 32:
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(key + "\n")
    return key


def environment(base: dict | None = None, lan: bool = False, owner: str | None = None) -> dict:
    """The board's and runner's environment: a local SQLite store and no sign-in, unless already set.
    With `lan`, name-and-password sign-in (GYMHUB_AUTH=local) with `owner` as the admin."""
    env = dict(os.environ if base is None else base)
    env.setdefault("GYMHUB_STORE", f"sqlite:{REPO / '.cache' / 'gymhub' / 'dev.db'}")
    env.setdefault("GYMHUB_AUTH", "local" if lan else "dev")
    if env["GYMHUB_AUTH"] == "dev":
        env.setdefault("GYMHUB_TRUST_JOBS", "1")  # every request is yours, so every job is too
    if env["GYMHUB_AUTH"] == "local":
        if not env.get("GYMHUB_SESSION_KEY"):
            folder = Path(env["GYMHUB_STORE"].removeprefix("sqlite:")).parent if env["GYMHUB_STORE"].startswith("sqlite:") \
                else REPO / ".cache" / "gymhub"
            env["GYMHUB_SESSION_KEY"] = _session_key(folder / "session.key")  # beside the store
        if not env.get("GYMHUB_ADMIN_EMAILS"):
            env["GYMHUB_ADMIN_EMAILS"] = owner or default_owner()
    return env


def owner_name(env: dict) -> str:
    return env.get("GYMHUB_ADMIN_EMAILS", "").split(",")[0].strip().lower()


def owner_link(env: dict, force: bool = False) -> str | None:
    """Local sign-in: an invite code for the owner's admin account when it does not exist yet (or, with
    `force`, a password reset code for it). None when the owner already has an account."""
    from gymhub.board.auth import USERNAME_RE, local_uid
    from gymhub.store import SqliteStore, now

    owner = owner_name(env)
    if not USERNAME_RE.match(owner):
        raise SystemExit(f"gymhub-board: {owner!r} cannot be an account name (2 to 32 lowercase letters, digits, "
                         "dots, dashes, underscores). Pick one with --owner.")
    store = SqliteStore(env["GYMHUB_STORE"].removeprefix("sqlite:"))
    uid = local_uid(owner)
    exists = store.get("users", uid) is not None
    if exists and not force:
        return None
    code = secrets.token_urlsafe(18)
    doc = {"created_by": "gymhub-board", "email": owner, "uses_left": 1, "uses_total": 1, "role": "admin",
           "expires_at": now() + 86400, "created_at": now(), "note": "owner's password reset" if exists else "owner's account"}
    if exists:
        doc["reset_uid"] = uid
    store.set("invites", code, doc)
    return code


def addresses(host: str) -> list[str]:
    """Where people can open the board: this machine's LAN address when listening on every address."""
    if host not in {"0.0.0.0", "::"}:
        return ["localhost" if host in LOOPBACK else host]
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))  # picks the outgoing interface; sends nothing
            found.append(sock.getsockname()[0])
    except OSError:
        pass
    return [*found, "localhost"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gymhub-board", description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default=None, help="Address to listen on (default 127.0.0.1; 0.0.0.0 with --lan)")
    parser.add_argument("--lan", action="store_true",
                        help="Host for a group: name-and-password sign-in (GYMHUB_AUTH=local), listening on every address")
    parser.add_argument("--owner", help="Local sign-in: the owner's (admin's) account name (default: your login name)")
    parser.add_argument("--owner-link", action="store_true",
                        help="Local sign-in: print a new one-time link for the owner's account (a lost password)")
    parser.add_argument("--no-runner", action="store_true", help="Start only the board (a runner runs elsewhere)")
    parser.add_argument("--check", action="store_true", help="Print what this machine can grade and exit")
    args = parser.parse_args(argv)

    env = environment(lan=args.lan, owner=args.owner)
    if args.host is None:
        args.host = "0.0.0.0" if args.lan else "127.0.0.1"
    if env["GYMHUB_AUTH"] == "dev" and args.host not in LOOPBACK:
        print(f"gymhub-board: with no sign-in (GYMHUB_AUTH=dev) anyone who reaches {args.host}:{args.port} is the "
              "admin and can run code on this machine. Keep --host 127.0.0.1, or use --lan for name-and-password sign-in "
              "(docs/self-hosting.md).",
              file=sys.stderr)
        return 2
    if not env["GYMHUB_STORE"].startswith("sqlite:"):
        print(f"note: GYMHUB_STORE={env['GYMHUB_STORE']} (not a local SQLite store)")

    print("What this machine can grade:")
    lines, _ready = readiness()
    print("\n".join(lines))
    if args.check:
        return 0
    print()
    if not _port_free(args.host, args.port):
        print(f"gymhub-board: port {args.port} is in use (a board already running?). Try --port {args.port + 1}.",
              file=sys.stderr)
        return 1
    if env["GYMHUB_STORE"].startswith("sqlite:"):
        Path(env["GYMHUB_STORE"].removeprefix("sqlite:")).parent.mkdir(parents=True, exist_ok=True)
    where = addresses(args.host)
    if env["GYMHUB_AUTH"] == "local":
        owner = owner_name(env)
        code = owner_link(env, force=args.owner_link) if env["GYMHUB_STORE"].startswith("sqlite:") else None
        if code:
            print(f"Open this link once, within a day, to {'pick a new password for' if args.owner_link else 'create'} "
                  f"the owner's account ({owner}):\n  http://{where[0]}:{args.port}/#/invite/{code}")
        else:
            print(f"Sign in as {owner}. Lost the password? Run gymhub-board --lan --owner-link.")
        print("Invite others from the Admin page. Their runs are graded here only with GYMHUB_TRUST_JOBS=1 or on "
              "a compute pool (docs/self-hosting.md).")

    children: list[subprocess.Popen] = []
    board = subprocess.Popen([sys.executable, "-m", "uvicorn", "gymhub.board.app:app", "--host", args.host,
                              "--port", str(args.port), "--no-access-log"], cwd=REPO, env=env)
    children.append(board)
    if not args.no_runner:
        children.append(subprocess.Popen([sys.executable, "-m", "gymhub.runner"],
                                         cwd=REPO, env=env))
    urls = "  ".join(f"http://{address}:{args.port}" for address in where)
    print(f"Board: {urls}   (Ctrl+C stops the board and the runner)", flush=True)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    code = 0
    try:
        while True:
            for child in children:
                if child.poll() is not None:
                    name = "board" if child is board else "runner"
                    print(f"gymhub-board: the {name} exited with status {child.returncode}; stopping", file=sys.stderr)
                    code = child.returncode or 1
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()  # both stop cleanly on SIGTERM (a running job goes back to the queue)
        deadline = time.time() + 15
        for child in children:
            try:
                child.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                child.kill()
    return code


if __name__ == "__main__":
    sys.exit(main())
