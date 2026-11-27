"""Keeping agents on the server's version of the pool package (standard library only).

The server serves its own `pool/*.py` as the agent package (server.package_tarball) and names it by a
digest of the files: `package_digest`. Each heartbeat reply carries that digest. An agent whose own
files differ can replace them:

- by hand: `pool agent update` (then restart the agent);
- on its own, when its owner opted in with `pool agent update --auto`: the agent waits until it runs
  no jobs, installs the new files and restarts itself in place.

Auto-update is off by default: it lets whoever runs the server change code that runs on the machine
outside the sandbox, which a machine owner should agree to. An agent that runs from a source checkout
(a git repo or a folder with pyproject.toml) never overwrites itself.

Each agent reports its own digest in its profile (`agent_digest`, with `auto_update` and the last
failed self-update as `agent_update_error`), so the server can say which machines run an outdated
agent: `agent_version`.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import tarfile
from pathlib import Path

NAME_RE = re.compile(r"^pool/[A-Za-z0-9_]+\.py$")
MAX_PACKAGE = 16 * 1024**2


def package_digest(root: Path) -> str:
    """sha256 over the package's `*.py` files: names and contents, in name order."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).glob("*.py")):
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def agent_state(reported: str | None, current: str | None) -> str:
    """current, outdated, or unknown (an agent from before agents reported their digest)."""
    if not reported:
        return "unknown"
    return "current" if reported == current else "outdated"


def agent_version(profile: dict | None, current: str | None) -> dict:
    """What the server says about a machine's agent, from the profile the agent reported."""
    profile = profile or {}
    digest = profile.get("agent_digest") or None
    out = {"digest": digest, "short": digest[:12] if digest else None, "state": agent_state(digest, current)}
    if "auto_update" in profile:
        out["auto_update"] = bool(profile["auto_update"])
    if profile.get("agent_update_error"):
        out["update_error"] = profile["agent_update_error"]
    return out


def short_error(exc: BaseException, root: Path, limit: int = 200) -> str:
    """A self-update failure as the agent reports it: one line, short, and no path above the package
    root (the root's parent becomes `..`; any other absolute path becomes `<path>`)."""
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    parent = str(Path(root).resolve().parent)
    if parent != "/":
        text = text.replace(parent, "..")
    text = re.sub(r"(?<=[\s'\"(=])/[^\s'\"]+", "<path>", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def files_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256(files[name]).digest())
    return digest.hexdigest()


def read_package(data: bytes) -> dict[str, bytes]:
    """The files of a package tarball, refusing anything but plain `pool/<name>.py` files."""
    if len(data) > MAX_PACKAGE:
        raise ValueError("the agent package is larger than 16 MB")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not NAME_RE.match(member.name):
                raise ValueError(f"unexpected entry in the agent package: {member.name!r}")
            handle = tar.extractfile(member)
            files[member.name.split("/", 1)[1]] = handle.read() if handle else b""
    if "agent.py" not in files or "__init__.py" not in files:
        raise ValueError("the agent package has no pool/agent.py")
    return files


def source_checkout(root: Path) -> bool:
    """True when `root` (the pool package folder) is part of a source tree rather than an install."""
    parent = Path(root).resolve().parent
    return (parent / "pyproject.toml").exists() or (parent / ".git").exists()


def install(data: bytes, root: Path, expect: str | None = None) -> str:
    """Replace the files of the package folder `root` with those in tarball `data`; return their digest.

    The new files are written next to `root` and swapped in with two renames, so a crash leaves either
    the old or the new package, never a mix."""
    root = Path(root).resolve()
    if source_checkout(root):
        raise RuntimeError(f"{root} is a source checkout; update it with git instead")
    files = read_package(data)
    digest = files_digest(files)
    if expect and digest != expect:
        raise ValueError("the downloaded package does not match the server's digest")
    new, old = root.with_name(".pool-new"), root.with_name(".pool-old")
    for path in (new, old):
        shutil.rmtree(path, ignore_errors=True)
    new.mkdir()
    for name, content in files.items():
        (new / name).write_bytes(content)
    root.rename(old)
    new.rename(root)
    shutil.rmtree(old, ignore_errors=True)
    return digest
