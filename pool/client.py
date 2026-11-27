"""HTTP client shared by the CLI and the agent (standard library only).

Settings live in $POOL_CONFIG_DIR (default ~/.config/pool):
  member.json   {server, token}         written by `pool login`
  agent.json    {server, token, ...}    written by `pool agent register` ($POOL_AGENT_CONFIG or
                                        --config instead: the system install keeps it in /etc/pool)
  hashes.json   chunk hashes of files already hashed (~/.cache/pool/hashes.json by default)
"""

import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pool


class PoolError(RuntimeError):
    def __init__(self, status: int, message: str, detail=None):
        super().__init__(f"HTTP {status}: {message}" if status else message)
        self.status = status
        self.message = message
        self.detail = detail if detail is not None else message  # structured, e.g. {"message", "current"} for a 409


def pairs(items: list[str] | None, flag: str) -> dict[str, str]:
    """K=V command-line values as a dict, or a readable error."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, not {item!r}")
        out[key] = value
    return out


def config_dir() -> Path:
    return Path(os.environ.get("POOL_CONFIG_DIR") or Path.home() / ".config" / "pool")


def save_config(name: str, data: dict) -> Path:
    return write_config(config_dir() / name, data)


def write_config(path: Path, data: dict) -> Path:
    """Write settings to `path` atomically, readable by this user only (0600)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.stat() if path.exists() else None
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    if old and hasattr(os, "geteuid") and os.geteuid() == 0 and (old.st_uid, old.st_gid) != (0, 0):
        os.chown(tmp, old.st_uid, old.st_gid)  # root (sudo pool-agent ...) keeps the agent's own file readable to it
    tmp.replace(path)
    return path


def load_config(name: str) -> dict:
    path = config_dir() / name
    return json.loads(path.read_text()) if path.is_file() else {}


class Client:
    def __init__(self, server: str, token: str | None = None):
        self.server = server.rstrip("/")
        self.token = token

    @classmethod
    def member(cls, server: str | None = None, token: str | None = None) -> "Client":
        saved = load_config("member.json")
        server = server or os.environ.get("POOL_SERVER") or saved.get("server")
        token = token or os.environ.get("POOL_TOKEN") or saved.get("token")
        if not server or not token:
            raise PoolError(0, "not logged in: pool login <server> --code <invite code>")
        return cls(server, token)

    def _request(self, method: str, path: str, data=None, headers: dict | None = None, timeout: float = 60):
        head = {"User-Agent": f"pool/{pool.__version__}", **(headers or {})}
        if self.token:
            head["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.server + path, data=data, method=method, headers=head)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode(errors="replace")
            try:
                detail = json.loads(text).get("detail", text)
            except (json.JSONDecodeError, AttributeError):
                detail = text[:300]
            message = detail.get("message", detail) if isinstance(detail, dict) else detail
            raise PoolError(exc.code, str(message), detail) from None
        except urllib.error.URLError as exc:
            raise PoolError(0, f"cannot reach {self.server}: {exc.reason}") from None
        except (TimeoutError, ConnectionError) as exc:
            raise PoolError(0, f"cannot reach {self.server}: {exc}") from None

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        with self._request(method, path, data, {"Content-Type": "application/json"}, timeout) as reply:
            raw = reply.read()
        return json.loads(raw) if raw else {}

    # ---- artifacts --------------------------------------------------------------------------------

    def upload_manifest(self, manifest: dict, sources: dict, *, pool: str | None = None, kind: str = "data",
                        job: str | None = None, threads: int = 4, say=None) -> dict:
        """Declare an artifact, send the chunks the server asks for, and return
        {"artifact", "size", "sent_bytes"}. `sources` maps chunk ids to (path, offset, length)."""
        from pool.artifacts import read_chunk

        body = {"manifest": manifest, "kind": kind, "pool": pool, "job": job}
        reply = self.call("POST", "/v1/artifacts", body, timeout=300)
        missing = reply["missing"]
        sent = 0
        if missing:
            total = sum(sources[sha][2] for sha in missing)
            if say:
                say(f"uploading {len(missing)} chunks, {total / 1024**2:.1f} MB")
            query = urllib.parse.urlencode({k: v for k, v in (("pool", pool), ("job", job)) if v})

            def send(sha: str) -> int:
                data = read_chunk(sources[sha])
                with self._request("PUT", f"/v1/chunks/{sha}?{query}", data,
                                   {"Content-Type": "application/octet-stream"}, timeout=900) as answer:
                    answer.read()
                return len(data)

            with ThreadPoolExecutor(max_workers=threads) as executor:
                sent = sum(executor.map(send, missing))
            reply = self.call("POST", "/v1/artifacts", body, timeout=300)
            if reply["missing"]:
                raise PoolError(0, f"the server still misses {len(reply['missing'])} chunks after the upload")
        return {"artifact": reply["artifact"], "size": reply["size"], "sent_bytes": sent}

    def upload_folder(self, folder: Path, *, pool: str | None = None, kind: str = "data", job: str | None = None,
                      follow_links: bool = True, max_bytes: int | None = None, say=None) -> dict:
        from pool.artifacts import HashCache, build_manifest

        cache = HashCache(config_dir().parent.parent / ".cache" / "pool" / "hashes.json" if not os.environ.get("POOL_CONFIG_DIR")
                          else config_dir() / "hashes.json")
        manifest, sources = build_manifest(Path(folder), follow_links=follow_links, cache=cache, max_bytes=max_bytes)
        return self.upload_manifest(manifest, sources, pool=pool, kind=kind, job=job, say=say)

    def manifest(self, artifact: str) -> dict:
        return self.call("GET", f"/v1/artifacts/{artifact}")["manifest"]

    def fetch_chunk(self, sha: str, artifact: str, dest: Path, limiter: "RateLimiter | None" = None) -> int:
        """Download one chunk into `dest` (a path), checking its hash."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".{os.getpid()}.{threading.get_ident()}.part")
        digest = hashlib.sha256()
        size = 0
        try:
            with self._request("GET", f"/v1/chunks/{sha}?artifact={urllib.parse.quote(artifact)}", timeout=900) as reply, \
                    open(tmp, "wb") as handle:
                while True:
                    block = reply.read(1 << 20)
                    if not block:
                        break
                    if limiter:
                        limiter.take(len(block))
                    digest.update(block)
                    handle.write(block)
                    size += len(block)
            if digest.hexdigest() != sha:
                raise PoolError(0, f"chunk {sha[:12]} arrived corrupt")
            os.chmod(tmp, 0o444)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return size

    def download_folder(self, artifact: str, dest: Path, threads: int = 4) -> int:
        """An artifact's files into `dest` (for pool get). Returns the number of files."""
        from pool.artifacts import ChunkStore, chunks_of, materialize

        manifest = self.manifest(artifact)
        with tempfile.TemporaryDirectory(dir=Path(dest).parent if Path(dest).parent.exists() else None) as tmp:
            store = ChunkStore(Path(tmp))
            with ThreadPoolExecutor(max_workers=threads) as executor:
                list(executor.map(lambda sha: self.fetch_chunk(sha, artifact, store.path(sha)), chunks_of(manifest)))
            materialize(manifest, store, Path(dest))
        return len(manifest["files"])


class RateLimiter:
    """A token bucket shared by downloads: simulates (or politely caps) a slow link."""

    def __init__(self, mbps: float):
        self.rate = mbps * 1e6 / 8
        self.allowance = self.rate
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def take(self, size: int) -> None:
        while True:
            with self.lock:
                current = time.monotonic()
                self.allowance = min(self.rate, self.allowance + (current - self.last) * self.rate)
                self.last = current
                if self.allowance >= size or self.allowance >= self.rate:
                    self.allowance -= size
                    return
                wait = (size - self.allowance) / self.rate
            time.sleep(min(wait, 0.5))
