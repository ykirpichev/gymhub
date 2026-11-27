"""The agent's unix socket for one running job (M3, the "unix-socket runtime").

A pool job has no network. To use refs and queues it talks to this socket, which the agent serves
from a short folder under /tmp (unix socket paths are limited to about 100 characters):

    <dir>/pool.sock     the socket; mounted into the container at /run/pool
    <dir>/pool_job.py   the job-side library (pool/job.py), importable with POOL_LIB on sys.path

Requests are `POST /api {op, args}`. Most are forwarded to the server's
/v1/agent/jobs/<job>/api, where they run as the job's owner in the job's pool. Two are done here,
because they move files:

    ref.fetch     the ref's artifact through the agent's cache (peers first, then the server),
                  copied (hard-linked when possible) to <job>/refs/<name>@<version>, which the
                  container sees under /refs
    ref.publish   a folder of the job's outputs uploaded as the job's result artifact, then
                  published with compare-and-swap
    checkpoint.save   the job's /checkpoint uploaded and recorded as its newest checkpoint
    round.contribute, round.finish   with `path`, a folder of the outputs uploaded first
    round.partial     with `path`, a tree round's island partial, held on this machine (pool/rounds.py);
                      a contribution to a tree round is held too: its chunks stay here and peers of
                      the island fetch them from this machine, never from the server
    artifact.fetch    a readable artifact (a round's contribution) under /refs
"""

import json
import os
import shutil
import socketserver
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from pool.artifacts import ArtifactError, build_manifest
from pool.client import PoolError

LIBRARY = Path(__file__).with_name("job.py")
REDUCERS = Path(__file__).with_name("reducers.py")
VERIFIERS = Path(__file__).with_name("verify.py")
MAX_BODY = 1 << 20


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def _link_tree(src: Path, dest: Path) -> None:
    def link(source, target):
        try:
            os.link(source, target)
        except OSError:  # another filesystem, or links not allowed: copy
            shutil.copy2(source, target)

    shutil.copytree(src, dest, copy_function=link, symlinks=True)


class JobSocket:
    def __init__(self, agent, job: dict, root: Path, container_paths: bool):
        self.agent, self.job = agent, job
        self.refs = root / "refs"
        self.refs.mkdir(parents=True, exist_ok=True)
        self.outputs = root / "outputs"
        self.container_paths = container_paths
        # $POOL_SOCKET_DIR: a containerized agent (deploy/pool/) needs a folder at the same path on the
        # host, since the job's container mounts it from there; its own /tmp is not the host's.
        self.dir = Path(tempfile.mkdtemp(prefix="pool-", dir=os.environ.get("POOL_SOCKET_DIR") or "/tmp"))
        shutil.copy(LIBRARY, self.dir / "pool_job.py")
        shutil.copy(REDUCERS, self.dir / "pool_reducers.py")
        shutil.copy(VERIFIERS, self.dir / "pool_verify.py")
        self.path = self.dir / "pool.sock"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if self.path != "/api" or length > MAX_BODY:
                        raise PoolError(400, "POST /api with a JSON body of at most 1 MB")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    status, reply = 200, owner.handle(str(body.get("op") or ""), body.get("args") or {})
                except PoolError as exc:
                    status, reply = (exc.status if 400 <= exc.status < 600 else 502), {"detail": exc.detail}
                except (ArtifactError, ValueError, OSError) as exc:
                    status, reply = 400, {"detail": f"{type(exc).__name__}: {exc}"}
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = _Server(str(self.path), Handler)
        os.chmod(self.path, 0o777)  # the container's uid differs from nothing here, but be explicit
        threading.Thread(target=self.server.serve_forever, daemon=True, name=f"socket-{job['id']}").start()

    @property
    def env(self) -> dict:
        if self.container_paths:
            return {"POOL_SOCKET": "/run/pool/pool.sock", "POOL_LIB": "/run/pool", "POOL_REFS": "/refs"}
        return {"POOL_SOCKET": str(self.path), "POOL_LIB": str(self.dir), "POOL_REFS": str(self.refs)}

    def forward(self, op: str, args: dict, timeout: float = 120) -> dict:
        return self.agent.client.call("POST", f"/v1/agent/jobs/{self.job['id']}/api", {"op": op, "args": args}, timeout=timeout)

    def handle(self, op: str, args: dict) -> dict:
        if op == "ref.fetch":
            return self.fetch(args)
        if op == "ref.publish":
            return self.publish(args)
        if op == "checkpoint.save":
            return self.save_checkpoint()
        if op in ("round.contribute", "round.finish", "round.partial") and args.get("path"):
            held = args.pop("held", None)
            if op == "round.partial":
                held = True
            elif op == "round.finish":
                held = False  # the result is the ref's next version: the server keeps it
            elif held is None:
                held = self.tree_round(args)
            args = {**args, "artifact": self.upload_output(args.pop("path"), held=bool(held))}
            return self.forward(op, args, timeout=300)
        if op == "artifact.fetch":
            return self.fetch_artifact(str(args.get("artifact") or ""))
        wait = float(args.get("wait") or 0)
        return self.forward(op, args, timeout=max(60.0, wait + 30))

    def fetch(self, args: dict) -> dict:
        ref = self.forward("ref.get", {"name": args.get("name"), "version": args.get("version")})["ref"]
        folder = f"{ref['name'].replace('/', '__')}@{ref['version']}"
        target = self.refs / folder
        if not target.exists():
            tree, _ = self.agent.cache.ensure(self.agent.client, ref["artifact"], self.agent.limiter,
                                              peer_sources=self.agent.peer_sources)
            staging = self.refs / f".{folder}.part"
            shutil.rmtree(staging, ignore_errors=True)
            _link_tree(tree, staging)
            staging.rename(target)
        shown = f"/refs/{folder}" if self.container_paths else str(target)
        return {"path": shown, "version": ref["version"], "artifact": ref["artifact"]}

    def tree_round(self, args: dict) -> bool:
        """Whether a contribution goes to an open tree round (and so stays on this machine)."""
        try:
            current = self.forward("round.current", {"ref": args.get("ref")})["round"]
        except PoolError:
            return False
        return bool(current and current["status"] == "open" and current["settings"].get("reduce") == "tree"
                    and args.get("round") in (None, current["id"])
                    and current["base_version"] == int(args.get("base_version") or 0))

    def upload_output(self, relative: str, held: bool = False) -> str:
        folder = (self.outputs / str(relative or "")).resolve()
        if not folder.is_relative_to(self.outputs.resolve()) or not folder.is_dir():
            raise PoolError(400, f"{relative!r} is not a folder under the job's outputs")
        manifest, sources = build_manifest(folder, follow_links=False, max_bytes=self.agent.max_output)
        if held:  # into this machine's cache, which serves it to peers; the server gets only the manifest
            self.agent.cache.adopt(manifest, folder)
            try:
                return self.agent.client.call("POST", "/v1/artifacts", {"manifest": manifest, "job": self.job["id"],
                                                                        "held": True}, timeout=300)["artifact"]
            except PoolError as exc:
                if exc.status != 409:  # 409: the server stores it already; upload as usual
                    raise
        return self.agent.client.upload_manifest(manifest, sources, job=self.job["id"])["artifact"]

    def fetch_artifact(self, artifact: str) -> dict:
        """Any artifact this machine may read for the job (a round's contributions, say), under /refs."""
        folder = "artifact-" + artifact.removeprefix("sha256:")[:24]
        target = self.refs / folder
        if not target.exists():
            tree, _ = self.agent.cache.ensure(self.agent.client, artifact, self.agent.limiter, peer_sources=self.agent.peer_sources)
            staging = self.refs / f".{folder}.part"
            shutil.rmtree(staging, ignore_errors=True)
            _link_tree(tree, staging)
            staging.rename(target)
        return {"path": f"/refs/{folder}" if self.container_paths else str(target), "artifact": artifact}

    def publish(self, args: dict) -> dict:
        relative = str(args.get("path") or "")
        folder = (self.outputs / relative).resolve()
        if not folder.is_relative_to(self.outputs.resolve()) or not folder.is_dir():
            raise PoolError(400, f"{relative!r} is not a folder under the job's outputs")
        manifest, sources = build_manifest(folder, follow_links=False, max_bytes=self.agent.max_output)
        uploaded = self.agent.client.upload_manifest(manifest, sources, job=self.job["id"])
        return self.forward("ref.publish", {"name": args.get("name"), "artifact": uploaded["artifact"],
                                            "parent": args.get("parent"), "force": bool(args.get("force")),
                                            "meta": args.get("meta")}, timeout=300)

    def save_checkpoint(self) -> dict:
        """Upload /checkpoint as this job's newest checkpoint; a later attempt starts from it."""
        folder = self.outputs.parent / "checkpoint"
        manifest, sources = build_manifest(folder, follow_links=False, max_bytes=self.agent.max_output)
        uploaded = self.agent.client.upload_manifest(manifest, sources, job=self.job["id"])
        self.forward("checkpoint.record", {"artifact": uploaded["artifact"], "attempt": self.job.get("tries")})
        return {"artifact": uploaded["artifact"], "size": uploaded["size"]}

    def close(self) -> None:
        try:
            self.server.shutdown()
            self.server.server_close()
        finally:
            shutil.rmtree(self.dir, ignore_errors=True)
