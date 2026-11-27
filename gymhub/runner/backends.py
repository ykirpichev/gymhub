"""How a runner reaches the board.

DirectBackend   talks to Firestore (or a local SQLite file) with the machine's own credentials.
                Jobs arrive instantly through a realtime listener. Used for local development
                and tests. Not what deploy/install-runner.sh installs: it can
                read and write the whole database. With no board in between to check whose job
                it is, it leases an isolated job (another user's code, an uploaded gym) only when
                it grades it on the pool, unless GYMHUB_TRUST_JOBS=1 (`trusts_jobs`).
ApiBackend      talks only to the board's worker API with this worker's token. No cloud credentials
                on the machine; the token can be revoked from the board. Polls for jobs: about every
                second while someone is using the board, every 10 s otherwise.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request

from gymhub import jobqueue, routing
from gymhub.store import Store, now, problem_id


class TokenRevoked(SystemExit):
    pass


def trusts_jobs() -> bool:
    """GYMHUB_TRUST_JOBS=1: run isolated jobs on this machine, as this user, with no isolation.
    For local development and tests only: another user's code could read anything this user can."""
    return os.environ.get("GYMHUB_TRUST_JOBS", "").strip().lower() in ("1", "true", "yes")


class DirectBackend:
    push = True
    can_publish = True

    def __init__(self, store: Store, worker: str):
        self.store = store
        self.worker = worker
        self.wake = threading.Event()
        self._unsubscribe = None
        self.leases: dict[str, str | None] = {}  # job id -> the lease this machine holds
        self.pool: dict | None = None
        self.dispatcher: dict | None = None

    def start(self) -> None:
        self._unsubscribe = self.store.watch("jobs", "status", "queued", lambda docs: docs and self.wake.set())

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
        self.store.set("runners", self.worker, {"heartbeat_at": 0, "busy": None}, merge=True)

    def heartbeat(self, info: dict) -> int:
        self.store.set("runners", self.worker, {**info, "heartbeat_at": now()})
        capabilities = info.get("capabilities") or {}
        self.pool = capabilities.get("pool")  # pool mode: what this runner grades there
        self.dispatcher = capabilities if capabilities.get("dispatch_only") else None  # runs nothing here
        return 30_000

    def publish_catalog(self, gym_id: str, meta: dict, problems: list[dict]) -> None:
        if (self.store.get("gyms", gym_id) or {}).get("source") == "bundle":
            raise RuntimeError(f"{gym_id!r} is an uploaded gym; this machine's local gym cannot replace it")
        slugs = set()
        for problem in problems:
            slugs.add(problem["slug"])
            self.store.set("problems", problem_id(gym_id, problem["slug"]), {**problem, "gym": gym_id, "updated_at": now()})
        for doc_id, _doc in self.store.where("problems", "gym", gym_id):
            if doc_id.split("__", 1)[1] not in slugs:
                self.store.delete("problems", doc_id)
        # merge: keep settings made on the board (visibility, tags)
        self.store.set("gyms", gym_id, {**meta, "id": gym_id, "problems": len(slugs), "runner": self.worker, "updated_at": now()}, merge=True)

    def publish_code(self, gym_id: str, slug: str, current: dict) -> None:
        doc_id = problem_id(gym_id, slug)
        if self.store.get("problems", doc_id) is not None:
            self.store.set("problems", doc_id, {"current": current, "updated_at": now()}, merge=True)

    def publish_progress(self, gym_id: str, data: dict) -> None:
        self.store.set("progress", gym_id, {"gym": gym_id, "data": data, "updated_at": now()})

    def lease(self, gyms: list[str]) -> dict | None:
        from gymhub.runner.capabilities import probe

        jobqueue.sweep(self.store)
        if self.dispatcher is not None:  # never routed by this machine's own hardware: it runs nothing
            capabilities = dict(self.dispatcher)
        else:
            capabilities = {**probe(), "bundles": True, **({"pool": self.pool} if self.pool else {})}
        # The worker API's rule (gymhub/board/workers.py), without an owner to exempt: an isolated
        # job this machine would run itself, not on the pool, is left for another machine.
        def unisolated(doc: dict) -> bool:
            return bool(doc.get("isolated")) and not trusts_jobs() and not (routing.for_job(capabilities, doc) or {}).get("sandbox")

        job = jobqueue.lease(self.store, gyms, self.worker, capabilities=capabilities, skip=unisolated)
        if job:
            self.leases[job["id"]] = job.get("lease_id")
        return job

    def fetch_bundle(self, job: dict) -> bytes:
        from gymhub.blobs import open_blobs

        return open_blobs().get(job["bundle"]["key"])

    def update(self, job_id: str, output: str | None, host: str | None = None) -> bool:
        return jobqueue.update(self.store, job_id, self.worker, output, self.leases.get(job_id), host=host)

    def complete(self, job_id: str, final: dict, user_progress: dict | None = None) -> None:
        jobqueue.complete(self.store, job_id, self.worker, final, user_progress, lease_id=self.leases.pop(job_id, None))

    def wait_for_work(self, timeout: float) -> None:
        self.wake.wait(timeout)
        self.wake.clear()


class ApiBackend:
    push = False
    can_publish = True  # set by the board on each heartbeat

    def __init__(self, base_url: str, token: str):
        self.base = base_url.rstrip("/")
        self.token = token
        self.poll_ms = 10_000
        self.worker = "?"
        self.leases: dict[str, str | None] = {}  # job id -> the lease this machine holds

    def _call(self, path: str, body: dict | None = None, timeout: float = 60) -> dict | None:
        data = json.dumps(body or {}).encode()
        request = urllib.request.Request(
            f"{self.base}/v1/worker/{path}", data=data, method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "User-Agent": "gymhub-runner"},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if response.status == 204:
                        return None
                    return json.loads(response.read() or b"{}")
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    detail = exc.read().decode(errors="replace")[:300]
                    raise TokenRevoked(f"The board refused this worker's token ({exc.code}): {detail}") from exc
                if exc.code < 500 and exc.code != 429:
                    raise RuntimeError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"{path}: the board did not answer")

    def start(self) -> None:
        pass

    def stop(self) -> None:
        try:
            self._call("heartbeat", {"stopping": True}, timeout=10)
        except Exception:
            pass

    def heartbeat(self, info: dict) -> int:
        reply = self._call("heartbeat", info) or {}
        self.worker = reply.get("worker", self.worker)
        self.poll_ms = int(reply.get("poll_ms", self.poll_ms))
        self.can_publish = bool(reply.get("can_publish", True))
        return 30_000

    def publish_catalog(self, gym_id: str, meta: dict, problems: list[dict]) -> None:
        self._call("catalog", {"gym": gym_id, "meta": meta, "problems": problems}, timeout=120)

    def publish_code(self, gym_id: str, slug: str, current: dict) -> None:
        self._call("code", {"gym": gym_id, "slug": slug, "current": current})

    def publish_progress(self, gym_id: str, data: dict) -> None:
        self._call("progress", {"gym": gym_id, "data": data})

    def lease(self, gyms: list[str]) -> dict | None:
        reply = self._call("lease", {"gyms": gyms})
        if reply is None:
            return None
        self.poll_ms = int(reply.pop("poll_ms", self.poll_ms))
        job = reply.get("job")
        if job:
            self.leases[job["id"]] = job.get("lease_id")
        return job

    def fetch_bundle(self, job: dict) -> bytes:
        """Only while this worker holds the job's lease: the board checks that before answering."""
        request = urllib.request.Request(
            f"{self.base}/v1/worker/jobs/{job['id']}/bundle", data=b"{}", method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "User-Agent": "gymhub-runner"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise TokenRevoked(f"The board refused this worker's token ({exc.code})") from exc
            raise

    def update(self, job_id: str, output: str | None, host: str | None = None) -> bool:
        body = {"output": output, "lease_id": self.leases.get(job_id), **({"host": host} if host else {})}
        reply = self._call(f"jobs/{job_id}/update", body) or {}
        return bool(reply.get("stop"))

    def complete(self, job_id: str, final: dict, user_progress: dict | None = None) -> None:
        try:
            self._call(f"jobs/{job_id}/complete", {"final": final, "user_progress": user_progress, "lease_id": self.leases.pop(job_id, None)})
        except RuntimeError as exc:
            if "HTTP 409" not in str(exc):  # 409: the lease expired and the job moved on; nothing to do
                raise

    def wait_for_work(self, timeout: float) -> None:
        time.sleep(min(timeout, self.poll_ms / 1000))
