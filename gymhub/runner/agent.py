"""The runner: lives on your GPU machine, executes jobs from the board, keeps the board in sync.

    python -m gymhub.runner                       # direct mode: Firestore with this machine's credentials
    python -m gymhub.runner --api https://<board> # worker mode: only the board's API, with a worker token
    python -m gymhub.runner --gyms gpu            # only the GPU gym
    python -m gymhub.runner --dispatch-only       # a dispatcher: every job to the compute pool, nothing here

Worker mode reads the token from GYMHUB_WORKER_TOKEN or ~/.config/gymhub/worker-token (create one on
the board's Compute page); it is what deploy/install-runner.sh installs. Direct mode is for local
development: Firestore credentials can read and write the whole board. Either way it only makes
outbound connections, and jobs run one at a time so benchmarks never share the GPU.

--dispatch-only (or GYMHUB_EXECUTOR=pool with GYMHUB_DISPATCH_ONLY=1) makes it a dispatcher
(gymhub/runner/pooled.py, #135): it leases jobs and grades every one of them on the compute pool, the
owner's too, and refuses to execute anything on this machine. It needs no GPU and no Docker.

Other users' code runs only on the compute pool, in its containers (pool mode is the default; see
gymhub/runner/pooled.py). A runner has no sandbox of its own: without a pool it grades only its
owner's jobs. In direct mode (no board in between to check whose job it is) it leases no isolated
job it cannot send to the pool, unless GYMHUB_TRUST_JOBS=1 (local development and tests).
"""

import argparse
import codecs
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from gymhub import poolview
from gymhub.gymversions import VALIDATE_TIMEOUT, bundle_for_job
from gymhub.jobqueue import empty_progress
from gymhub.runner.adapters import REPO, GymAdapter, all_adapters, bundle_adapter
from gymhub.runner import bundles, pooled
from gymhub.runner.capabilities import probe
from gymhub.runner.backends import ApiBackend, DirectBackend, TokenRevoked, trusts_jobs
from gymhub.store import open_store

MAX_OUTPUT = 200_000  # characters kept per job (Firestore documents are capped at 1 MiB)
MAX_PRINTED = 5_000_000  # characters a job may print before it is stopped
CUT_NOTE = "[earlier output was cut]\n"
TIMEOUTS = {"run": 300, "test": 300, "bench": 1200, "validate": VALIDATE_TIMEOUT}
BACKUPS = REPO / ".cache" / "gymhub" / "backups"
# Run-mode code and other users' code live here, never in solutions/. One directory per job, removed afterwards.
SCRATCH = REPO / ".cache" / "gymhub" / "scratch"
TOKEN_FILE = Path.home() / ".config" / "gymhub" / "worker-token"
LEASE_REFRESH_SECONDS = 20
EXIT_REVOKED = 78  # EX_CONFIG: the unit has RestartPreventExitStatus=78, so a revoked token is not retried
# Never passed on to a job: cloud and API credentials of whoever started the runner.
SECRET_PREFIXES = ("GOOGLE_", "CLOUDSDK_", "GCLOUD_", "AWS_", "AZURE_", "ANTHROPIC_", "OPENAI_")
SECRET_NAMES = {"GYMHUB_WORKER_TOKEN", "GYMHUB_TOKEN", "GYMHUB_STORE", "GYMHUB_STORE_PREFIX", "GYMHUB_BLOBS",
                "GYMHUB_SESSION_KEY", "GYMHUB_API",
                # agents and stores that hand out credentials to whoever can reach them
                "SSH_AUTH_SOCK", "GPG_AGENT_INFO", "CREDENTIALS_DIRECTORY", "KUBECONFIG", "POOL_SERVER"}
# Any other variable named like a credential (HF_TOKEN, GH_TOKEN, POOL_TOKEN, DB_PASSWORD, ...).
SECRET_SUFFIXES = ("_TOKEN", "_SECRET", "_PASSWORD", "_PASSWD", "_API_KEY", "_ACCESS_KEY", "_PRIVATE_KEY",
                   "_CREDENTIALS", "_SESSION_KEY")
NO_GCLOUD = REPO / ".cache" / "gymhub" / "no-gcloud"  # an empty gcloud config: no login to find


def is_secret(name: str) -> bool:
    upper = name.upper()
    return upper in SECRET_NAMES or upper.startswith(SECRET_PREFIXES) or upper.endswith(SECRET_SUFFIXES)


def job_environment(extra: dict | None = None) -> dict:
    """The environment a job runs with: the runner's own minus every credential.

    CLOUDSDK_CONFIG points the Google libraries at an empty folder, so a login stored in the
    user's ~/.config/gcloud is not found by a job either."""
    env = {key: value for key, value in os.environ.items() if not is_secret(key)}
    NO_GCLOUD.mkdir(parents=True, exist_ok=True)
    env.update({"NO_COLOR": "1", "PYTHONUNBUFFERED": "1", "TERM": "dumb", "CLOUDSDK_CONFIG": str(NO_GCLOUD)})
    env.update(extra or {})
    return env


class OutputTail:
    """The last `keep` characters of a job's output, however much it prints.

    The runner holds this, not the whole output: memory stays under about 2 x `keep` plus one read.
    `total` counts everything printed, so the caller can stop a job that never stops printing.
    """

    def __init__(self, keep: int, limit: int):
        self.keep, self.limit = keep, limit
        self.parts: list[str] = []
        self.size = self.total = self.version = 0
        self.cut = False

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.size += len(text)
        self.total += len(text)
        self.version += 1
        if self.size > 2 * self.keep:
            self._compact()

    def _compact(self) -> None:
        joined = "".join(self.parts)
        if len(joined) > self.keep:
            joined, self.cut = joined[-self.keep:], True
        self.parts, self.size = [joined], len(joined)

    def text(self) -> str:
        self._compact()
        return (CUT_NOTE if self.cut else "") + self.parts[0] if self.parts else ""

    @property
    def over_limit(self) -> bool:
        return self.total > self.limit


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


class Runner:
    def __init__(self, backend, adapters: dict[str, GymAdapter], name: str, courses: list[str] | None = None):
        self.backend = backend
        self.courses = courses  # seed these courses at start ([] = every one; None = none): a local board only
        self.adapters = adapters
        self.name = name
        self.stop = threading.Event()
        self.code_hashes: dict[str, str] = {}
        self.solution_stamps: dict[str, dict] = {}
        self.progress_stamps: dict[str, int] = {}
        self.catalog_stamps: dict[str, tuple] = {}
        self.current_job: str | None = None
        self.revoked = False  # the board refused this machine's token; main() exits with EXIT_REVOKED
        self.pool_client = None  # GYMHUB_EXECUTOR=pool: the pool member that grades isolated jobs
        self.pool_bridge: dict | None = None  # what the last heartbeat told the board the pool grades
        self.pool_report: dict | None = None  # what it told the board's Compute page of the pool (gymhub/poolview.py)
        self._pool_missing_logged = False
        self._active_lock = threading.Lock()
        self._active: subprocess.Popen | None = None  # the job's process running now
        self._stopped_note = ""

    def kill_active(self, note: str = "\nStopped: the runner is shutting down.\n") -> None:
        """Stop the job running now (its process group).

        Called when the board refuses this machine's token and when the runner stops, so nothing it
        started outlives it."""
        with self._active_lock:
            process, self._active = self._active, None
            if process is None:
                return
            self._stopped_note = note
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # it had just finished
            pass

    def _revoked(self, exc: BaseException) -> None:
        self.log(str(exc))
        self.revoked = True
        self.kill_active("\nStopped: the board revoked this machine's token.\n")
        self.stop.set()

    # ---- publishing --------------------------------------------------------------------------

    def log(self, message: str) -> None:
        print(time.strftime("%H:%M:%S ") + message, flush=True)

    def publish_catalog(self, adapter: GymAdapter) -> None:
        started = time.time()
        catalog = adapter.load_catalog()
        problems = []
        for problem in catalog["problems"]:
            current = adapter.current_code(problem["slug"])
            files = {
                lang: str(adapter.solution_path(problem["slug"], lang).relative_to(REPO))
                for lang in problem.get("langs", [])
                if lang in adapter.lang_files
            }
            problems.append({**problem, "current": current, "files": files})
            self.code_hashes[f"{adapter.id}__{problem['slug']}"] = _hash(repr(sorted(current.items())))
        self.backend.publish_catalog(adapter.id, catalog["gym"], problems)
        self.publish_progress(adapter)
        self.catalog_stamps[adapter.id] = adapter.catalog_stamp()
        self.solution_stamps[adapter.id] = adapter.solution_stamps()
        self.log(f"published {adapter.id}: {len(problems)} exercises in {time.time() - started:.1f}s")

    def publish_progress(self, adapter: GymAdapter) -> None:
        self.backend.publish_progress(adapter.id, adapter.progress())
        self.progress_stamps[adapter.id] = adapter.progress_stamp()

    def publish_code(self, adapter: GymAdapter, slug: str) -> None:
        key = f"{adapter.id}__{slug}"
        current = adapter.current_code(slug)
        digest = _hash(repr(sorted(current.items())))
        if self.code_hashes.get(key) == digest:
            return
        self.backend.publish_code(adapter.id, slug, current)
        self.code_hashes[key] = digest
        self.log(f"synced local edit {adapter.id}/{slug}")

    def heartbeat(self) -> None:
        self.pool_report = None
        info = {
            "name": self.name,
            "host": socket.gethostname(),
            "gpu": self.gpu,
            "platform": f"{platform.system()} {platform.machine()}",
            "gyms": sorted(self.adapters),
            "busy": self.current_job,
            "repo": str(REPO),
            "capabilities": self.capabilities(),
        }
        if self.pool_report is not None:  # the Compute page's view of the pool, for a board that cannot reach it (#162)
            info["pool_report"] = self.pool_report
        self.backend.heartbeat(info)

    def capabilities(self) -> dict:
        """This machine's, plus, in pool mode, what it grades on the pool (gymhub/routing.py). A
        dispatcher reports nothing of its own: no GPU, no tools, since it runs nothing here. `sandbox`
        is always false: this machine never isolates another user's code itself, only the pool does
        (gymhub/routing.py `for_job`). Older runners with the host-mount sandbox reported true."""
        if pooled.dispatch_only():
            found = {"gpus": [], "tools": [], "sandbox": False, "bundles": True, "arch": platform.machine(), "dispatch_only": True}
        else:
            found = {**probe(), "sandbox": False, "bundles": True}
        if pooled.enabled():
            bridge = self._pool_bridge()
            if bridge is not None:
                found["pool"] = bridge
        return found

    def _pool_bridge(self) -> dict | None:
        from pool.client import Client, PoolError

        try:
            if self.pool_client is None:
                self.pool_client = Client.member()
        except PoolError as exc:
            if not self._pool_missing_logged:  # pool mode is the default: say it once, not every heartbeat
                self.log(f"no pool member to grade as ({exc}); other users' jobs will not run here")
                self._pool_missing_logged = True
            return None
        pool_name = os.environ.get("GYMHUB_POOL")
        machines = pooled.fetch_machines(self.pool_client, pool_name)
        bridge = pooled.advertised(self.pool_client, pool_name, machines)
        if machines is not None:
            self.pool_report = poolview.report(self.pool_client, pool_name, machines)
        self.pool_bridge = bridge
        if bridge is None:
            self.log("the pool cannot be reached; leasing only what this machine runs itself")
        return bridge

    # ---- job handling ------------------------------------------------------------------------

    def _backup(self, path: Path, gym: str, slug: str) -> None:
        if not path.is_file():
            return
        target = BACKUPS / gym / slug / f"{time.strftime('%Y%m%d-%H%M%S')}-{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        backups = sorted(target.parent.iterdir())
        for old in backups[:-30]:
            old.unlink()

    def write_solution(self, adapter: GymAdapter, slug: str, lang: str, code: str) -> bool:
        path = adapter.solution_path(slug, lang)
        if path.is_file() and path.read_text() == code:
            return False
        self._backup(path, adapter.id, slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        return True

    def _adapter_for(self, job: dict) -> GymAdapter | None:
        """A local gym, or an uploaded gym's bundle (fetched and unpacked on first use)."""
        bundle = bundle_for_job(job)
        if bundle is not None:
            return bundle_adapter(bundles.ensure(self.backend, job, bundle))
        return self.adapters.get(job.get("gym", ""))

    def run_job(self, job: dict) -> None:
        job_id = job["id"]
        slug, lang, action = job["slug"], job.get("lang", ""), job["action"]
        self.current_job = job_id
        self.user_progress = None
        self.job_result = None
        self.job_hardware = None
        self.job_host = None  # the pool machine that graded it, when not this one
        try:
            adapter = self._adapter_for(job)
        except TokenRevoked:
            raise
        except Exception as exc:
            self.current_job = None
            self.backend.complete(job_id, {"status": "error", "ok": False, "output": f"Could not get this gym's files: {exc}\n"})
            return
        if adapter is None:
            self.current_job = None
            self.backend.complete(job_id, {"status": "error", "ok": False, "output": f"This machine has no gym {job.get('gym')!r}.\n"})
            return
        self.log(f"job {job_id}: {action} {adapter.id}/{slug} [{lang}]")
        try:
            if action == "validate":
                output, ok, exit_code = self._execute_validation(job_id, job, adapter)
            else:
                output, ok, exit_code = self._execute(job_id, job, adapter, slug, lang, action)
            # Stopped because the runner is stopping: the machine failed, not the submission.
            status = "error" if self.stop.is_set() and not ok else "done"
        except TokenRevoked:
            raise
        except pooled.ImageUnavailable as exc:  # the pool machine could not start it: nothing was graded
            output, ok, exit_code, status = exc.output, False, None, "error"
        except Exception as exc:
            output, ok, exit_code, status = f"runner error: {type(exc).__name__}: {exc}\n", False, None, "error"
        final = {"status": status, "ok": ok, "exit_code": exit_code, "output": output[-MAX_OUTPUT:]}
        if self.job_result is not None:
            final["result"] = self.job_result
        if self.job_hardware:
            final["hardware"] = self.job_hardware
        if self.job_host:
            final.update(host=self.job_host, via=self.name)
        self.backend.complete(job_id, final, self.user_progress)
        self.current_job = None
        if job.get("isolated"):
            self.log(f"job {job_id}: {'ok' if ok else 'failed'} (workspace of {job.get('created_by')})")
            return
        if action == "run":
            self.log(f"job {job_id}: {'ok' if ok else 'failed'} (run mode, not recorded)")
            return
        self.publish_code(adapter, slug)
        self.publish_progress(adapter)
        self.solution_stamps[adapter.id] = adapter.solution_stamps()
        self.log(f"job {job_id}: {'ok' if ok else 'failed'}")

    def _execute(self, job_id, job, adapter: GymAdapter, slug, lang, action):
        if job.get("isolated") or bundle_for_job(job):  # an uploaded gym never runs in place
            return self._execute_isolated(job_id, job, adapter, slug, lang, action)
        if pooled.dispatch_only() and action in {"run", "test", "bench"}:
            return self._execute_owner_on_pool(job_id, job, adapter, slug, lang, action)
        if pooled.enabled() and action in pooled.actions() and pooled.target_machine(job):  # "Run on" a pool machine
            return self._execute_owner_on_pool(job_id, job, adapter, slug, lang, action)
        if action in {"save", "test", "bench"} and job.get("code") is not None:
            if self.write_solution(adapter, slug, lang, job["code"]) and action == "save":
                return f"Saved {adapter.solution_path(slug, lang).relative_to(REPO)}\n", True, 0
        if action == "save":
            return "No changes to save.\n", True, 0
        if action == "reset":
            starter = adapter.starter_path(slug, lang)
            if not starter.is_file():
                return f"No starter for {slug} [{lang}]\n", False, 1
            self.write_solution(adapter, slug, lang, starter.read_text())
            return f"Restored the starter for {slug} [{lang}]. The previous file is backed up locally.\n", True, 0
        if action == "hint":
            progress = adapter.progress()
            count = int(job.get("options", {}).get("count", 0))
            entry = progress.setdefault("exercises", {}).setdefault(slug, {"hints_seen": 0, "langs": {}})
            if count > entry.get("hints_seen", 0):
                entry["hints_seen"] = count
                adapter.progress_file.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
            return "", True, 0
        if action == "run":
            if job.get("code") is None:
                return "Run mode needs code from the editor.\n", False, 1
            work = SCRATCH / job_id / slug
            work.mkdir(parents=True, exist_ok=True)
            file = work / adapter.solution_path(slug, lang).name
            file.write_text(job["code"])
            options = job.get("options") or {}
            run_options: dict = {}
            if options.get("cases"):
                cases_file = work / "custom_cases.json"
                cases_file.write_text(json.dumps(options["cases"]))
                run_options = {"cases_file": cases_file, "only_custom": bool(options.get("only_custom"))}
            try:
                return self._stream(job_id, adapter.command("run", slug, lang, run_options, file), TIMEOUTS["run"])
            finally:
                shutil.rmtree(SCRATCH / job_id, ignore_errors=True)
        if action not in TIMEOUTS:
            return f"Unknown action {action!r}\n", False, 1
        return self._stream(job_id, adapter.command(action, slug, lang, job.get("options", {})), TIMEOUTS[action])

    def _execute_validation(self, job_id, job, adapter: GymAdapter):
        """Check an uploaded gym version (gymhub/runner/validate.py). It runs the uploader's code, so it
        goes to the pool like any other isolated job; here only when the board let this machine have
        it (its owner uploaded it, or GYMHUB_TRUST_JOBS in direct mode)."""
        if pooled.dispatch_only() or self._on_pool(job, "validate"):
            output, ok, exit_code, result = pooled.validate(self._pool(), adapter, job, self._pool_output(job_id),
                                                            TIMEOUTS["validate"], self._pool_machine(job_id))
            self.job_result = result
            return output, ok, exit_code
        root = adapter.mounts[0]
        work = SCRATCH / job_id
        work.mkdir(parents=True, exist_ok=True)
        try:
            out = work / "result.json"
            command = [sys.executable, "-m", "gymhub.runner.validate", str(root), "--out", str(out)]
            result = self._stream(job_id, command, TIMEOUTS["validate"])
            if out.is_file():
                self.job_result = json.loads(out.read_text())
            return result
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _execute_isolated(self, job_id, job, adapter: GymAdapter, slug, lang, action):
        """Another user's job: their code in a scratch folder, their progress, nothing of the owner's touched."""
        if action not in {"run", "test", "bench"}:
            return f"Action {action!r} is handled by the board for workspace users.\n", False, 1
        if job.get("code") is None:
            return "No code was sent with this job.\n", False, 1
        if pooled.dispatch_only() or self._on_pool(job, action):
            return self._execute_on_pool(job_id, job, adapter, slug, lang, action)
        work = SCRATCH / job_id / slug
        work.mkdir(parents=True, exist_ok=True)
        try:
            file = work / adapter.solution_path(slug, lang).name
            file.write_text(job["code"])
            options = dict(job.get("options") or {})
            if action == "run" and options.get("cases"):
                cases_file = work / "custom_cases.json"
                cases_file.write_text(json.dumps(options["cases"]))
                options = {"cases_file": cases_file, "only_custom": bool(options.get("only_custom"))}
            env = dict(adapter.env)
            progress_file = work / "progress.json"
            if action != "run":
                progress_file.write_text(json.dumps(job.get("user_progress") or empty_progress()))
                variable = "GPU_GYM_PROGRESS_FILE" if adapter.runtime == "gpu_gym" else "GYMKIT_PROGRESS_FILE"
                env[variable] = str(progress_file)
            command = adapter.command(action, slug, lang, options, file)
            result = self._stream(job_id, command, TIMEOUTS[action], env)
            if action != "run" and progress_file.is_file():
                self.user_progress = json.loads(progress_file.read_text())
            return result
        finally:
            shutil.rmtree(SCRATCH / job_id, ignore_errors=True)

    def _on_pool(self, job: dict, action: str) -> bool:
        """Is this isolated job graded on the pool? Yes when this runner has a pool member, the action
        is one it sends there, and, for a gym that is not public, it reported own pool machines (with
        none, the board routed the job by this machine, so it is graded here: only its owner's)."""
        return pooled.enabled() and action in pooled.actions() and pooled.takes(self.pool_bridge, job) and self._has_pool()

    def _has_pool(self) -> bool:
        """A pool member to submit as (POOL_SERVER/POOL_TOKEN or ~/.config/pool/member.json)."""
        from pool.client import PoolError

        try:
            self._pool()
        except PoolError:
            return False
        return True

    def _pool(self):
        from pool.client import Client

        if self.pool_client is None:
            self.pool_client = Client.member()
        return self.pool_client

    def _pool_output(self, job_id: str):
        def on_output(text: str) -> bool:
            try:
                return bool(self.backend.update(job_id, text))
            except TokenRevoked:
                raise
            except Exception as exc:
                self.log(f"job {job_id}: could not report progress: {exc}")
                return False

        return on_output

    def _pool_machine(self, job_id: str):
        """on_machine for pooled.run: remember the pool machine and tell the board ("Running on ...")."""
        def on_machine(name: str) -> None:
            self.job_host = name
            self.log(f"job {job_id}: on pool machine {name}")
            try:
                self.backend.update(job_id, None, host=name)
            except TokenRevoked:
                raise
            except Exception as exc:
                self.log(f"job {job_id}: could not report the pool machine: {exc}")

        return on_machine

    def _execute_owner_on_pool(self, job_id, job, adapter: GymAdapter, slug, lang, action):
        """A dispatcher grades the owner's own job on the pool too: the editor's code (saved to the
        owner's solution file first, as in place) and the owner's progress, which comes back here."""
        code = job.get("code")
        if code is not None and action in {"test", "bench"}:
            self.write_solution(adapter, slug, lang, code)
        if code is None:
            path = adapter.solution_path(slug, lang)
            if action == "run" or not path.is_file():
                return "No code was sent with this job.\n", False, 1
            code = path.read_text()
        pool_job = {**job, "code": code, "user_progress": adapter.progress() if action != "run" else None}
        output, ok, exit_code = self._execute_on_pool(job_id, pool_job, adapter, slug, lang, action)
        if self.user_progress is not None:
            adapter.progress_file.write_text(json.dumps(self.user_progress, indent=2, sort_keys=True) + "\n")
            self.user_progress = None  # the owner's progress lives in the owner's file, published after the job
        return output, ok, exit_code

    def _execute_on_pool(self, job_id, job, adapter: GymAdapter, slug, lang, action):
        """Grade on the compute pool (gymhub/runner/pooled.py): any machine of the pool that fits (only our
        own machines for a gym that is not public)."""
        self.log(f"job {job_id}: grading on the pool")
        output, ok, exit_code, progress, hardware = pooled.run(self._pool(), adapter, job, slug, lang, action,
                                                               self._pool_output(job_id), TIMEOUTS[action],
                                                               self._pool_machine(job_id))
        if progress is not None:
            self.user_progress = progress
        self.job_hardware = hardware  # the pool machine's, for the board's per-GPU best times
        return output, ok, exit_code

    def _stream(self, job_id: str, command: list[str], timeout: int, extra_env: dict | None = None):
        """Run `command`, report its output about once a second, and stop it when it overruns.

        A job is stopped at its time limit, when the board cancels it, or when it prints more than
        MAX_PRINTED characters. Only the last MAX_OUTPUT characters are ever kept.
        """
        if pooled.dispatch_only():  # the one place a job's process starts: a dispatcher starts none
            raise RuntimeError("this runner is a dispatcher (GYMHUB_DISPATCH_ONLY): it executes nothing locally")
        env = job_environment(extra_env)
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        with self._active_lock:
            self._active, self._stopped_note = process, ""
        try:
            return self._watch(job_id, process, timeout)
        finally:
            with self._active_lock:
                if self._active is process:
                    self._active = None
            if process.poll() is None:  # an exception on the way out: never leave it running
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    def _watch(self, job_id: str, process: subprocess.Popen, timeout: int):
        tail = OutputTail(MAX_OUTPUT - len(CUT_NOTE), MAX_PRINTED)
        lock = threading.Lock()
        flooded = threading.Event()

        def reader():
            assert process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while True:
                data = os.read(process.stdout.fileno(), 65536)  # whatever is there, never a whole unbounded line
                if not data:
                    break
                with lock:
                    tail.add(decoder.decode(data))
                    if tail.over_limit:
                        flooded.set()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        started = time.time()
        last_version = -1
        last_update = 0.0
        killed = ""
        while process.poll() is None:
            flooded.wait(1.0)
            with lock:
                text, version, over = tail.text(), tail.version, tail.over_limit
            stop = False
            # Send new output about once a second; otherwise just keep the lease alive.
            if version != last_version or time.time() - last_update > LEASE_REFRESH_SECONDS:
                try:
                    stop = self.backend.update(job_id, text if version != last_version else None)
                    last_version, last_update = version, time.time()
                except TokenRevoked:
                    # This machine was removed from the board: nothing it runs can be reported, so stop it now.
                    self.kill_active("\nStopped: the board revoked this machine's token.\n")
                    process.wait()
                    raise
                except Exception as exc:
                    self.log(f"job {job_id}: could not report progress: {exc}")
            if time.time() - started > timeout:
                killed = f"\nStopped: over the {timeout} s time limit.\n"
            elif stop:
                killed = "\nCancelled from the board.\n"
            elif over:
                killed = f"\nStopped: it printed more than {MAX_PRINTED:,} characters.\n"
            if killed:
                os.killpg(process.pid, signal.SIGKILL)
                break
        process.wait()
        thread.join(timeout=5)
        if not killed and self._stopped_note and process.returncode != 0:
            killed = self._stopped_note  # stopped by kill_active from another thread
        with lock:
            text = tail.text() + killed
        return text, process.returncode == 0 and not killed, process.returncode

    def worker(self) -> None:
        gyms = sorted(self.adapters)
        while not self.stop.is_set():
            try:
                job = self.backend.lease(gyms)
            except TokenRevoked as exc:
                self._revoked(exc)
                return
            except Exception as exc:
                self.log(f"could not fetch jobs: {exc}")
                self.stop.wait(10)
                continue
            if job is None:
                self.backend.wait_for_work(30)
                continue
            try:
                self.run_job(job)
            except TokenRevoked as exc:
                self._revoked(exc)
                return
            except Exception as exc:
                self.log(f"job {job.get('id')} crashed the worker loop: {exc}")

    # ---- local sync --------------------------------------------------------------------------

    def sync_loop(self) -> None:
        last_beat = 0.0
        while not self.stop.wait(2.0):
            try:
                if time.time() - last_beat > 30:
                    self.heartbeat()
                    last_beat = time.time()
                if self.current_job or not self.backend.can_publish:
                    continue
                for adapter in self.adapters.values():
                    if adapter.catalog_stamp() != self.catalog_stamps.get(adapter.id):
                        self.publish_catalog(adapter)
                        continue
                    stamps = adapter.solution_stamps()
                    previous = self.solution_stamps.get(adapter.id, {})
                    for slug, stamp in stamps.items():
                        if previous.get(slug) != stamp:
                            self.publish_code(adapter, slug)
                    self.solution_stamps[adapter.id] = stamps
                    if adapter.progress_stamp() != self.progress_stamps.get(adapter.id):
                        self.publish_progress(adapter)
            except TokenRevoked as exc:
                self._revoked(exc)
            except Exception as exc:
                self.log(f"sync error: {exc}")

    def seed_courses(self) -> None:
        """A local board's courses (gymhub/courses.py), next to the worker so jobs are not held up."""
        from gymhub import courses
        from gymhub.blobs import open_blobs

        try:
            outcomes = courses.seed(self.backend.store, open_blobs(), only=self.courses or None, log=self.log)
        except Exception as exc:  # noqa: BLE001 - the runner's own gyms still work
            self.log(f"could not seed the courses: {type(exc).__name__}: {exc}")
            return
        if outcomes and not trusts_jobs() and not self._has_pool():
            self.log("courses with code (ml-coding) are graded as uploaded gyms, so their runs wait for a pool; "
                     "on a board only you use, start the runner with GYMHUB_TRUST_JOBS=1 to grade them here")

    def run(self) -> None:
        self.gpu = "none (dispatcher)" if pooled.dispatch_only() else _gpu_name()
        self.current_job = None
        self.heartbeat()
        if not self.backend.can_publish:
            self.log("this machine runs only its owner's jobs; the board's exercises come from the board owner's machine")
        for adapter in self.adapters.values() if self.backend.can_publish else []:
            try:
                self.publish_catalog(adapter)
            except TokenRevoked:
                raise
            except Exception as exc:
                self.log(f"could not publish {adapter.id}: {exc}")
        self.backend.start()
        targets = [self.worker, self.sync_loop] + ([self.seed_courses] if self.courses is not None else [])
        threads = [threading.Thread(target=target, daemon=True) for target in targets]
        for thread in threads:
            thread.start()
        mode = "worker API" if isinstance(self.backend, ApiBackend) else "direct"
        self.log(f"runner {self.name} ready ({mode}): gyms {', '.join(self.adapters)}  GPU {self.gpu}")
        try:
            while not self.stop.wait(1):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            self.kill_active()
            threads[0].join(timeout=10)  # the worker: lets a pool job be cancelled on its way out
            self.backend.stop()


def read_worker_token() -> str:
    """The worker token, from GYMHUB_WORKER_TOKEN or TOKEN_FILE. The variable is removed from this
    process's environment once read, and a token file readable by others is made private."""
    token = os.environ.pop("GYMHUB_WORKER_TOKEN", "").strip()
    if token or not TOKEN_FILE.is_file():
        return token
    if TOKEN_FILE.stat().st_mode & 0o077:
        TOKEN_FILE.chmod(0o600)
    return TOKEN_FILE.read_text().strip()


def protect_process() -> None:
    """Make this process non-dumpable (Linux): other processes of the same user, the jobs it runs
    among them, can then no longer read its memory or /proc/<pid>/environ, where the token lives.
    Children are dumpable again after exec, so this does not change how jobs run."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        PR_SET_DUMPABLE = 4
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except Exception:
        pass


def owner_login_warnings() -> list[str]:
    """What on this machine still gives cloud access a worker does not need (issue #52)."""
    found = []
    adc = Path(os.environ.get("CLOUDSDK_CONFIG") or Path.home() / ".config" / "gcloud") / "application_default_credentials.json"
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        found.append("GOOGLE_APPLICATION_CREDENTIALS is set for the runner; remove it from its environment.")
    if adc.is_file():
        found.append(f"A Google application-default login is stored in {adc}; the runner does not need it. "
                     "Remove it with: gcloud auth application-default revoke")
    if trusts_jobs():
        found.append("GYMHUB_TRUST_JOBS=1: isolated jobs run here as this user and could read the worker token file.")
    return found


def _local_courses(args, only: list[str] | None) -> list[str] | None:
    """The courses to seed: on a local store (sqlite) in direct mode only, where nobody uploads them.
    A hosted board gets them pushed (gyms/README.md)."""
    if args.api or args.no_courses or not os.environ.get("GYMHUB_STORE", "").startswith("sqlite:"):
        return None
    if not only:
        return []
    from gymhub import courses

    chosen = [package.id for package in courses.courses(only=only)]
    return chosen or None


def main() -> int:
    parser = argparse.ArgumentParser(prog="gymhub-runner", description=__doc__.splitlines()[0])
    parser.add_argument("--gyms", help="Comma-separated gym ids. Default: every gym found")
    parser.add_argument("--name", default=socket.gethostname(), help="Runner name shown on the board (direct mode)")
    parser.add_argument("--api", default=os.environ.get("GYMHUB_API"), help="Board URL: use the worker API with a token")
    parser.add_argument("--no-courses", action="store_true",
                        help="With a local store (GYMHUB_STORE=sqlite:...), do not publish the repository's courses "
                             "(gyms/README.md) to it at start")
    parser.add_argument("--dispatch-only", action="store_true",
                        help="Grade every job on the compute pool and execute nothing here (GYMHUB_EXECUTOR=pool GYMHUB_DISPATCH_ONLY=1)")
    args = parser.parse_args()
    if args.dispatch_only:
        os.environ["GYMHUB_EXECUTOR"], os.environ["GYMHUB_DISPATCH_ONLY"] = "pool", "1"
    if pooled.dispatch_only():
        try:
            from pool.client import Client

            Client.member()
        except Exception as exc:  # nothing could be graded: say so now rather than fail every job
            print(f"--dispatch-only needs a pool member to submit as: {exc}", file=sys.stderr)
            return EXIT_REVOKED
    only = args.gyms.split(",") if args.gyms else None
    adapters = all_adapters(only)
    if not adapters:
        raise SystemExit("No gyms selected.")
    if args.api:
        token = read_worker_token()
        if not token:  # EX_CONFIG as well: systemd would only restart into the same missing token
            print(f"No worker token: set GYMHUB_WORKER_TOKEN or put it in {TOKEN_FILE}. Create one on the board's Compute page.",
                  file=sys.stderr)
            return EXIT_REVOKED
        protect_process()
        backend = ApiBackend(args.api, token)
        for warning in owner_login_warnings():
            print("warning: " + warning, file=sys.stderr, flush=True)
    else:
        backend = DirectBackend(open_store(), args.name)
    runner = Runner(backend, adapters, args.name, courses=_local_courses(args, only))
    signal.signal(signal.SIGTERM, lambda *_: runner.stop.set())
    try:
        runner.run()
    except TokenRevoked as exc:
        runner.log(str(exc))
        runner.revoked = True
    if runner.revoked:
        runner.log("This machine's token was revoked. Register it again on the board to bring it back.")
        return EXIT_REVOKED
    return 0
