"""Issue #52, the whole path: jobs see no credentials, and a revoked token stops what the runner started.

From the board's revoke button to the dead child process and a job that is queued again or ended.
"""

import io
import json
import os
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse

import pytest

from gymhub import jobqueue
from gymhub.runner import agent, backends, pooled
from gymhub.runner.backends import TokenRevoked
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_worker_api import _admin_and_token, _queue

SECRETS = ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CONFIG", "GYMHUB_WORKER_TOKEN", "HF_TOKEN",
           "GH_TOKEN", "POOL_TOKEN", "SSH_AUTH_SOCK", "CREDENTIALS_DIRECTORY", "AWS_ACCESS_KEY_ID", "DB_PASSWORD")


class Quiet:
    """A backend that accepts output and never cancels."""

    def update(self, job_id, output):
        return False


def _wait_for(condition, seconds=20.0):
    deadline = time.time() + seconds
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.05)


def _sleeper(pid_file) -> list[str]:
    return [sys.executable, "-c", f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(300)"]


def _gone(pid_file) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        return True
    return False


def test_a_spawned_job_sees_no_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "NO_GCLOUD", tmp_path / "no-gcloud")
    for name in SECRETS:
        monkeypatch.setenv(name, "secret-" + name)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    child = [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"]
    output, ok, _ = agent.Runner(Quiet(), {}, "t")._stream("j", child, 60, {"GYMKIT_PROGRESS_FILE": "/p"})
    seen = json.loads(output.strip().splitlines()[-1])
    assert ok and not [name for name, value in seen.items() if value.startswith("secret-")]
    assert seen["CUDA_VISIBLE_DEVICES"] == "0" and seen["GYMKIT_PROGRESS_FILE"] == "/p" and "PATH" in seen
    assert seen["CLOUDSDK_CONFIG"] == str(tmp_path / "no-gcloud")


def test_the_worker_token_leaves_the_environment_and_its_file_is_private(monkeypatch, tmp_path):
    token_file = tmp_path / "worker-token"
    monkeypatch.setattr(agent, "TOKEN_FILE", token_file)
    monkeypatch.setenv("GYMHUB_WORKER_TOKEN", "gw_env")
    assert agent.read_worker_token() == "gw_env" and "GYMHUB_WORKER_TOKEN" not in os.environ
    token_file.write_text("gw_file\n")
    token_file.chmod(0o644)
    assert agent.read_worker_token() == "gw_file"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    token_file.unlink()
    assert agent.read_worker_token() == ""


def test_no_token_exits_with_the_status_systemd_does_not_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "TOKEN_FILE", tmp_path / "missing")
    monkeypatch.delenv("GYMHUB_WORKER_TOKEN", raising=False)
    monkeypatch.setattr(agent, "all_adapters", lambda gyms: {"algo": object()})
    monkeypatch.setattr(sys, "argv", ["gymhub-runner", "--api", "https://workers.example"])
    assert agent.main() == agent.EXIT_REVOKED


def test_stopping_the_runner_stops_the_running_job(tmp_path):
    pid_file = tmp_path / "pid"
    runner = agent.Runner(Quiet(), {}, "t")
    result = {}
    thread = threading.Thread(target=lambda: result.update(out=runner._stream("j", _sleeper(pid_file), 300)))
    thread.start()
    _wait_for(pid_file.is_file)
    runner.kill_active()
    thread.join(20)
    assert not thread.is_alive()
    text, ok, _ = result["out"]
    assert not ok and "shutting down" in text and _gone(pid_file)


class _Reply:
    def __init__(self, response):
        self.status, self._body = response.status_code, response.content

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _route_to(http, monkeypatch):
    """ApiBackend's urllib calls, answered by the board's TestClient."""

    def urlopen(request, timeout=None):
        path = urllib.parse.urlsplit(request.full_url).path
        response = http.request(request.get_method(), path, content=request.data, headers=dict(request.header_items()))
        if response.status_code >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status_code, "", {}, io.BytesIO(response.content))
        return _Reply(response)

    monkeypatch.setattr(backends.urllib.request, "urlopen", urlopen)


class SleepingGym:
    id = "algo"

    def __init__(self, pid_file):
        self.pid_file = pid_file

    def command(self, action, slug, lang, options):
        return _sleeper(self.pid_file)


@pytest.mark.parametrize("tries, status", [(0, "queued"), (jobqueue.MAX_TRIES - 1, "error")])
def test_revoking_the_token_kills_the_job_and_requeues_or_ends_it(board, monkeypatch, tmp_path, tries, status):
    owner, token = _admin_and_token(board)
    job_id = _queue(board, tries=tries, code=None)  # nothing to save first: straight to the test run
    _route_to(client(board), monkeypatch)
    monkeypatch.setattr(agent, "LEASE_REFRESH_SECONDS", 0.2)
    pid_file = tmp_path / "pid"
    runner = agent.Runner(backends.ApiBackend("https://workers.example", token), {"algo": SleepingGym(pid_file)}, "spark")
    thread = threading.Thread(target=runner.worker, daemon=True)
    thread.start()
    _wait_for(pid_file.is_file)
    assert board.store.get("jobs", job_id)["status"] == "running"

    import gymhub.board.workers as workers

    assert owner.delete(f"/api/workers/{token.split('_')[1]}", headers=H).status_code == 200
    workers._cache.clear()
    thread.join(30)
    assert not thread.is_alive() and runner.revoked and runner.stop.is_set()
    assert _gone(pid_file)
    job = board.store.get("jobs", job_id)
    assert job["status"] == status and "removed from the board" in job["output"]
    if status == "queued":  # another machine picks it up
        assert job["runner"] is None and not job["lease_id"]
    else:  # it has had its tries: ended, like a job whose machine stopped answering
        assert job["finished_at"] and job["ok"] is False


def test_a_revoked_token_cancels_the_pool_job(monkeypatch):
    calls = []

    class Pool:
        def upload_folder(self, folder, **_):
            return {"artifact": "a"}

        def call(self, method, path, body=None, timeout=None):
            calls.append((method, path))
            if path == "/v1/jobs":
                return {"job": {"id": "p1"}}
            return {"chunks": [], "next": 0, "status": "running"}

    def revoked(_text):
        raise TokenRevoked("revoked")

    monkeypatch.setattr(pooled, "stage", lambda *a, **k: ["true"])
    monkeypatch.setattr(pooled, "spec_for", lambda *a, **k: {"network": "none"})
    gym = type("Gym", (), {"id": "algo"})()
    with pytest.raises(TokenRevoked):
        pooled.run(Pool(), gym, {"code": "x", "requires": {}}, "p", "python", "test", revoked, 60)
    assert ("POST", "/v1/jobs/p1/cancel") in calls
