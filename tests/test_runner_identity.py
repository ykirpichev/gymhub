"""The runner holds a worker token, not the owner's cloud login (#52)."""

import os
import sys
import time

import pytest

from gymhub.runner import agent
from gymhub.runner.backends import TokenRevoked
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_worker_api import _admin_and_token, _queue, _worker


def test_the_worker_surface_serves_only_the_worker_api(board, monkeypatch):
    monkeypatch.setenv("GYMHUB_SURFACE", "workers")
    anon = client(board)
    assert anon.get("/api/me").status_code == 404
    assert anon.get("/").status_code == 404
    assert anon.post("/v1/worker/heartbeat", json={}).status_code == 401  # reached, token checked
    assert anon.get("/install.sh").status_code == 200


def test_registration_points_machines_at_the_worker_url(board, monkeypatch):
    monkeypatch.setenv("GYMHUB_WORKER_URL", "https://workers.example")
    owner = client(board)
    sign_in(owner, "owner")
    reply = owner.post("/api/my/workers/registration", json={"name": "spark"}, headers=H).json()
    assert reply["install_url"] == "https://workers.example"


def test_revoking_a_machine_returns_its_running_job_to_the_queue(board):
    owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    running = _queue(board)
    cancelled = _queue(board)
    first = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    second = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    assert {first["id"], second["id"]} == {running, cancelled}
    board.store.set("jobs", cancelled, {"cancel": True}, merge=True)
    worker_id = token.split("_")[1]
    assert owner.delete(f"/api/workers/{worker_id}", headers=H).status_code == 200
    requeued = board.store.get("jobs", running)
    assert requeued["status"] == "queued" and requeued["runner"] is None and not requeued["lease_id"]
    assert board.store.get("jobs", cancelled)["status"] == "cancelled"
    late = {"final": {"status": "done", "ok": True}, "lease_id": first["lease_id"]}
    assert worker.post(f"/v1/worker/jobs/{first['id']}/complete", json=late).status_code in (401, 409)


def test_jobs_do_not_inherit_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "NO_GCLOUD", tmp_path / "no-gcloud")
    for name in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CORE_ACCOUNT", "GYMHUB_STORE",
                 "GYMHUB_WORKER_TOKEN", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("SOME_SETTING", "kept")
    env = agent.job_environment({"GYMKIT_PROGRESS_FILE": "/x"})
    assert not [key for key in env if "secret" == env[key]], [key for key in env if env[key] == "secret"]
    assert env["SOME_SETTING"] == "kept" and env["GYMKIT_PROGRESS_FILE"] == "/x"
    assert env["CLOUDSDK_CONFIG"] == str(tmp_path / "no-gcloud") and not os.listdir(env["CLOUDSDK_CONFIG"])


class RevokedBackend:
    def update(self, job_id, output):
        raise TokenRevoked("The board refused this worker's token (401)")


def test_a_revoked_token_kills_the_running_job(tmp_path):
    pid_file = tmp_path / "pid"
    child = [sys.executable, "-c", f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(120)"]
    runner = agent.Runner(RevokedBackend(), {}, "t")
    started = time.time()
    with pytest.raises(TokenRevoked):
        runner._stream("j", child, 300)
    assert time.time() - started < 20
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_a_revoked_runner_exits_so_systemd_does_not_restart_it(monkeypatch):
    class Backend:
        can_publish = False

        def lease(self, gyms):
            raise TokenRevoked("revoked")

    runner = agent.Runner(Backend(), {"algo": object()}, "t")
    runner.worker()
    assert runner.revoked and runner.stop.is_set()
    assert agent.EXIT_REVOKED == 78
    unit = (agent.REPO / "deploy" / "gymhub-runner.service").read_text()
    installer = (agent.REPO / "gymhub" / "board" / "install.sh").read_text()
    assert "RestartPreventExitStatus=78" in unit and "RestartPreventExitStatus=78" in installer
    assert "GYMHUB_STORE" not in unit and "GOOGLE_CLOUD_PROJECT" not in unit  # no cloud identity in the default unit
