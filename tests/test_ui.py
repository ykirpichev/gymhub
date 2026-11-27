"""Headless-browser checks of the board's pages (issue #24): each main page must load without a
JavaScript error, a console error or the board's "Something went wrong" panel. On phones (issue #112)
they must also fit the screen's width, and an exercise page must keep Run and Submit on screen.

Starts a real board (uvicorn, dev auth, a fresh SQLite store seeded with two packaged gyms) and opens
the pages with Playwright's Chromium through Node (tests/ui/check_pages.cjs). Skips when Node or
Playwright with its Chromium is not installed, unless GYMHUB_UI_CHECKS=required (CI, which installs
them: .github/workflows/tests.yml).
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from gymhub import packages
from gymhub.store import SqliteStore, now, problem_id

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tests" / "ui" / "check_pages.cjs"


def _playwright_ready() -> str | None:
    """Why the browser check cannot run here, or None when it can."""
    if not shutil.which("node"):
        return "node is not installed"
    probe = "const {chromium} = require('playwright'); require('fs').accessSync(chromium.executablePath())"
    result = subprocess.run(["node", "-e", probe], cwd=REPO, capture_output=True, text=True, timeout=60)
    if result.returncode:
        return "Playwright for Node, or its Chromium, is not installed (npm install playwright; npx playwright install chromium)"
    return None


def _seed(store: SqliteStore, gym_id: str) -> dict:
    """What a machine publishes for a local gym (gymhub/board/workers.py publish_catalog)."""
    catalog = packages.public_catalog(packages.load(gym_id))
    for problem in catalog["problems"]:
        store.set("problems", problem_id(gym_id, problem["slug"]), {**problem, "gym": gym_id, "updated_at": now()})
    store.set("gyms", gym_id, {**catalog["gym"], "id": gym_id, "problems": len(catalog["problems"]), "updated_at": now()})
    return catalog


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def live_board(tmp_path):
    reason = _playwright_ready()
    if reason and os.environ.get("GYMHUB_UI_CHECKS") == "required":  # CI: a missing browser is a failure
        pytest.fail(reason)
    if reason:
        pytest.skip(reason)
    db = tmp_path / "board.sqlite"
    store = SqliteStore(db)
    code = _seed(store, "algo")
    questions = _seed(store, "ml-fundamentals")
    store.set("jobs", "seeded", {"gym": "algo", "slug": code["problems"][0]["slug"], "problem": problem_id("algo", code["problems"][0]["slug"]),
                                 "lang": "python", "action": "test", "status": "done", "ok": True, "output": "ok\n",
                                 "created_at": now() - 30, "started_at": now() - 20, "finished_at": now() - 10,
                                 "uid": "local-dev", "runner": "box", "host": "jetson", "via": "box", "hardware": "Orin"})
    # An online pool bridge (its heartbeat in the future keeps it online through the check), so the
    # exercise toolbar shows its "Run on" choice: the runner and its pool machines.
    pool = {"actions": ["bench", "run", "test"], "machines": 2, "gpus": [], "tools": ["g++"], "own": {"machines": 2, "gpus": []},
            "list": [{"name": "spark", "gpus": [{"name": "NVIDIA GB10"}], "own": True}, {"name": "jetson", "gpus": [{"name": "Orin"}], "own": True}]}
    store.set("runners", "box", {"name": "box", "heartbeat_at": now() + 3600, "gyms": ["algo"],
                                 "capabilities": {"arch": "x86_64", "gpus": [], "tools": ["g++"], "sandbox": True, "bundles": True, "pool": pool}})
    # Compute: one machine you added and connected, one you added that never connected, one of someone else's.
    caps = {"gpus": [{"name": "Orin (nvgpu)", "compute_capability": "8.7"}], "cuda": "12.6", "cpus": 6, "memory_gb": 7, "sandbox": True}
    for worker_id, name, seen in (("w-mine", "jetson", now()), ("w-never", "spark", None)):
        store.set("workers", worker_id, {"name": name, "owner_uid": "local-dev", "owner_email": "local-dev", "token_hash": "x",
                                         "created_at": now(), "last_seen": seen, "revoked": False})
    for worker_id, name in (("w-mine", "jetson"), ("w-other", "spark-10dc")):
        store.set("runners", worker_id, {"name": name, "worker_id": worker_id, "gpu": "GPU", "platform": "Linux aarch64",
                                         "gyms": ["algo"], "capabilities": caps, "heartbeat_at": now()})
    port = _free_port()
    env = {**os.environ, "GYMHUB_STORE": f"sqlite:{db}", "GYMHUB_AUTH": "dev", "GYMHUB_TRUST_JOBS": "1"}
    for name in ("GYMHUB_STORE_PREFIX", "K_SERVICE"):
        env.pop(name, None)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "gymhub.board.app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while True:
            try:
                urllib.request.urlopen(f"{base}/healthz", timeout=2)
                break
            except OSError:
                if server.poll() is not None or time.time() > deadline:
                    pytest.fail(f"the board did not start:\n{server.stdout.read() if server.poll() is not None else ''}")
                time.sleep(0.2)
        code_slug = code["problems"][0]["slug"]
        # One question of each kind the gym has (answer, choice, order, match, fill, essay, lesson...).
        kinds = {}
        for problem in questions["problems"]:
            kinds.setdefault(problem["kind"], problem["slug"])
        yield base, ["#/", "#/gyms", "#/gym/algo", "#/gym/ml-fundamentals", f"#/gym/algo/{code_slug}",
                     *(f"#/gym/ml-fundamentals/{slug}" for kind, slug in sorted(kinds.items()) if kind != "code"),
                     "#/compute", "#/admin", "#/new-gym"]
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


# A desktop window, then two phones (issue #112): Safari's and Chrome's screen sizes and touch.
DEVICES = ["desktop", "iPhone 13", "Pixel 7"]


def test_main_pages_load_without_errors(live_board):
    base, pages = live_board
    result = subprocess.run(["node", str(SCRIPT), f"--devices={','.join(DEVICES)}", base, *pages],
                            cwd=REPO, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"the browser check could not run:\n{result.stderr}"
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(report) == set(DEVICES)
    problems = {}
    for device, seen in report.items():
        assert set(seen) == set(pages), device
        problems.update({f"{device} {page}": found for page, found in seen.items() if found})
    assert not problems, "pages with errors:\n" + "\n".join(f"{page}: {found}" for page, found in problems.items())
