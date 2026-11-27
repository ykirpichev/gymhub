"""Live job output over Server-Sent Events: deltas as output grows, then the final state, then the stream ends."""

import json
import threading
import time

from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)


def test_events_stream_output_deltas_until_done(board):
    owner = client(board)
    sign_in(owner, "owner")
    job_id = owner.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}, headers=H).json()["id"]

    def worker():
        time.sleep(0.8)
        board.store.set("jobs", job_id, {"status": "running", "output": "line 1\n"}, merge=True)
        time.sleep(0.8)
        board.store.set("jobs", job_id, {"output": "line 1\nline 2\n"}, merge=True)
        time.sleep(0.8)
        board.store.set("jobs", job_id, {"status": "done", "ok": True, "output": "line 1\nline 2\ndone\n"}, merge=True)

    threading.Thread(target=worker, daemon=True).start()
    events = []
    with owner.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    statuses = [event["status"] for event in events]
    assert statuses[0] == "queued" and statuses[-1] == "done"
    text = ""
    for event in events:
        text = event["output"] if "output" in event else text + event.get("append", "")
    assert text == "line 1\nline 2\ndone\n"
    assert any(event.get("append") == "line 2\n" for event in events)  # only the new part is sent


def test_events_are_private(board):
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com", "school.edu"], "members_can_run": True}, headers=H)
    ann, bob = client(board), client(board)
    sign_in(ann, "ann")
    sign_in(bob, "bob")
    job_id = ann.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}, headers=H).json()["id"]
    assert bob.get(f"/api/jobs/{job_id}/events").status_code == 404
