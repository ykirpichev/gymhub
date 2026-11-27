"""The AI tutor (issue #18): actions, what it may see, owner switch, monthly budgets. Claude is faked."""

import time
from types import SimpleNamespace

import pytest

from gymhub import tutor
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_access import UIDS, _gym, people  # noqa: F401  (fixture)


class FakeClaude:
    def __init__(self, text="Look at the loop bound.", stop="end_turn", input_tokens=2000, output_tokens=500):
        self.calls = []
        self.reply = SimpleNamespace(
            stop_reason=stop, content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


@pytest.fixture()
def claude(monkeypatch):
    fake = FakeClaude()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(tutor, "_client", lambda: fake)
    return fake


def _ask(c, action, **body):
    return c.post("/api/gyms/g1/problems/p/tutor", json={"action": action, "lang": "python", **body}, headers=H)


def _setup(board):
    _gym(board, visibility="public")
    board.store.set("problems", "g1__p", {"statement": "Return the sum.", "hints": ["Use a loop."]}, merge=True)


def test_not_offered_without_a_key(board, people, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _setup(board)
    assert people["eve"].get("/api/gyms/g1").json()["gym"]["tutor"] is False
    assert _ask(people["eve"], "review", code="x = 1").status_code == 503


def test_review_and_hint_see_only_what_the_learner_sees(board, people, claude):
    _setup(board)
    eve = people["eve"]
    assert eve.get("/api/gyms/g1").json()["gym"]["tutor"] is True
    reply = _ask(eve, "review", code="def solve(a): return a")
    assert reply.status_code == 200 and reply.json()["text"] == "Look at the loop bound."
    request = claude.calls[-1]
    assert request["model"] == "claude-sonnet-5-5" and request["fallbacks"] == "default"
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    prompt = request["messages"][0]["content"]
    assert "def solve(a): return a" in prompt and "Return the sum." in prompt and "Use a loop." in prompt
    assert _ask(eve, "hint").status_code == 200  # a hint needs no code
    assert _ask(eve, "review", code="  ").status_code == 400
    assert _ask(eve, "solve-it-for-me", code="x").status_code == 400


def test_explain_uses_your_own_failed_submission(board, people, claude):
    _setup(board)
    def job(uid, ok, output="case 2: expected 5, got 4"):
        job_id = f"j{time.time_ns()}"
        board.store.set("jobs", job_id, {"gym": "g1", "slug": "p", "problem": "g1__p", "lang": "python", "action": "test", "uid": uid,
                                         "isolated": True, "status": "done", "ok": ok, "output": output, "code": "def solve(a): return a - 1",
                                         "created_at": time.time(), "finished_at": time.time()})
        return job_id
    mine, passed, theirs = job(UIDS["eve"], False), job(UIDS["eve"], True), job(UIDS["bob"], False)
    eve = people["eve"]
    assert _ask(eve, "explain", job_id=mine).status_code == 200
    prompt = claude.calls[-1]["messages"][0]["content"]
    assert "expected 5, got 4" in prompt and "return a - 1" in prompt
    assert _ask(eve, "explain", job_id=theirs).status_code == 404
    assert _ask(eve, "explain", job_id=passed).status_code == 400
    assert _ask(eve, "explain").status_code == 404


def test_owner_can_turn_it_off_and_private_gyms_stay_private(board, people, claude):
    _setup(board)
    ann = people["ann"]
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "public", "tutor": False}, headers=H).status_code == 200
    assert people["eve"].get("/api/gyms/g1").json()["gym"]["tutor"] is False
    assert _ask(people["eve"], "hint").status_code == 403
    board.store.set("gyms", "g1", {"visibility": "private", "tutor": True}, merge=True)
    assert _ask(people["eve"], "hint").status_code == 404
    assert claude.calls == []


def test_monthly_budget(board, people, claude, monkeypatch):
    _setup(board)
    monkeypatch.setenv("GYMHUB_TUTOR_MONTHLY_USD", "0.025")
    eve = people["eve"]
    # Sonnet 5.5: 2000 input + 500 output tokens = $0.004 + $0.005 = $0.009 per request.
    for _ in range(3):
        assert _ask(eve, "hint").status_code == 200
    usage = eve.get("/api/my/tutor").json()
    assert usage["budget"] == 0.025 and usage["usage"]["requests"] == 3 and usage["usage"]["usd"] == pytest.approx(0.027)
    over = _ask(eve, "hint")
    assert over.status_code == 429 and "budget" in over.json()["detail"]
    assert len(claude.calls) == 3
    assert _ask(people["owner"], "hint").status_code == 200  # admins have no budget
    assert people["owner"].get("/api/my/tutor").json()["budget"] is None


def test_refusals_are_explained(board, people, monkeypatch):
    _setup(board)
    fake = FakeClaude(text="", stop="refusal")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(tutor, "_client", lambda: fake)
    assert "can't help" in _ask(people["eve"], "hint").json()["text"]


def test_board_wide_monthly_cap(board, people, claude, monkeypatch):
    _setup(board)
    monkeypatch.setenv("GYMHUB_AI_MONTHLY_USD", "0.02")
    # Two requests at $0.009 each stay under $0.02; the third passes it, by one request at most.
    assert [_ask(people[name], "hint").status_code for name in ("eve", "bob", "eve")] == [200, 200, 200]
    assert tutor.board_usage(board.store)["usd"] == pytest.approx(0.027)
    for name in ("eve", "bob", "owner"):  # admins too: the cap protects the bill
        over = _ask(people[name], "hint")
        assert over.status_code == 429 and "paused until the 1st" in over.json()["detail"]
    assert len(claude.calls) == 3
    assert people["eve"].get("/api/my/tutor").json()["paused"] is True
    # An admin raises the cap on the Admin page; that setting wins over the env var.
    owner = people["owner"]
    ai = owner.get("/api/admin/users").json()["ai"]
    assert ai["monthly_usd"] == 0.02 and ai["spent"]["requests"] == 3
    assert people["eve"].put("/api/admin/ai", json={"monthly_usd": 5}, headers=H).status_code == 403
    assert owner.put("/api/admin/ai", json={"monthly_usd": -1}, headers=H).status_code == 422
    assert owner.put("/api/admin/ai", json={"monthly_usd": 5}, headers=H).status_code == 200
    assert _ask(people["eve"], "hint").status_code == 200
    assert owner.put("/api/admin/ai", json={"monthly_usd": 0}, headers=H).status_code == 200  # 0: no cap
    assert not tutor.board_cap_reached(board.store)


def test_charged_at_the_served_models_price():
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    assert tutor.cost(usage) == pytest.approx(2.0)  # the default, claude-sonnet-5-5
    assert tutor.cost(usage, "claude-opus-4-8") == pytest.approx(5.0)  # a fallback
    assert tutor.cost(usage, "some-new-model") == pytest.approx(10.0)  # unknown: the highest rate
