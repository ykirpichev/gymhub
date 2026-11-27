"""A job that never stops printing, or fills the disk, cannot take the shared machine down (#54)."""

import sys
import time

import pytest

from gymhub.runner import agent

NOISY = [sys.executable, "-c", "import sys\nwhile True: sys.stdout.write('x' * 99 + '\\n')"]
LONG_LINE = [sys.executable, "-c", "import sys\nwhile True: sys.stdout.write('y' * 100000)"]  # never a newline


class Backend:
    def __init__(self, cancel_after: float | None = None):
        self.cancel_after, self.started, self.sent = cancel_after, time.time(), []

    def update(self, job_id, output):
        if output is not None:
            self.sent.append(len(output))
        return self.cancel_after is not None and time.time() - self.started > self.cancel_after


@pytest.fixture()
def runner(monkeypatch):
    peak = {"size": 0}

    class Recording(agent.OutputTail):
        def add(self, text):
            super().add(text)
            peak["size"] = max(peak["size"], self.size)

    monkeypatch.setattr(agent, "OutputTail", Recording)
    monkeypatch.setattr(agent, "MAX_OUTPUT", 10_000)
    made = lambda backend=None: agent.Runner(backend or Backend(), {}, "t")  # noqa: E731
    made.peak = peak
    return made


def test_output_past_the_print_limit_stops_the_job(runner, monkeypatch):
    monkeypatch.setattr(agent, "MAX_PRINTED", 2_000_000)
    started = time.time()
    text, ok, _code = runner()._stream("j", NOISY, 60)
    assert not ok and "printed more than 2,000,000 characters" in text
    assert time.time() - started < 20
    assert len(text) <= 10_000 + 200 and text.startswith(agent.CUT_NOTE)
    assert runner.peak["size"] <= 3 * 10_000 + 65_536  # what the runner held: the tail, not the output


def test_a_job_that_prints_without_newlines_is_bounded_too(runner, monkeypatch):
    monkeypatch.setattr(agent, "MAX_PRINTED", 3_000_000)
    text, ok, _code = runner()._stream("j", LONG_LINE, 60)
    assert not ok and "Stopped" in text and len(text) <= 10_000 + 200
    assert runner.peak["size"] <= 3 * 10_000 + 65_536


def test_the_buffer_stays_bounded_through_a_timeout(runner, monkeypatch):
    monkeypatch.setattr(agent, "MAX_PRINTED", 10**12)  # only the time limit applies
    text, ok, _code = runner()._stream("j", NOISY, 3)
    assert not ok and "time limit" in text and len(text) <= 10_000 + 200
    assert runner.peak["size"] <= 3 * 10_000 + 65_536


def test_the_buffer_stays_bounded_through_a_cancel(runner, monkeypatch):
    monkeypatch.setattr(agent, "MAX_PRINTED", 10**12)
    backend = Backend(cancel_after=2)
    text, ok, _code = runner(backend)._stream("j", NOISY, 60)
    assert not ok and "Cancelled" in text and len(text) <= 10_000 + 200
    assert runner.peak["size"] <= 3 * 10_000 + 65_536
    assert backend.sent and max(backend.sent) <= 10_000 + 200  # only the tail goes to the board


def test_normal_output_is_kept_whole(runner):
    text, ok, code = runner()._stream("j", [sys.executable, "-c", "print('héllo'); print('done')"], 30)
    assert ok and code == 0 and text == "héllo\ndone\n"
