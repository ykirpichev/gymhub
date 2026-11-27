"""Essay rubrics grade what the prompt asks (#55).

The grader scores an answer against every numbered rubric line, with a pass bar of 0.7, so a line
the prompt never asks for makes a direct answer fail. These checks keep the rubrics short and
numbered; whether each line is asked is reviewed by hand, and the known contradictions are pinned.
"""

import json
import re
from pathlib import Path

import pytest

GYMS = Path(__file__).resolve().parents[1] / "gyms"
ESSAYS = sorted(p.parent for p in GYMS.glob("*fundamentals/problems/*/problem.json")
                if json.loads(p.read_text()).get("kind") == "essay")


def _rubric_lines(problem: Path) -> list[str]:
    rubric = json.loads((problem / "answer.json").read_text())["rubric"]
    return [line for line in rubric.splitlines() if re.match(r"\s*\d+\.\s", line)]


def test_all_essays_are_found():
    assert len(ESSAYS) >= 16


@pytest.mark.parametrize("problem", ESSAYS, ids=lambda p: p.name)
def test_rubric_is_short_enough_for_a_direct_answer(problem):
    lines = _rubric_lines(problem)
    key = json.loads((problem / "answer.json").read_text())
    assert 3 <= len(lines) <= 5, f"{problem.name}: {len(lines)} rubric lines"
    assert key.get("pass_score", 0.7) <= 0.7


def test_diffusion_rubric_asks_for_one_faster_sampler_like_the_prompt():
    problem = GYMS / "llm-fundamentals/problems/diffusion_vs_autoregressive"
    assert "name one way to cut the number of denoising steps" in (problem / "PROBLEM.md").read_text()
    rubric = "\n".join(_rubric_lines(problem))
    assert "one way to cut" in rubric.lower() and "at least two" not in rubric.lower()
    assert "latent diffusion" not in rubric.lower() and "masked" not in rubric.lower()


@pytest.mark.parametrize("slug, absent", [
    ("vlm_and_vla", ["Flamingo", "instruction-tuning"]),
    ("evaluate_rag_system", ["Elo", "red-teaming", "agent"]),
    ("compare_post_training", ["SimPO", "KTO", "ORPO"]),
    ("explain_transformer_block", ["SwiGLU"]),
])
def test_unasked_topics_are_not_required(slug, absent):
    rubric = "\n".join(_rubric_lines(GYMS / "llm-fundamentals/problems" / slug))
    for word in absent:
        assert word not in rubric, f"{slug} still requires {word!r}"
