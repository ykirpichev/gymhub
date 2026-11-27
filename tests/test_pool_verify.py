"""M6 verifiers that run in jobs: TOPLOC proofs (pool/verify.py, pool_verify in a job) and update sanity
on in-memory updates (pool/reducers.py outliers, used by the DiLoCo example)."""

import ast
from pathlib import Path

import numpy as np
import pytest

from pool import reducers, verify
from pool.jobsocket import VERIFIERS


def test_pool_verify_imports_only_the_standard_library_at_the_top():
    tree = ast.parse(Path(VERIFIERS).read_text())
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [alias.name for node in top for alias in node.names] == ["os"]


def test_judge_thresholds():
    honest = [(9, 5.2, 3.0), (19, 7.7, 6.0), (2, 2.8, 2.0)]  # the worst honest chunks measured on a GB10
    assert verify.judge(honest)["ok"]
    fp8 = [(125, 67.3, 73.0), (128, 2.0**64, 2.0**64), (126, 49.0, 49.0)]
    found = verify.judge(fp8)
    assert not found["ok"] and found["failed"] == 3 and found["detail"].startswith("toploc: 3 of 3 chunks fail")
    assert not verify.judge(honest + [(40, 3.0, 2.0)])["ok"]  # one bad chunk is enough
    assert not verify.judge([])["ok"]


def test_logprob_gap():
    assert verify.logprob_gap([-0.5, -1.0], [-0.5, -1.5]) == pytest.approx(0.25)
    assert verify.logprob_gap([-0.5], [-0.5, -1.0]) == float("inf")  # a logprob per token, or it fails
    assert verify.logprob_gap([], []) == 0.0


def test_toploc_round_trip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("toploc")
    torch.manual_seed(0)
    states = torch.randn(70, 64, dtype=torch.bfloat16)  # a prompt of 6 tokens and 64 generated: 1 + 2 chunks
    proofs = verify.toploc_proofs(states, 6)
    assert len(proofs) == 3
    assert verify.toploc_verify(states * 1.002, 6, proofs)["ok"]  # the noise of another kernel
    assert not verify.toploc_verify(torch.randn(70, 64, dtype=torch.bfloat16), 6, proofs)["ok"]  # another model
    assert not verify.toploc_verify(states, 6, proofs[:2])["ok"]  # a missing chunk
    assert not verify.toploc_verify(states, 6, ["not a proof"] * 3)["ok"]


def test_chunks_checks_its_input():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        verify._chunks(torch.zeros(4, 8), 5)


def test_outliers_drop_scaled_flipped_and_random_updates():
    rng = np.random.default_rng(0)
    common = rng.normal(size=4000)
    honest = [common + rng.normal(scale=1.5, size=4000) for _ in range(4)]  # cosine with each other about 0.3
    own = honest[3]  # what the bad contributor computed honestly, before it poisons it
    norm = np.linalg.norm(own)
    for poisoned in (10 * own, -own, rng.normal(size=4000) / np.sqrt(4000) * norm):
        assert reducers.outliers(honest[:3] + [poisoned], min_cosine=0.1) == [3]
    assert reducers.outliers(honest[:2] + [10 * own]) == [2]
    assert reducers.outliers(honest[:2] + [-own]) == [2]
    assert reducers.outliers(honest, min_cosine=0.1) == []
    assert reducers.outliers(honest[:1] + [-honest[0]]) == []  # two cannot outvote each other
