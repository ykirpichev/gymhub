"""Verifiers that run inside jobs (M6, #45 section 9), importable in a job as pool_verify (it sits next
to pool_job). What they find goes to the server with pool_job.report, so a checked pool's per-machine
reputation and exclusion apply (pool/trust.py).

TOPLOC proofs (Ong et al., "TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable
Inference", 2025; the `toploc` package, which has to be in the job's image). A worker that generates
text commits to the model's last hidden states: for every chunk (the prompt, then each 32 generated
tokens) the 128 largest values, encoded as a small polynomial (about 260 bytes of base64 per chunk).
A verifier with the claimed model runs one prefill of prompt + completion and checks that the same
positions hold the same values, up to the noise of a different GPU, kernel or batch. A worker that
used another model, another adapter, another precision (fp8 for bf16) or another prompt fails.

    states = pool_verify.VllmStates(llm)                 # worker, vLLM with enforce_eager=True
    proofs = pool_verify.toploc_proofs(states(prompt_ids, ids, lora), len(prompt_ids))
    ...
    found = pool_verify.toploc_verify(pool_verify.hf_states(model, prompt_ids, ids), len(prompt_ids), proofs)
    pool_job.report(machine, found["ok"], "toploc", found["detail"])

What it does not prove: which tokens were sampled. A worker may still pick its own tokens and then
run the right model over them; check the sampling with the logprobs (`logprob_gap`) and the outcome
with a reward re-check. Proofs are cheap to make (one prefill, a few ms for a short completion) and
the same cost as recomputing logprobs to verify, so a trainer can do both in one forward pass.

Only the standard library is imported here; torch and toploc are imported when a function needs them.
"""

import os

TOPK = 128  # values committed per chunk
BATCH = 32  # generated tokens per chunk
# A chunk fails when more than this many of its TOPK values have another exponent, or the mantissas
# of the rest differ by more than these (in bf16 units of the last place). Measured on a GB10 with
# Qwen2.5-0.5B and a LoRA: honest vLLM-to-transformers worst cases 19 / 7.7 / 6; fp8 for bf16: 122+.
MAX_EXP_MISMATCHES = 38
MAX_MANT_ERR_MEAN = 10.0
MAX_MANT_ERR_MEDIAN = 8.0


def _chunks(states, prompt_len: int) -> list:
    """TOPLOC's layout: the prompt's states as one prefill chunk, then one entry per generated token."""
    states = states.detach().to("cpu")
    if states.dim() != 2 or prompt_len < 1 or prompt_len > states.shape[0]:
        raise ValueError(f"expected [tokens, hidden] states with a prompt of {prompt_len}, got {tuple(states.shape)}")
    return [states[:prompt_len]] + list(states[prompt_len:])


def toploc_proofs(states, prompt_len: int, topk: int = TOPK, batch: int = BATCH) -> list[str]:
    """Proofs (base64 strings) for the last hidden states of prompt + completion, shape [tokens, hidden]."""
    import toploc

    return toploc.build_proofs_base64(_chunks(states, prompt_len), decode_batching_size=batch, topk=topk)


def toploc_verify(states, prompt_len: int, proofs: list[str], topk: int = TOPK, batch: int = BATCH) -> dict:
    """Check proofs against the verifier's own states for the same tokens. Returns {ok, failed, chunks,
    worst: (exp mismatches, mantissa error mean, median), detail}."""
    import toploc

    chunks = _chunks(states, prompt_len)
    expected = 1 + (len(chunks) - 1 + batch - 1) // batch
    if not isinstance(proofs, list) or len(proofs) != expected:
        return {"ok": False, "failed": expected, "chunks": expected, "worst": None,
                "detail": f"toploc: {len(proofs) if isinstance(proofs, list) else 'no'} proofs for {expected} chunks"}
    try:
        results = toploc.verify_proofs_base64(chunks, proofs, decode_batching_size=batch, topk=topk)
    except Exception as exc:  # a malformed proof is a failed proof
        return {"ok": False, "failed": expected, "chunks": expected, "worst": None, "detail": f"toploc: unreadable proof ({exc})"}
    return judge([(r.exp_mismatches, r.mant_err_mean, r.mant_err_median) for r in results])


def judge(results: list[tuple[int, float, float]]) -> dict:
    """Apply the thresholds to per-chunk (exp mismatches, mantissa error mean, median) results."""
    failed = [r for r in results if r[0] > MAX_EXP_MISMATCHES or r[1] > MAX_MANT_ERR_MEAN or r[2] > MAX_MANT_ERR_MEDIAN]
    worst = max(results, key=lambda r: (r[0], r[1])) if results else None
    shown = f"{worst[0]}/{TOPK} exponents differ, mantissa error {min(worst[1], 999):.1f}" if worst else "no chunks"
    ok = bool(results) and not failed
    return {"ok": ok, "failed": len(failed), "chunks": len(results), "worst": worst,
            "detail": f"toploc: {len(failed)} of {len(results)} chunks fail (worst {shown})" if not ok
            else f"toploc: {len(results)} chunks match (worst {shown})"}


def logprob_gap(reported: list[float], recomputed) -> float:
    """Mean absolute difference between reported sampling logprobs and the verifier's own (infinite
    when the counts differ: a logprob per token is part of the claim)."""
    values = [float(v) for v in (recomputed.tolist() if hasattr(recomputed, "tolist") else recomputed)]
    if len(values) != len(reported):
        return float("inf")
    return sum(abs(float(a) - b) for a, b in zip(reported, values)) / len(values) if values else 0.0


def hf_states(model, prompt_ids: list[int], ids: list[int]):
    """Last hidden states (after the final norm) of a transformers model for prompt + completion."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.tensor([list(prompt_ids) + list(ids)], device=device), output_hidden_states=True)
    return out.hidden_states[-1][0]


class VllmStates:
    """Last hidden states from a vLLM engine, for the worker's side of TOPLOC.

    It hooks the final norm of the model (model.model.norm: Llama, Qwen, Mistral and most decoder
    models) and runs one prefill of prompt + completion per call, with a fresh cache salt so that no
    cached prefix is skipped. Needs LLM(..., enforce_eager=True) (hooks do not run inside CUDA graphs)
    and VLLM_ENABLE_V1_MULTIPROCESSING=0 set before vLLM is imported (the engine in this process)."""

    def __init__(self, llm):
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise RuntimeError("set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing vllm, for VllmStates")
        self.llm, self.captured = llm, []
        captured = self.captured

        def install(model):
            def hook(_module, _inputs, output):
                captured.append((output[0] if isinstance(output, tuple) else output).detach().clone())
            model.model.norm.register_forward_hook(hook)
            return True

        llm.apply_model(install)

    def __call__(self, prompt_ids: list[int], ids: list[int], lora_request=None):
        import torch
        from vllm import SamplingParams

        tokens = list(prompt_ids) + list(ids)
        self.captured.clear()
        self.llm.generate({"prompt_token_ids": tokens, "cache_salt": os.urandom(8).hex()}, SamplingParams(max_tokens=1),
                          lora_request=lora_request, use_tqdm=False)
        states = torch.cat([c for c in self.captured if c.dim() == 2])
        self.captured.clear()
        if states.shape[0] < len(tokens):
            raise RuntimeError(f"captured {states.shape[0]} states for {len(tokens)} tokens (is enforce_eager on?)")
        return states[-len(tokens):]
