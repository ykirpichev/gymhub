"""Shared by the standalone baseline and the pool roller: the prompts, the engine, one way to count."""

import os
import random
import time

SUBJECTS = ["a lighthouse keeper", "a retired chess champion", "a city of bees", "an orbital repair crew",
            "a bakery at 4 a.m.", "a lost satellite", "two rival cartographers", "a library that rearranges itself",
            "a desert railway", "a whale who collects songs", "a clockmaker's apprentice", "a winter market"]
FORMS = ["a short story", "a letter", "a diary entry", "a news report", "a dialogue", "a fable"]
TONES = ["hopeful", "dry and funny", "melancholy", "tense", "warm", "matter-of-fact"]


def prompts(count: int, seed: int = 0) -> list[str]:
    """Deterministic, varied prompts: every run, standalone or pooled, generates for the same text."""
    rng = random.Random(seed)
    return [f"Write {rng.choice(FORMS)} about {rng.choice(SUBJECTS)}, {rng.choice(TONES)} in tone. "
            f"Detail number {i}: mention the color {rng.choice(['amber', 'teal', 'crimson', 'slate', 'ivory'])}."
            for i in range(count)]


def engine(model: str):
    """vLLM's engine, configured from the environment the same way for both runs."""
    from vllm import EngineArgs, LLMEngine

    args = EngineArgs(model=model, max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
                      gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.80")),
                      max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "256")), seed=0)
    return LLMEngine.from_engine_args(args)


def params():
    from vllm import SamplingParams

    # A fixed output length (ignore_eos) makes the token count the same in every run.
    return SamplingParams(max_tokens=int(os.environ.get("MAX_TOKENS", "256")), temperature=0.8, ignore_eos=True, seed=0)


def apply_chat(engine, text: str) -> str:
    tokenizer = engine.get_tokenizer()
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


class Feeder:
    """Keep the engine full: add requests while fewer than `target` are in flight, step, collect.

    `more()` returns the next list of (request_id, prompt) or None when there is nothing left right
    now; `done(request_id, output)` gets each finished request."""

    def __init__(self, engine, target: int):
        self.engine, self.target = engine, target
        self.sampling = params()
        self.generated = 0

    def run(self, more, done, stop) -> None:
        while True:
            if self.engine.get_num_unfinished_requests() < self.target:
                batch = more()
                for request_id, text in batch or []:
                    self.engine.add_request(request_id, apply_chat(self.engine, text), self.sampling)
            if not self.engine.has_unfinished_requests():
                if stop():
                    return
                time.sleep(0.05)
                continue
            for output in self.engine.step():
                if output.finished:
                    self.generated += len(output.outputs[0].token_ids)
                    done(output.request_id, output)
