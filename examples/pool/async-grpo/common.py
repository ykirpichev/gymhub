"""Shared by the trainer and the rollers: the task, the reward, and how a prompt is rendered."""

import random
import re

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'  # as in the S1 spike


def problem(seed: int, index: int) -> tuple[str, int]:
    """A deterministic arithmetic question and its answer."""
    rng = random.Random(seed * 1000 + index)
    a, b, c = rng.randint(2, 99), rng.randint(2, 9), rng.randint(2, 199)
    return f"What is {a} * {b} + {c}?", a * b + c


def reward(text: str, answer: int) -> float:
    """1 for the right number after ####, 0.1 for the format alone, else 0."""
    found = re.findall(r"####\s*(-?\d+)", text)
    if not found:
        return 0.0
    return 1.0 if int(found[-1]) == answer else 0.1


def render(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": f"{question} {INSTRUCTION}"}],
                                         tokenize=False, add_generation_prompt=True)
