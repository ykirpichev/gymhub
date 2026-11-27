"""Grader for sum_list. Private: learners never see this file, only PROBLEM.md, HINTS.md and the starter."""

import random

from gymkit.spec import Problem


def reference(nums):
    return sum(nums)


def functional_cases():
    rng = random.Random(1)
    return [
        {"_name": "example", "nums": [1, 2, 3]},
        {"_name": "empty", "nums": []},
        {"_name": "negatives", "nums": [-5, 2, -1]},
        {"_name": "random", "nums": [rng.randint(-1000, 1000) for _ in range(100)]},
    ]


def build() -> Problem:
    return Problem(
        slug="sum_list",
        title="Sum a list",
        difficulty="easy",
        summary="Return the sum of a list of integers.",
        params=["nums"],
        reference=reference,
        functional_cases=functional_cases,
    )
