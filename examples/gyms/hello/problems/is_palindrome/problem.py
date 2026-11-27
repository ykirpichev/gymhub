"""Grader for is_palindrome."""

from gymkit.spec import Problem


def reference(text):
    cleaned = [ch.lower() for ch in text if ch.isalnum()]
    return cleaned == cleaned[::-1]


def functional_cases():
    return [
        {"_name": "example", "text": "A man, a plan, a canal: Panama"},
        {"_name": "not a palindrome", "text": "gym"},
        {"_name": "empty", "text": ""},
        {"_name": "digits", "text": "12 3 21"},
        {"_name": "case only", "text": "Aa"},
    ]


def build() -> Problem:
    return Problem(
        slug="is_palindrome",
        title="Palindrome check",
        difficulty="easy",
        summary="Is a string the same read backwards, ignoring case and non-letters?",
        params=["text"],
        reference=reference,
        functional_cases=functional_cases,
    )
