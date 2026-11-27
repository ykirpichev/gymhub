"""Terminal colors and Markdown rendering. Plain text when output is not a terminal."""

import os
import sys

_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def _wrap(code: str):
    def apply(text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if _COLOR else text

    return apply


bold = _wrap("1")
dim = _wrap("2")
red = _wrap("31")
green = _wrap("32")
yellow = _wrap("33")
cyan = _wrap("36")


def markdown(text: str) -> None:
    """Print Markdown with formatting if `rich` is installed and stdout is a terminal."""
    if _COLOR:
        try:
            from rich.console import Console
            from rich.markdown import Markdown

            Console().print(Markdown(text, code_theme="ansi_dark", inline_code_theme="ansi_dark"))
            return
        except ImportError:
            pass
    print(text)


def bar(fraction: float, width: int = 30) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = round(fraction * width)
    return green("█" * filled) + dim("░" * (width - filled))
