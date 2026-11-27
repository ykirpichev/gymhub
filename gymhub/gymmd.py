"""Gym Markdown: a whole gym in one Markdown file, compiled into an ordinary package (gymhub/packages.py).

The language is described in gymhub/GYM_MARKDOWN.md (`python -m gymhub.cli gym guide`). In short:

    ---
    id: sql-basics
    title: SQL basics
    ---
    One paragraph that describes the gym.

    # Track name
    One sentence about the track.

    ## Exercise title
    id: capital
    difficulty: easy

    The statement, in Markdown.

    ### Choices
    - [x] Paris
    - [ ] Lyon

The compiler splits what learners see (problem.json) from the key (answer.json), shuffles ordering and
matching items so the public file never gives the answer away, and generates the grader for Python
exercises whose tests are written as `solve(...) == expected`. It never runs any of the author's code:
the board compiles uploads with it, and a machine validates the result like any other version.

    compile_text(text) -> Gym                   raises GymMarkdownError (errors with line numbers)
    write(gym, root) -> Package                 the package files under root
    check(text) -> list[dict]                   compile, write to a temporary directory, validate
    export(package) -> str                      any package back to Gym Markdown
"""

import ast
import difflib
import io
import json
import math
import random
import re
import tempfile
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

from gymhub import gymdraft, kinds, packages

GUIDE = Path(__file__).with_name("GYM_MARKDOWN.md")
MAX_TEXT = 2 * 1024 * 1024  # the largest gym in gyms/ exports to 420 KB; compiling 2 MB takes at most a few seconds
MAX_EXERCISES = 500

LANGS = {
    "python": {"id": "python", "label": "Python", "editor": "python", "file": "solution.py"},
    "cpp": {"id": "cpp", "label": "C++", "editor": "cpp", "file": "solution.cpp"},
    "cuda": {"id": "cuda", "label": "CUDA", "editor": "cpp", "file": "solution.cu"},
    "pytorch": {"id": "pytorch", "label": "PyTorch", "editor": "python", "file": "solution.py"},
    "triton": {"id": "triton", "label": "Triton", "editor": "python", "file": "triton_solution.py"},
}
LANG_ALIASES = {"py": "python", "python3": "python", "c++": "cpp", "cc": "cpp", "cxx": "cpp", "cu": "cuda", "torch": "pytorch"}

GYM_KEYS = {"id", "title", "description", "runtime", "gpu", "langs", "order", "publish"}
EXERCISE_KEYS = {
    "id", "kind", "difficulty", "summary", "minutes", "concepts", "warmup",
    "multiple", "placeholder", "case_sensitive", "match", "words", "pass_score", "params",
}
SECTIONS = {"hints", "choices", "order", "match", "blanks", "answer", "explanation", "rubric", "checklist",
            "starter", "solution", "tests", "files", "editorial"}
# A section that only one kind has decides the kind, so "kind:" is rarely needed.
KIND_OF_SECTION = {"choices": "choice", "order": "order", "match": "match", "blanks": "fill", "answer": "answer",
                   "rubric": "essay", "checklist": "checklist", "tests": "code", "starter": "code", "solution": "code",
                   "files": "code"}
SECTIONS_OF_KIND = {
    "lesson": {"hints"},
    "choice": {"choices", "hints", "explanation"},
    "order": {"order", "hints", "explanation"},
    "match": {"match", "hints", "explanation"},
    "fill": {"blanks", "hints", "explanation"},
    "answer": {"answer", "hints", "explanation"},
    "essay": {"rubric", "hints"},
    "checklist": {"checklist", "hints"},
    "code": {"starter", "solution", "tests", "files", "hints", "editorial"},
}
KEYS_OF_KIND = {
    "answer": {"placeholder", "case_sensitive", "match"},
    "fill": {"case_sensitive", "match"},
    "choice": {"multiple"},
    "essay": {"words", "pass_score"},
    "code": {"params"},
}
GENERATED = {"problem.json", "PROBLEM.md", "HINTS.md", "answer.json", "EDITORIAL.md"}

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
META = re.compile(r"^([a-z_]+):\s*(.*?)\s*$")
ITEM = re.compile(r"^ {0,3}(?:[-*+]|\d{1,3}[.)])\s+(.*)$")
TASK = re.compile(r"^\[([ xX])\]\s+(.*)$", re.S)
EXPLICIT_ID = re.compile(r"\s*\{#([A-Za-z0-9_-]{1,40})\}\s*$")
BLANK = re.compile(r"\[\[((?:(?!\[\[).)+?)\]\]")  # a blank never holds [[, so many open [[ stay linear
NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
PLUS_MINUS = re.compile(r"\s*(?:±|\+/-|\+-)\s*")
FILE_ATTR = re.compile(r"""\bfile=(?:"([^"]+)"|'([^']+)'|(\S+))""")
SAFE_PATH = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")


class GymMarkdownError(ValueError):
    """Errors as {"line": int | None, "exercise": slug | None, "message": str}."""

    def __init__(self, errors: list[dict]):
        super().__init__(format_errors(errors[:3]))
        self.errors = errors


def format_errors(errors: list[dict]) -> str:
    return "\n".join(
        f"line {e['line']}: " * bool(e.get("line")) + f"{e['exercise']}: " * bool(e.get("exercise")) + e["message"]
        for e in errors
    )


@dataclass
class Exercise:
    slug: str
    line: int
    meta: dict  # problem.json
    statement: str  # PROBLEM.md
    hints: list[str] = field(default_factory=list)
    editorial: str = ""  # EDITORIAL.md: the worked solution, shown after a first attempt
    key: dict | None = None  # answer.json
    files: dict[str, str] = field(default_factory=dict)  # other files, relative to the exercise directory
    draft: dict | None = None  # a Python exercise with tests: gymdraft writes its grader


@dataclass
class Gym:
    manifest: dict
    exercises: list[Exercise]
    lines: dict[str, int] = field(default_factory=dict)  # slug -> line of its heading


# ---- reading the file ----------------------------------------------------------------------------


@dataclass
class _Block:
    title: str
    line: int
    rows: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class _RawExercise(_Block):
    track: int = 0
    sections: list[_Block] = field(default_factory=list)
    headings: list[tuple[int, str]] = field(default_factory=list)  # ### headings that name no section


def _scan(text: str, errors: list[dict]):
    """Front matter, intro rows, tracks [(_Block)], exercises [(_RawExercise)]. Headings in code fences are text."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    front: list[tuple[int, str]] = []
    start = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                front = [(n + 1, lines[n]) for n in range(1, index)]
                start = index + 1
                break
        else:
            errors.append({"line": 1, "exercise": None, "message": "the front matter that starts with --- never ends with ---"})
            return [], [], [], []
    else:
        errors.append({"line": 1, "exercise": None, "message": "start with front matter: ---, id: <gym-id>, title: <title>, ---"})
    intro: list[tuple[int, str]] = []
    tracks: list[_Block] = []
    exercises: list[_RawExercise] = []
    target = intro
    fence, fence_line = None, 0
    for index in range(start, len(lines)):
        number, line = index + 1, lines[index]
        opened = FENCE.match(line)
        if fence:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence) and not opened.group(2).strip():
                fence = None
            target.append((number, line))
            continue
        if opened:
            fence, fence_line = opened.group(1), number
            target.append((number, line))
            continue
        heading = HEADING.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            if level == 1:
                tracks.append(_Block(title, number))
                target = tracks[-1].rows
                continue
            if level == 2:
                exercises.append(_RawExercise(title, number, track=len(tracks) - 1))
                target = exercises[-1].rows
                continue
            if exercises and exercises[-1].line > (tracks[-1].line if tracks else 0):
                if title.lower() in SECTIONS:
                    exercises[-1].sections.append(_Block(title.lower(), number))
                    target = exercises[-1].sections[-1].rows
                    continue
                exercises[-1].headings.append((number, title))
        target.append((number, line))
    if fence:
        errors.append({"line": fence_line, "exercise": None,
                       "message": f"this code block never closes: end it with a line of {fence}"})
    return front, intro, tracks, exercises


def _text(rows) -> str:
    return "\n".join(line for _, line in rows).strip("\n").rstrip()


def _dedent(rows) -> str:
    return _text(rows).strip()


def _fields(rows, allowed: set[str], where: str | None, errors: list[dict]) -> tuple[dict, list]:
    """Leading `key: value` lines (after blank lines) and the rows after them."""
    values: dict[str, tuple[str, int]] = {}
    index = 0
    while index < len(rows) and not rows[index][1].strip():
        index += 1
    while index < len(rows):
        number, line = rows[index]
        found = META.match(line)
        if not found or (not values and found.group(1) not in allowed):
            break
        key, value = found.groups()
        if key not in allowed:
            errors.append({"line": number, "exercise": where, "message": f'unknown field "{key}" (fields here: {", ".join(sorted(allowed))}); '
                           "to start the text with such a line, leave a blank line after the fields"})
        elif key in values:
            errors.append({"line": number, "exercise": where, "message": f'"{key}" is given twice'})
        else:
            values[key] = (value, number)
        index += 1
    return values, rows[index:]


def _items(rows, where: str, section: str, errors: list[dict]) -> list[tuple[int, str]]:
    """Top-level list items of a section. A line indented to the item's text (2 for `- `, 3 for `1. `) or further
    continues it, with exactly that indent removed, so code and nested lists inside an item keep their shape."""
    items: list[tuple[int, list[str]]] = []
    column = 0  # where the text of the current item starts
    for number, line in rows:
        lead = re.match(r"[ \t]*", line).group()
        if "\t" in lead:
            line = lead.expandtabs(4) + line[len(lead):]
            lead = lead.expandtabs(4)
        found = ITEM.match(line)
        if not line.strip():
            if items:
                items[-1][1].append("")
        elif items and len(lead) >= column:
            items[-1][1].append(line[column:])
        elif found:
            column = found.start(1)
            items.append((number, [found.group(1)]))
        elif items and lead:  # indented, but less than the text: still the item above
            items[-1][1].append(line.lstrip())
        else:
            errors.append({"line": number, "exercise": where, "message": f"### {section.title()} holds a list: start each entry with - or 1."})
            return []
    return [(number, "\n".join(parts).strip()) for number, parts in items]


def _split_id(text: str) -> tuple[str, str | None]:
    found = EXPLICIT_ID.search(text)
    return (text[: found.start()].rstrip(), found.group(1)) if found else (text, None)


def _auto_ids(explicit: list[str | None], style: str) -> list[str]:
    used = {i for i in explicit if i}
    out, counter = [], 0
    for given in explicit:
        if given:
            out.append(given)
            continue
        while True:
            counter += 1
            candidate = _letters(counter) if style == "letters" else f"{style}{counter}"
            if candidate not in used:
                break
        used.add(candidate)
        out.append(candidate)
    return out


def _letters(n: int) -> str:
    text = ""
    while n:
        n, rest = divmod(n - 1, 26)
        text = chr(97 + rest) + text
    return text


def _shuffled(values: list, seed: str) -> list:
    """A fixed shuffle (the same for every build), never the original order."""
    out = list(values)
    random.Random(seed).shuffle(out)
    if out == list(values) and len(out) > 1:
        out = out[1:] + out[:1]
    return out


def _fences(rows) -> list[tuple[int, str, str]]:
    """Fenced code blocks: (line, info string, content). Text outside them is commentary."""
    blocks, current = [], None
    for number, line in rows:
        if current is None:
            opened = FENCE.match(line)
            if opened:
                current = (number, opened.group(1), opened.group(2).strip(), [])
            continue
        closed = FENCE.match(line)
        if closed and closed.group(1)[0] == current[1][0] and len(closed.group(1)) >= len(current[1]) and not closed.group(2).strip():
            blocks.append((current[0], current[2], "\n".join(current[3])))
            current = None
        else:
            current[3].append(line)
    return blocks  # a block left open is reported by _scan


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if slug and not slug[0].isalpha():
        slug = "x_" + slug
    return slug[:60] or "exercise"


# ---- answers -------------------------------------------------------------------------------------


def _answer_spec(alternatives: list[str], options: dict, line: int, where: str, errors: list[dict]) -> dict:
    """Accepted answers, written `42`, `0.5 ± 0.01`, `3 ± 2%` or `Paris`, into a kinds.py answer key."""
    values, tolerance, relative = [], None, None
    for alternative in alternatives:
        parts = PLUS_MINUS.split(alternative.strip())
        values.append(parts[0].strip())
        for part in parts[1:]:
            percent = part.endswith("%")
            number = part[:-1].strip() if percent else part
            if not NUMBER.fullmatch(number):
                errors.append({"line": line, "exercise": where, "message": f"the tolerance in {alternative!r} is not a number"})
                continue
            amount = float(number) / 100 if percent else float(number)
            previous = relative if percent else tolerance
            if previous is not None and previous != amount:
                errors.append({"line": line, "exercise": where, "message": "give one tolerance for all the answers"})
            if percent:
                relative = amount
            else:
                tolerance = amount
    if not values or not all(values):
        errors.append({"line": line, "exercise": where, "message": "an empty answer"})
        return {"answers": values}
    match = options.get("match")
    numeric = all(NUMBER.fullmatch(value) for value in values)
    spec: dict = {"answers": values}
    if match == "number" or (match != "text" and numeric):
        if not numeric:
            errors.append({"line": line, "exercise": where, "message": "match: number needs numbers"})
        spec["match"] = "number"
        spec["tolerance"] = _number(tolerance or 0)
        if relative:
            spec["rel_tolerance"] = _number(relative)
    elif tolerance is not None or relative is not None:
        errors.append({"line": line, "exercise": where, "message": "± works with numbers only"})
    if options.get("case_sensitive"):
        spec["case_sensitive"] = True
    return spec


def _number(value: float):
    return int(value) if float(value).is_integer() and abs(value) < 2**53 else value


def _bool(value: str, line: int, where: str | None, errors: list[dict]) -> bool:
    if value.lower() in {"true", "yes"}:
        return True
    if value.lower() in {"false", "no"}:
        return False
    errors.append({"line": line, "exercise": where, "message": f"{value!r}: write true or false"})
    return False


def _int(value: str, line: int, where: str | None, errors: list[dict]) -> int | None:
    try:
        return int(value)
    except ValueError:
        errors.append({"line": line, "exercise": where, "message": f"{value!r} is not a whole number"})
        return None


def _list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


# ---- tests written as calls ----------------------------------------------------------------------


def _jsonable(value, line: int, where: str, errors: list[dict]):
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_jsonable(item, line, where, errors) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            errors.append({"line": line, "exercise": where, "message": "dictionary keys in tests must be strings"})
        return {str(key): _jsonable(item, line, where, errors) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append({"line": line, "exercise": where, "message": "tests cannot use inf or nan"})
        return value
    errors.append({"line": line, "exercise": where, "message": f"tests use JSON-like values: numbers, strings, lists, dicts, True/False/None, not {type(value).__name__}"})
    return None


def _comment(source: str) -> str:
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                return token.string.lstrip("#").strip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return ""


MAX_TEST_LINES = 200
MAX_TEST_BYTES = 64 * 1024
_TOKEN = re.compile(r"""[][(){}#\\]|'{3}|"{3}|['"]""")


def _open_after(line: str, depth: int, quote: str | None) -> tuple[int, str | None]:
    """Bracket depth and the open string (its quote) at the end of `line`, given those at its start. Skips comments
    and the inside of strings. Linear in the line, so a call over many lines is scanned once and parsed once."""
    position = 0
    while found := _TOKEN.search(line, position):
        token, position = found.group(), found.end()
        if quote:
            if token == "\\":
                position += 1
            elif token[0] == quote[0] and len(token) >= len(quote):
                quote, position = None, found.start() + len(quote)
        elif token == "#":
            break
        elif token in "([{":
            depth += 1
        elif token in ")]}":
            depth -= 1
        elif token != "\\":
            quote = token
    if quote and len(quote) == 1 and not line.endswith("\\"):
        quote = None  # a one-quote string ends with its line (ast.parse reports it)
    return depth, quote


def _cases(rows, params: list[str], where: str, errors: list[dict]) -> list[dict]:
    """`solve(1, 2) == 3  # name` per line (a call may run over several lines); no `== ...` uses the reference."""
    cases: list[dict] = []
    pending: list[tuple[int, str]] = []
    size, depth, quote, skipping = 0, 0, None, False
    for number, line in rows + [(0, "")]:
        if number:
            if not quote and FENCE.match(line):
                continue
            if skipping:  # the rest of a test that ran too long
                depth, quote = _open_after(line, depth, quote)
                skipping = depth > 0 or quote is not None
                depth, quote = (depth, quote) if skipping else (0, None)
                continue
            if not pending and (not line.strip() or line.strip().startswith("#")):
                continue
            depth, quote = _open_after(line, depth, quote)
            pending.append((number, line))
            size += len(line) + 1
            if len(pending) > MAX_TEST_LINES or size > MAX_TEST_BYTES:
                errors.append({"line": pending[0][0], "exercise": where, "message":
                               f"this test runs past {MAX_TEST_LINES} lines or {MAX_TEST_BYTES // 1024} KB (is a bracket left open?)"})
                pending, size, skipping = [], 0, depth > 0 or quote is not None
                depth, quote = (depth, quote) if skipping else (0, None)
                continue
            if depth > 0 or quote or line.rstrip().endswith("\\"):
                continue  # the call goes on below
        elif not pending:
            continue
        first, source = pending[0][0], "\n".join(text for _, text in pending)
        pending, size, depth, quote = [], 0, 0, None
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError:
            errors.append({"line": first, "exercise": where, "message": "a test is written solve(args) == expected"
                           + " (a bracket or string is left open)" * (not number)})
            continue
        except (RecursionError, MemoryError, ValueError):
            errors.append({"line": first, "exercise": where, "message": "this test is too deeply nested"})
            continue
        body, expected = tree.body, None
        if isinstance(body, ast.Compare):
            if len(body.ops) != 1 or not isinstance(body.ops[0], ast.Eq):
                errors.append({"line": first, "exercise": where, "message": "compare with == only"})
                continue
            body, expected = body.left, body.comparators[0]
        if not (isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id == "solve"):
            errors.append({"line": first, "exercise": where, "message": "a test calls solve(...)"})
            continue
        try:
            args = [ast.literal_eval(node) for node in body.args]
            named = {kw.arg: ast.literal_eval(kw.value) for kw in body.keywords if kw.arg}
            result = ast.literal_eval(expected) if expected is not None else None
        except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
            errors.append({"line": first, "exercise": where, "message": "tests use literal values only (no names or expressions)"})
            continue
        if len(args) > len(params) or any(name not in params for name in named) or len(args) + len(named) != len(params) \
                or any(name in named for name in params[: len(args)]):
            errors.append({"line": first, "exercise": where, "message": f"give exactly these inputs: solve({', '.join(params)})"})
            continue
        values = dict(zip(params, args)) | named
        case = {"args": {name: _jsonable(values[name], first, where, errors) for name in params}}
        if expected is not None:
            case["expected"] = _jsonable(result, first, where, errors)
        name = _comment(source)
        case = ({"name": name[:80]} if name else {}) | case
        cases.append(case)
    return cases


def _params_of(source: str) -> list[str] | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            return [arg.arg for arg in node.args.posonlyargs + node.args.args]
    return None


# ---- compiling -----------------------------------------------------------------------------------


SOURCE_LINE = re.compile(r"^- \[(?P<title>[^\]]+)\]\((?P<url>[^)\s]+)\)(?::\s*(?P<note>.*))?$")


def _sources(intro: list[tuple[int, str]], errors: list[dict]) -> tuple[list[tuple[int, str]], list[dict]]:
    """Intro lines `- [Title](https://...): note` are the gym's sources; the rest is its description."""
    rest, found = [], []
    for number, line in intro:
        match = SOURCE_LINE.match(line.strip())
        if not match:
            rest.append((number, line))
            continue
        item = {"title": match["title"].strip(), "url": match["url"]}
        if match["note"]:
            item["note"] = match["note"].strip()
        for problem in packages.source_errors(item):
            errors.append({"line": number, "exercise": None, "message": f"source: {problem}"})
        found.append(item)
    return rest, found


def _source_line(item: dict) -> str:
    return f"- [{item['title']}]({item['url']})" + (f": {item['note']}" if item.get("note") else "")


def compile_text(text: str) -> Gym:
    errors: list[dict] = []
    if len(text) > MAX_TEXT:
        raise GymMarkdownError([{"line": None, "exercise": None, "message": f"the file is over {MAX_TEXT // 1024 // 1024} MB"}])
    front, intro, tracks, raws = _scan(text, errors)
    intro, sources = _sources(intro, errors)
    values, rest = _fields(front, GYM_KEYS, None, errors)
    for number, line in rest:
        if line.strip():
            errors.append({"line": number, "exercise": None, "message": f"front matter holds key: value lines (keys: {', '.join(sorted(GYM_KEYS))})"})
            break
    get = lambda key, default=None: values[key][0] if key in values else default  # noqa: E731
    gym_id = get("id", "")
    manifest: dict = {"schema": packages.SCHEMA, "id": gym_id, "title": get("title", ""),
                      "description": get("description") or " ".join(_dedent(intro).split()),
                      "runtime": get("runtime", "gymkit")}
    for key in ("id", "title"):
        if not manifest[key]:
            errors.append({"line": 1, "exercise": None, "message": f"the front matter needs {key}:"})
    if not manifest["description"]:
        errors.append({"line": 1, "exercise": None, "message": "describe the gym: a paragraph after the front matter, before the first heading"})
    gpu = _bool(get("gpu"), values["gpu"][1], None, errors) if "gpu" in values else manifest["runtime"] == "gpu_gym"
    manifest["requires"] = {"gpu": gpu}
    if "order" in values:
        manifest["order"] = _int(get("order"), values["order"][1], None, errors)
    if "publish" in values:
        manifest["publish"] = get("publish")
    if sources:
        manifest["sources"] = sources
    lang_ids = [LANG_ALIASES.get(lang.lower(), lang.lower()) for lang in _list(get("langs", ""))]
    for lang in lang_ids:
        if lang not in LANGS:
            errors.append({"line": values["langs"][1], "exercise": None, "message": f"language {lang!r}: use {', '.join(LANGS)}"})
    if len(raws) > MAX_EXERCISES:
        errors.append({"line": None, "exercise": None, "message": f"at most {MAX_EXERCISES} exercises"})
        raws = raws[:MAX_EXERCISES]

    exercises: list[Exercise] = []
    seen: dict[str, int] = {}
    for raw in raws:
        exercise = _exercise(raw, gym_id, errors)
        if exercise is None:
            continue
        if exercise.slug in seen:
            errors.append({"line": raw.line, "exercise": exercise.slug, "message": f"another exercise (line {seen[exercise.slug]}) has this id"})
            continue
        seen[exercise.slug] = raw.line
        exercises.append(exercise)
    if not raws:
        errors.append({"line": None, "exercise": None, "message": "no exercises: each one starts with ## <title>"})

    has_code = any(item.meta["kind"] == "code" for item in exercises)
    if has_code and not lang_ids:
        lang_ids = ["python"]
    if lang_ids:
        manifest["langs"] = [dict(LANGS[lang]) for lang in lang_ids if lang in LANGS]
    for item in exercises:
        for path in item.files:
            found = re.fullmatch(r"(starter|solution)/\{(\w+)\}", path)
            if found and found.group(2) not in lang_ids:
                errors.append({"line": seen[item.slug], "exercise": item.slug,
                               "message": f"a {found.group(2)} {found.group(1)}, but langs: is {', '.join(lang_ids)}"})
        if item.draft and "python" not in lang_ids:
            errors.append({"line": seen[item.slug], "exercise": item.slug, "message": "### Tests generate a Python grader: add python to langs:"})

    track_rows = []
    for index, track in enumerate(tracks):
        slugs = [item.slug for item, raw in _paired(exercises, raws, seen) if raw.track == index]
        summary = " ".join(_dedent(track.rows).split())
        if not slugs:
            errors.append({"line": track.line, "exercise": None, "message": f"track {track.title!r} has no exercises"})
        track_rows.append({"name": track.title, **({"summary": summary} if summary else {}), "exercises": slugs})
    loose = [item.slug for item, raw in _paired(exercises, raws, seen) if raw.track < 0]
    if loose:
        track_rows.insert(0, {"name": "Exercises", "exercises": loose})
    manifest["tracks"] = track_rows
    warmup = [item.slug for item in exercises if item.meta.pop("_warmup", False)]
    if warmup:
        manifest["warmup"] = warmup
    if errors:
        raise GymMarkdownError(errors)
    return Gym(manifest=manifest, exercises=exercises, lines=dict(seen))


def _paired(exercises: list[Exercise], raws: list[_RawExercise], seen: dict[str, int]):
    by_line = {raw.line: raw for raw in raws}
    return [(item, by_line[seen[item.slug]]) for item in exercises]


def _exercise(raw: _RawExercise, gym_id: str, errors: list[dict]) -> Exercise | None:
    title = raw.title
    fields, rows = _fields(raw.rows, EXERCISE_KEYS, _slugify(title), errors)
    get = lambda key, default=None: fields[key][0] if key in fields else default  # noqa: E731
    line_of = lambda key: fields[key][1] if key in fields else raw.line  # noqa: E731
    slug = get("id") or _slugify(title)
    where = slug
    if not packages.SLUG.match(slug):
        errors.append({"line": line_of("id"), "exercise": where, "message": "id: lowercase letters, digits and _, starting with a letter"})
    sections: dict[str, _Block] = {}
    for section in raw.sections:
        if section.title in sections:
            errors.append({"line": section.line, "exercise": where, "message": f"### {section.title.title()} appears twice"})
        sections[section.title] = section
    implied = {KIND_OF_SECTION[name] for name in sections if name in KIND_OF_SECTION}
    kind = get("kind")
    if kind is None:
        if len(implied) > 1:
            errors.append({"line": raw.line, "exercise": where, "message": f"sections of different kinds: {', '.join(sorted(implied))}"})
            return None
        kind = implied.pop() if implied else "lesson"
    if kind not in SECTIONS_OF_KIND:
        errors.append({"line": line_of("kind"), "exercise": where, "message": f"kind {kind!r}: use {', '.join(sorted(SECTIONS_OF_KIND))}"})
        return None
    for name, section in sections.items():
        if name not in SECTIONS_OF_KIND[kind]:
            allowed = ", ".join(f"### {s.title()}" for s in sorted(SECTIONS_OF_KIND[kind])) or "none"
            errors.append({"line": section.line, "exercise": where, "message": f"{_a(kind)} exercise has no ### {name.title()} (its sections: {allowed})"})
    if kind != "lesson":  # an unknown ### would publish its text with the statement: "### Answer key" would leak
        for number, heading in raw.headings:
            close = difflib.get_close_matches(heading.lower(), sorted(SECTIONS_OF_KIND[kind]), n=1, cutoff=0.5)
            hint = f"did you mean ### {close[0].title()}? " if close else ""
            errors.append({"line": number, "exercise": where, "message": f"### {heading} is not a section of {_a(kind)} exercise; {hint}"
                           f"its sections: {', '.join(f'### {s.title()}' for s in sorted(SECTIONS_OF_KIND[kind]))}. "
                           "A heading inside the statement uses ####"})
    for key, (_, number) in fields.items():
        if key in {"multiple", "placeholder", "case_sensitive", "match", "words", "pass_score", "params"} and key not in KEYS_OF_KIND.get(kind, set()):
            errors.append({"line": number, "exercise": where, "message": f'"{key}" does not apply to {_a(kind)} exercise'})

    difficulty = get("difficulty", "easy" if kind == "lesson" else None)
    if difficulty not in packages.DIFFICULTIES:
        errors.append({"line": line_of("difficulty"), "exercise": where, "message": "difficulty: easy, medium or hard"})
    statement = _text(rows)
    if not statement.strip():
        errors.append({"line": raw.line, "exercise": where, "message": "write the statement under the heading (and its fields)"})
    meta: dict = {"title": title, "difficulty": difficulty, "summary": get("summary", ""), "kind": kind}
    if "minutes" in fields:
        meta["minutes"] = _int(get("minutes"), line_of("minutes"), where, errors)
    if "concepts" in fields:
        meta["concepts"] = _list(get("concepts"))
    if "warmup" in fields:
        meta["_warmup"] = _bool(get("warmup"), line_of("warmup"), where, errors)
    if "placeholder" in fields:
        meta["placeholder"] = get("placeholder")
    options = {"match": get("match"), "case_sensitive": _bool(get("case_sensitive"), line_of("case_sensitive"), where, errors)
               if "case_sensitive" in fields else False}
    if options["match"] not in {None, "text", "number"}:
        errors.append({"line": line_of("match"), "exercise": where, "message": "match: text or number"})

    exercise = Exercise(slug=slug, line=raw.line, meta=meta, statement=f"# {title}\n\n{statement.strip()}\n")
    if "hints" in sections:
        exercise.hints = [text for _, text in _items(sections["hints"].rows, where, "hints", errors)]
    explanation = _dedent(sections["explanation"].rows) if "explanation" in sections else ""
    if "editorial" in sections:
        exercise.editorial = _dedent(sections["editorial"].rows).strip()

    def need(name: str) -> _Block | None:
        if name not in sections:
            errors.append({"line": raw.line, "exercise": where, "message": f"{_a(kind)} exercise needs ### {name.title()}"})
            return None
        return sections[name]

    key: dict | None = None
    if kind == "choice" and (section := need("choices")):
        entries = []
        for number, text in _items(section.rows, where, "choices", errors):
            task = TASK.match(text)
            if not task:
                errors.append({"line": number, "exercise": where, "message": "write each choice as - [x] right or - [ ] wrong"})
                continue
            body, given = _split_id(task.group(2).strip())
            entries.append((task.group(1).lower() == "x", body, given))
        ids = _auto_ids([given for _, _, given in entries], "letters")
        meta["choices"] = [{"id": i, "text": body} for i, (_, body, _) in zip(ids, entries)]
        correct = [i for i, (right, _, _) in zip(ids, entries) if right]
        multiple = _bool(get("multiple"), line_of("multiple"), where, errors) if "multiple" in fields else len(correct) > 1
        meta["multiple"] = multiple
        if not correct:
            errors.append({"line": section.line, "exercise": where, "message": "mark the right choice with - [x]"})
        key = {"correct": correct}
    elif kind == "order" and (section := need("order")):
        entries = [_split_id(text) for _, text in _items(section.rows, where, "order", errors)]
        shown = _shuffled(list(range(len(entries))), f"{gym_id}/{slug}")
        ids = _auto_ids([entries[i][1] for i in shown], "letters")
        id_of = dict(zip(shown, ids))
        meta["items"] = [{"id": id_of[i], "text": entries[i][0]} for i in shown]
        key = {"order": [id_of[i] for i in range(len(entries))]}
    elif kind == "match" and (section := need("match")):
        key = _match(section, meta, gym_id, slug, where, errors)
    elif kind == "fill" and (section := need("blanks")):
        key = _fill(section, meta, options, where, errors)
    elif kind == "answer" and (section := need("answer")):
        rows_ = [(n, t) for n, t in section.rows if t.strip()]
        if rows_ and all(ITEM.match(t) for _, t in rows_):
            alternatives = [text for _, text in _items(section.rows, where, "answer", errors)]
        else:
            alternatives = [_dedent(section.rows)]
        key = _answer_spec(alternatives, options, section.line, where, errors)
    elif kind == "essay" and (section := need("rubric")):
        key = {"rubric": _dedent(section.rows)}
        if "pass_score" in fields:
            try:
                key["pass_score"] = float(get("pass_score"))
            except ValueError:
                errors.append({"line": line_of("pass_score"), "exercise": where, "message": "pass_score: a number from 0 to 1"})
        if "words" in fields:
            found = re.fullmatch(r"(\d*)\s*-\s*(\d*)", get("words"))
            if not found or not any(found.groups()):
                errors.append({"line": line_of("words"), "exercise": where, "message": "words: 150-400, 150- or -400"})
            else:
                if found.group(1):
                    meta["min_words"] = int(found.group(1))
                if found.group(2):
                    meta["max_words"] = int(found.group(2))
    elif kind == "checklist" and (section := need("checklist")):
        meta["items"] = [text for _, text in _items(section.rows, where, "checklist", errors)]
    elif kind == "code":
        _code(exercise, sections, fields, where, errors)
    if key is not None:
        if explanation:
            key["explanation"] = explanation
        exercise.key = key
    if kind != "code" and not any(error.get("exercise") == where for error in errors):
        errors.extend({"line": raw.line, "exercise": where, "message": _plain(message)} for message in kinds.key_errors(kind, meta, key))
    return exercise


PLAIN = [  # kinds.py speaks of problem.json and answer.json; say it in this file's terms
    ('problem.json: "items" needs at least three {"id", "text"} entries with distinct ids', "### Order needs at least three distinct items"),
    ('problem.json: "prompts" needs at least two {"id", "text"} entries with distinct ids', "### Match needs at least two rows with a prompt"),
    ('problem.json: "options" needs at least two {"id", "text"} entries with distinct ids', "### Match needs at least two different options"),
    ('problem.json: "choices" needs at least two {"id", "text"} entries with distinct ids', "### Choices needs at least two choices"),
    ('problem.json: a [[id]] blank cannot sit inside $math$', "a [[blank]] cannot sit inside $math$"),
    ('problem.json: each [[id]] blank appears once in "text"', "two blanks share an {#id}"),
    ('problem.json: "items" lists the steps to tick off', "### Checklist lists the steps to tick off"),
    ('several correct choices need "multiple": true in problem.json', "several choices are marked [x], so drop multiple: false"),
    ('problem.json: "minutes"', "minutes:"),
    ('problem.json: "concepts" lists up to 8 short tags', "concepts: up to 8 short tags, comma-separated"),
    ('answer.json: "rubric" says what a good answer contains', "### Rubric says what a good answer contains"),
    ("answer.json: ", ""),
    ("problem.json: ", ""),
]


def _plain(message: str) -> str:
    for old, new in PLAIN:
        message = message.replace(old, new)
    return message


def _a(kind: str) -> str:
    return ("an " if kind[0] in "aeiou" else "a ") + kind


def _cells(line: str) -> list[str]:
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", body)]


def _match(section: _Block, meta: dict, gym_id: str, slug: str, where: str, errors: list[dict]) -> dict:
    """| Prompt | Its option | rows; a row with an empty prompt adds an option that matches nothing."""
    rows = [(number, line) for number, line in section.rows if line.strip()]
    if not rows or not all(line.strip().startswith("|") for _, line in rows):
        errors.append({"line": section.line, "exercise": where, "message": "### Match holds a table: | prompt | option |, one row per pair"})
        return {}
    table = [(number, _cells(line)) for number, line in rows]
    if len(table) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in table[1][1]):
        table = table[2:]  # header and separator
    prompts, options, pairs = [], [], []
    index_of: dict[str, int] = {}  # option text -> its index in options
    for number, cells in table:
        if len(cells) != 2:
            errors.append({"line": number, "exercise": where, "message": "each row has two cells: | prompt | option |"})
            continue
        prompt, option = cells
        if not option:
            errors.append({"line": number, "exercise": where, "message": "a row needs an option (leave the prompt empty for an option that matches nothing)"})
            continue
        option_text, option_id = _split_id(option)
        known = index_of.get(option_text)
        if known is None:
            options.append((option_text, option_id))
            known = index_of[option_text] = len(options) - 1
        if prompt:
            prompt_text, prompt_id = _split_id(prompt)
            prompts.append((prompt_text, prompt_id))
            pairs.append(known)
    prompt_ids = _auto_ids([given for _, given in prompts], "p")
    shown = _shuffled(list(range(len(options))), f"{gym_id}/{slug}/options")
    option_ids = _auto_ids([options[i][1] for i in shown], "letters")
    id_of = dict(zip(shown, option_ids))
    meta["prompts"] = [{"id": i, "text": text} for i, (text, _) in zip(prompt_ids, prompts)]
    meta["options"] = [{"id": id_of[i], "text": options[i][0]} for i in shown]
    return {"pairs": {prompt_id: id_of[option] for prompt_id, option in zip(prompt_ids, pairs)}}


def _fill(section: _Block, meta: dict, options: dict, where: str, errors: list[dict]) -> dict:
    """Text with [[answer]] blanks, or a single code block with them (shown as code)."""
    body = _text(section.rows).strip()
    fences = _fences(section.rows)
    outside = [line for _, line in section.rows if line.strip()]
    if len(fences) == 1 and len(outside) >= 2 and FENCE.match(outside[0]) and FENCE.match(outside[-1]):
        body, meta["code"] = fences[0][2], True
    explicit, specs, order = [], [], []
    for found in BLANK.finditer(body):
        content, given = _split_id(found.group(1))
        explicit.append(given)
        alternatives = [part.strip() for part in content.split("|")]
        specs.append(_answer_spec(alternatives, options, section.line, where, errors))
        order.append(found.span())
    ids = _auto_ids(explicit, "b")
    pieces, last = [], 0
    for (start, end), blank in zip(order, ids):
        pieces.append(body[last:start] + f"[[{blank}]]")
        last = end
    meta["text"] = "".join(pieces) + body[last:]
    if not ids:
        errors.append({"line": section.line, "exercise": where, "message": "### Blanks needs at least one [[answer]]"})
    return {"blanks": dict(zip(ids, specs))}


def _safe(path: str) -> bool:
    return bool(SAFE_PATH.match(path)) and ".." not in path.split("/") and not path.startswith(".")


def _code(exercise: Exercise, sections: dict, fields: dict, where: str, errors: list[dict]) -> None:
    files = exercise.files
    starters: dict[str, str] = {}
    for name, folder in (("starter", "starter"), ("solution", "solution")):
        for number, info, content in _fences(sections[name].rows) if name in sections else []:
            attribute = FILE_ATTR.search(info)
            if attribute:
                path = next(group for group in attribute.groups() if group)
                if not _safe(path):
                    errors.append({"line": number, "exercise": where, "message": f"file={path}: a relative path of letters, digits, _ . - and /"})
                    continue
                files[f"{folder}/{path}"] = content.rstrip() + "\n"
                continue
            word = (info.split() or [""])[0].lower()
            lang = LANG_ALIASES.get(word, word)
            if lang not in LANGS:
                errors.append({"line": number, "exercise": where, "message": f"say the language after the fence: ```python, ```cpp, ... (or file=<name>)"})
                continue
            files[f"{folder}/{{{lang}}}"] = content.rstrip() + "\n"  # resolved to the language's file in write()
            if name == "starter":
                starters[lang] = content
    for number, info, content in _fences(sections["files"].rows) if "files" in sections else []:
        attribute = FILE_ATTR.search(info)
        path = next((group for group in attribute.groups() if group), "") if attribute else ""
        if not path:
            errors.append({"line": number, "exercise": where, "message": "each block in ### Files says where it goes: ```python file=problem.py"})
        elif not _safe(path) or path in GENERATED:
            errors.append({"line": number, "exercise": where, "message": f"file={path}: not a path this exercise may write"})
        else:
            files[path] = content.rstrip() + "\n"
    if "starter" not in sections:
        errors.append({"line": exercise.line, "exercise": where, "message": "a code exercise needs ### Starter"})
    if "tests" not in sections:
        if "problem.py" not in files:
            errors.append({"line": exercise.line, "exercise": where, "message": "add ### Tests (solve(...) == expected), or a grader in ### Files (file=problem.py)"})
        return
    if set(starters) - {"python"} or "python" not in starters:
        errors.append({"line": sections["tests"].line, "exercise": where, "message": "### Tests grade Python: give a ```python starter"})
        return
    if "problem.py" in files or "grader.json" in files:
        errors.append({"line": sections["tests"].line, "exercise": where, "message": "### Tests generate problem.py; drop the one in ### Files"})
    params = _list(fields["params"][0]) if "params" in fields else _params_of(starters["python"])
    if not params:
        errors.append({"line": sections["starter"].line, "exercise": where, "message": "the starter defines def solve(...) with its inputs (or give params:)"})
        return
    cases = _cases(sections["tests"].rows, params, where, errors)
    if not cases:
        errors.append({"line": sections["tests"].line, "exercise": where, "message": "### Tests needs at least one solve(...) line"})
    solution = files.get("solution/{python}", "")
    if any("expected" not in case for case in cases) and "def solve" not in solution:
        errors.append({"line": sections["tests"].line, "exercise": where, "message": "a test without == expected needs a ### Solution to compute it"})
    exercise.draft = {"slug": exercise.slug, "title": exercise.meta["title"], "difficulty": exercise.meta["difficulty"],
                      "summary": exercise.meta["summary"], "statement": exercise.statement, "params": params,
                      "starter": starters["python"], "solution": solution, "cases": cases}
    files.pop("starter/{python}", None)
    files.pop("solution/{python}", None)


# ---- writing -------------------------------------------------------------------------------------


def write(gym: Gym, root: Path | str) -> packages.Package:
    """The package under `root`, which must be empty or missing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "gym.json").write_text(json.dumps(gym.manifest, indent=2, ensure_ascii=False) + "\n")
    lang_files = {lang["id"]: lang["file"] for lang in gym.manifest.get("langs", [])}
    for exercise in gym.exercises:
        directory = root / "problems" / exercise.slug
        if exercise.draft:
            gymdraft.write_exercise(directory, exercise.draft)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {key: value for key, value in exercise.meta.items() if not key.startswith("_")}
        (directory / "problem.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        (directory / "PROBLEM.md").write_text(exercise.statement)
        if exercise.hints:
            (directory / "HINTS.md").write_text("".join(f"## Hint {n}\n\n{hint}\n\n" for n, hint in enumerate(exercise.hints, 1)).rstrip() + "\n")
        if exercise.editorial:
            (directory / "EDITORIAL.md").write_text(exercise.editorial + "\n")
        if exercise.key is not None:
            (directory / "answer.json").write_text(json.dumps(exercise.key, indent=2, ensure_ascii=False) + "\n")
        for path, content in exercise.files.items():
            if "{" in path:  # starter/{python} -> starter/solution.py
                folder, lang = path.split("/{")
                path = f"{folder}/{lang_files.get(lang.rstrip('}'), lang.rstrip('}'))}"
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    return packages.load(root)


def _attribute(gym: Gym, message: str) -> dict:
    """A package validator message, pointed at the exercise it is about."""
    for slug, line in gym.lines.items():
        if re.search(rf"(^|/){re.escape(slug)}(/|:)", message):
            return {"line": line, "exercise": slug, "message": message.split(": ", 1)[-1]}
    return {"line": None, "exercise": None, "message": message}


def check(text: str) -> tuple[Gym | None, list[dict]]:
    """Compile and validate like an upload. No code runs: graders are checked by `gym validate` or a machine."""
    try:
        gym = compile_text(text)
    except GymMarkdownError as exc:
        return None, exc.errors
    with tempfile.TemporaryDirectory(prefix="gymmd-") as tmp:
        package = write(gym, Path(tmp) / (gym.manifest["id"] or "gym"))
        return gym, [_attribute(gym, message) for message in packages.validate_upload(package)]


def build(text: str, root: Path | str) -> packages.Package:
    """Compile, write and validate; raises GymMarkdownError with every problem found."""
    gym = compile_text(text)
    package = write(gym, root)
    errors = [_attribute(gym, message) for message in packages.validate_upload(package)]
    if errors:
        raise GymMarkdownError(errors)
    return package


def is_gym_markdown(data: bytes) -> bool:
    return not data.startswith(b"\x1f\x8b") and data.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"---")


def bundle(data: bytes) -> bytes:
    """A .gym.md upload as a .tar.gz bundle; raises GymMarkdownError."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GymMarkdownError([{"line": None, "exercise": None, "message": f"not UTF-8 text: {exc}"}]) from exc
    with tempfile.TemporaryDirectory(prefix="gymmd-") as tmp:
        package = build(text, Path(tmp) / "pkg")
        return packages.pack(package.root)


# ---- exporting -----------------------------------------------------------------------------------


class ExportError(ValueError):
    pass


def _fence_for(content: str) -> str:
    longest = max((len(run) for run in re.findall(r"`{3,}", content)), default=2)
    return "`" * max(3, longest + 1)


def _block(info: str, content: str) -> str:
    fence = _fence_for(content)
    return f"{fence}{info}\n{content.rstrip()}\n{fence}"


def _statement_body(statement: str) -> str:
    """PROBLEM.md without its title, its headings moved below ###, which an exercise keeps for its sections."""
    lines = statement.rstrip().split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    out, fence = [], None
    for line in lines:
        opened = FENCE.match(line)
        if fence:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence) and not opened.group(2).strip():
                fence = None
        elif opened:
            fence = opened.group(1)
        else:
            heading = HEADING.match(line)
            if heading:  # # and ## become ####, ### becomes #####: only sections use ### in an exercise
                line = ("#####" if len(heading.group(1)) == 3 else "####") + " " + heading.group(2).strip()
        out.append(line)
    return "\n".join(out).strip()


def _item(text: str, prefix: str = "- ") -> str:
    """A list item; continuation lines are indented to the text after the marker (`- ` or `1. `), as _items reads them."""
    lines = text.strip().split("\n")
    pad = " " * len(prefix.split(" ")[0] + " ")
    return prefix + lines[0] + "".join("\n" + (pad + line if line.strip() else "") for line in lines[1:])


def _answers_text(spec: dict) -> list[str]:
    out = []
    for index, answer in enumerate(spec.get("answers", [])):
        text = str(answer)
        if spec.get("match") == "number" and index == 0:
            if spec.get("tolerance"):
                text += f" ± {spec['tolerance']}"
            if spec.get("rel_tolerance"):
                text += f" ± {_number(spec['rel_tolerance'] * 100)}%"
        out.append(text)
    return out


def _needs_text_match(spec: dict) -> bool:
    return spec.get("match", "text") == "text" and all(NUMBER.fullmatch(str(a)) for a in spec.get("answers", []))


def export(package: packages.Package) -> str:
    """The package as Gym Markdown. compile_text(export(p)) grades like p (item ids aside)."""
    manifest = package.manifest
    out = ["---", f"id: {package.id}", f"title: {manifest.get('title', '')}"]
    if manifest.get("runtime", "gymkit") != "gymkit":
        out.append(f"runtime: {manifest['runtime']}")
    if bool((manifest.get("requires") or {}).get("gpu")) != (manifest.get("runtime") == "gpu_gym"):
        out.append(f"gpu: {str(bool(manifest['requires']['gpu'])).lower()}")
    langs = [lang["id"] for lang in manifest.get("langs", [])]
    if langs and langs != ["python"]:
        out.append(f"langs: {', '.join(langs)}")
    for key in ("order", "publish"):
        if key in manifest:
            out.append(f"{key}: {manifest[key]}")
    out += ["---", "", manifest.get("description", "").strip(), ""]
    if manifest.get("sources"):
        out += [_source_line(item) for item in manifest["sources"]] + [""]
    warmup = set(manifest.get("warmup", []))
    tracks = manifest.get("tracks") or []
    placed = {slug for track in tracks for slug in track.get("exercises", [])}
    groups = [(track.get("name", ""), track.get("summary", ""), track.get("exercises", [])) for track in tracks]
    rest = [slug for slug in package.slugs() if slug not in placed]
    if rest:
        groups.append(("Your exercises", "", rest))
    for name, summary, slugs in groups:
        out += [f"# {name}", ""] + ([summary, ""] if summary else [])
        for slug in slugs:
            out += [_export_exercise(package, slug, slug in warmup), ""]
    return "\n".join(out).rstrip() + "\n"


def _export_exercise(package: packages.Package, slug: str, warmup: bool) -> str:
    meta = package.meta(slug)
    key = package.answer_key(slug) or {}
    kind = meta.get("kind", "code")
    directory = package.problem_dir(slug)
    head = [f"## {meta['title']}", f"id: {slug}"]
    if not (kind == "lesson" and meta.get("difficulty") == "easy"):  # lessons default to easy
        head.append(f"difficulty: {meta.get('difficulty', '')}")
    if meta.get("summary"):
        head.append(f"summary: {' '.join(str(meta['summary']).split())}")
    if meta.get("minutes"):
        head.append(f"minutes: {meta['minutes']}")
    if meta.get("concepts"):
        head.append(f"concepts: {', '.join(meta['concepts'])}")
    if warmup:
        head.append("warmup: true")
    if meta.get("placeholder"):
        head.append(f"placeholder: {meta['placeholder']}")
    if kind == "essay" and (meta.get("min_words") or meta.get("max_words")):
        head.append(f"words: {meta.get('min_words') or ''}-{meta.get('max_words') or ''}")
    if kind == "essay" and "pass_score" in key:
        head.append(f"pass_score: {key['pass_score']}")
    parts = ["\n".join(head), _statement_body(package.statement(slug))]
    sections: list[tuple[str, str]] = []

    if kind == "choice":
        correct = set(key.get("correct", []))
        auto = _auto_ids([None] * len(meta.get("choices", [])), "letters")
        lines = []
        for index, choice in enumerate(meta.get("choices", [])):
            mark = "x" if choice["id"] in correct else " "
            suffix = f" {{#{choice['id']}}}" if choice["id"] != auto[index] else ""
            lines.append(_item(str(choice["text"]) + suffix, f"- [{mark}] "))
        if meta.get("multiple") and len(correct) < 2:
            head.append("multiple: true")
            parts[0] = "\n".join(head)
        elif not meta.get("multiple") and len(correct) > 1:
            head.append("multiple: false")
            parts[0] = "\n".join(head)
        sections.append(("Choices", "\n".join(lines)))
    elif kind == "order":
        text = {item["id"]: item["text"] for item in meta.get("items", [])}
        sections.append(("Order", "\n".join(_item(text[i], f"{n}. ") for n, i in enumerate(key.get("order", []), 1))))
    elif kind == "match":
        options = {item["id"]: item["text"] for item in meta.get("options", [])}
        cell = lambda text: str(text).replace("|", "\\|").replace("\n", " ")  # noqa: E731
        rows = ["| Prompt | Match |", "|---|---|"]
        used = set()
        for prompt in meta.get("prompts", []):
            option = key.get("pairs", {}).get(prompt["id"])
            used.add(option)
            rows.append(f"| {cell(prompt['text'])} | {cell(options.get(option, ''))} |")
        rows += [f"|  | {cell(text)} |" for option, text in options.items() if option not in used]
        sections.append(("Match", "\n".join(rows)))
    elif kind == "fill":
        blanks = key.get("blanks", {})
        if any(_needs_text_match(spec) for spec in blanks.values()):
            raise ExportError(f"{slug}: a blank with numbers that match as text cannot be written yet")
        if blanks and any(spec.get("case_sensitive") for spec in blanks.values()):
            head.append("case_sensitive: true")
            parts[0] = "\n".join(head)
        text = kinds.BLANK.sub(lambda found: "[[" + " | ".join(_answers_text(blanks.get(found.group(1), {}))) + "]]", meta.get("text", ""))
        sections.append(("Blanks", _block("", text) if meta.get("code") else text))
    elif kind == "answer":
        if _needs_text_match(key):
            head.append("match: text")
        if key.get("case_sensitive"):
            head.append("case_sensitive: true")
        parts[0] = "\n".join(head)
        sections.append(("Answer", "\n".join(_item(answer) for answer in _answers_text(key))))
    elif kind == "essay":
        sections.append(("Rubric", str(key.get("rubric", "")).strip()))
    elif kind == "checklist":
        sections.append(("Checklist", "\n".join(_item(item) for item in meta.get("items", []))))
    elif kind == "code":
        sections += _export_code(package, slug, directory)

    if package.hints(slug):
        sections.insert(0, ("Hints", "\n".join(_item(hint) for hint in package.hints(slug))))
    if key.get("explanation") and kind in {"choice", "order", "match", "fill", "answer"}:
        sections.append(("Explanation", str(key["explanation"]).strip()))
    if kind == "code" and package.editorial(slug):
        sections.append(("Editorial", package.editorial(slug)))
    parts += [f"### {name}\n\n{body}" for name, body in sections]
    return "\n\n".join(part for part in parts if part)


def _read_text(path: Path, slug: str) -> str:
    try:
        return path.read_text()
    except UnicodeDecodeError:
        raise ExportError(f"{slug}: {path.name} is not text; Gym Markdown holds text files only") from None


def _export_code(package: packages.Package, slug: str, directory: Path) -> list[tuple[str, str]]:
    langs = {lang["file"]: lang["id"] for lang in package.langs}
    grader = directory / "grader.json"
    sections: list[tuple[str, str]] = []
    if grader.is_file() and (directory / "problem.py").read_text() == gymdraft.GRADER:
        spec = json.loads(grader.read_text())
        starter = _read_text(directory / "starter" / "solution.py", slug).replace("# GYM_STARTER\n", "", 1)
        sections.append(("Starter", _block("python", starter)))
        solution = directory / "solution" / "solution.py"
        if solution.is_file():
            sections.append(("Solution", _block("python", _read_text(solution, slug))))
        lines = []
        for case in spec["cases"]:
            call = "solve(" + ", ".join(repr(case["args"][name]) for name in spec["params"]) + ")"
            if "expected" in case:
                call += f" == {repr(case['expected'])}"
            if case.get("name"):
                call += f"  # {case['name']}"
            lines.append(call)
        sections.append(("Tests", _block("python", "\n".join(lines))))
        if _params_of(starter) != spec["params"]:
            raise ExportError(f"{slug}: the starter's solve() does not take {spec['params']}")
        return sections
    starters, solutions, files = [], [], []
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        relative = path.relative_to(directory).as_posix()
        if relative in GENERATED:
            continue
        top, _, name = relative.partition("/")
        content = _read_text(path, slug)
        if top in {"starter", "solution"} and name in langs:
            (starters if top == "starter" else solutions).append(_block(langs[name], content))
        elif top in {"starter", "solution"}:
            (starters if top == "starter" else solutions).append(_block(f" file={name}", content))
        else:
            files.append(_block(f"{_fence_lang(relative)} file={relative}", content))
    sections.append(("Starter", "\n\n".join(starters)))
    if solutions:
        sections.append(("Solution", "\n\n".join(solutions)))
    if files:
        sections.append(("Files", "\n\n".join(files)))
    return sections


def _fence_lang(path: str) -> str:
    return {".py": "python", ".cpp": "cpp", ".cu": "cuda", ".h": "cpp", ".hpp": "cpp", ".json": "json", ".md": "markdown"}.get(Path(path).suffix, "")


def guide() -> str:
    return GUIDE.read_text()


TEMPLATE = '''---
id: {id}
title: {title}
---

One paragraph about the gym: who it is for and what they will practise.

# First steps
One sentence about this track.

## How this gym works
minutes: 1

A lesson is a page to read. It has no sections; learners mark it as read.

## Pick the capital
id: capital
difficulty: easy
concepts: geography

Which city is the capital of France?

### Choices
- [ ] Lyon
- [x] Paris
- [ ] Marseille

### Explanation
Paris has been the capital since 987.

## Add two numbers
id: add_two
difficulty: easy
minutes: 5

Write `solve(a, b)` that returns `a + b`.

### Hints
- Python's `+` works on integers of any size.

### Starter
```python
def solve(a, b):
    pass
```

### Solution
```python
def solve(a, b):
    return a + b
```

### Tests
```python
solve(1, 2) == 3  # example
solve(-5, 3) == -2  # negatives
solve(1000000000000, 1)  # no == : the solution computes the expected value
```
'''
