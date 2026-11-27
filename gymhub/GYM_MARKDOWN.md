# Gym Markdown

A whole gym in one Markdown file, `<id>.gym.md`. People and agents write it, and a compiler turns it into a gym package:
the public files learners see, the private answer keys, and the grader for Python exercises.

```
python -m gymhub.cli gym new my-gym.gym.md          start from a template
python -m gymhub.cli gym check my-gym.gym.md        compile and check: file:line: exercise: message  (--json for tools)
python -m gymhub.cli gym validate my-gym.gym.md     also run every Python test on this computer
python -m gymhub.cli gym push my-gym.gym.md --wait  upload a new version (or upload the file on the board's Gyms page)
python -m gymhub.cli gym export gyms/<id>           any existing gym as Gym Markdown: a good place to learn from
```

## The shape of a file

```markdown
---
id: sql-basics
title: SQL basics
---

One paragraph that describes the gym: who it is for and what they practise.

# Reading queries
One sentence about this track.

## What a JOIN returns
id: join_rows
difficulty: easy
minutes: 3
concepts: joins

The statement, in Markdown.

### Choices
- [x] The right answer
- [ ] A wrong one
```

- **Front matter** (between `---` lines) holds `key: value` lines: `id` (3-40 characters: lowercase letters, digits and `-`)
  and `title` are required. Optional: `description` (otherwise the paragraph after the front matter), `langs` (for code:
  `python`, `cpp`, `cuda`, `pytorch`, `triton`; default `python`), `runtime` (`gymkit`, or `gpu_gym` for CUDA gyms),
  `gpu: true` when exercises need an NVIDIA GPU, and `order`.
- **Sources:** lines `- [Title](https://...): note` between the front matter and the first heading list what the gym
  is based on and further reading. The board shows them on the gym's page. The note is optional.
- **`# Track`** starts a track. The paragraph under it is its one-sentence summary (240 characters at most).
- **`## Exercise title`** starts an exercise. Its fields come right under the heading, one `key: value` per line, then a
  blank line and the statement.
- **`### Section`** headings with the names below hold the parts of an exercise. In an exercise, `###` is for sections
  only: any other `###` heading is an error (a misspelt `### Explanations` would otherwise publish the explanation with
  the statement). Headings inside a statement use `####`; a lesson, which has no sections, may also use `###`. The
  statement may not use `#` or `##`.
- Every code block (```` ``` ```` or `~~~`) must be closed; an open one is an error at the line that opens it.
- Exercises appear in the order they are written. A file is at most 2 MB.

### Exercise fields

| field | meaning |
|---|---|
| `id` | lowercase letters, digits and `_`, starting with a letter. Default: made from the title. Keep it stable: progress is stored under it. |
| `difficulty` | `easy`, `medium` or `hard`. Required, except for lessons. |
| `summary` | one line, shown in lists |
| `minutes` | rough time to solve, 1 to 240 |
| `concepts` | up to 8 short tags, comma-separated |
| `kind` | rarely needed: the sections decide the kind (below) |
| `warmup: true` | a worked example: listed, not counted |

Text everywhere is Markdown with math (`$…$`, `$$…$$`) and callouts (`> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`).

## Kinds

The sections an exercise has decide its kind.

| kind | sections | learners… |
|---|---|---|
| lesson | none | read it and mark it read; not counted in "solved" |
| choice | `### Choices` | pick one answer, or several |
| answer | `### Answer` | type a word, phrase or number |
| order | `### Order` | put items in order |
| match | `### Match` | pair each prompt with an option |
| fill | `### Blanks` | fill blanks in a sentence or in code |
| essay | `### Rubric` | write a paragraph, graded against the rubric |
| checklist | `### Checklist` | tick off steps; self-reported |
| code | `### Starter` and `### Tests` (or a grader in `### Files`) | write code a machine runs |

Any kind may add `### Hints` (a list; each item is revealed one at a time). Choice, answer, order, match and fill may add
`### Explanation`, shown once the answer is right and never before.
A code exercise may add `### Editorial`: the worked solution, with the approach, its complexity and the usual
pitfalls. Learners see it after their first submission, or earlier if they choose to open it. Use `####` headings
inside it (`#` to `###` start a new gym part).

Statements, lessons, hints and editorials can hold diagrams. A ```` ```mermaid ```` block is drawn with
[Mermaid](https://mermaid.js.org) (flowcharts, sequence diagrams). Inline `<svg>` is shown as written; keep it on
consecutive lines (a blank line ends it) and give shapes the classes below so they follow the light and dark themes:
`d-box` (outlined box), `d-fill` (soft fill), `d-a`, `d-b`, `d-c` (three accent fills), `d-line` (line or arrow;
add `marker-end="url(#d-arrow)"` for an arrowhead), `d-text`, `d-muted` (text). Wrap it in
`<figure class="diagram">` and add a `<figcaption>`.

A list item (a hint, choice, step...) runs over several lines when the lines after it are indented to its text: 2
spaces under `- `, 3 under `1. `. Indent further for an indented code block or a nested list inside the item.

### choice

```markdown
### Choices
- [ ] O(1)
- [x] O(log n)
- [ ] O(n)
```

Mark every right choice `[x]`. With several, learners pick all that apply. `multiple: true` asks for "all that apply" with
a single right choice.

### answer

```markdown
### Answer
- 0.25 ± 0.01
```

Each list item is an accepted answer. Numbers match as numbers: `42` exactly, `0.25 ± 0.01` within an absolute
tolerance, `3.2 ± 1%` within a relative one. Text matches ignoring case and extra spaces; list each spelling you accept
(`- Paris`, `- Paris, France`). Fields: `placeholder: e.g. 0.25`, `case_sensitive: true`, `match: text` (numbers that
must match as text).

### order

```markdown
### Order
1. Resolve the host name with DNS
2. Open a TCP connection
3. Send the HTTP request
```

Write the items in the right order, at least three. The compiler shuffles them for learners.

### match

```markdown
### Match
| Operation | Cost |
|---|---|
| Hash table lookup | $O(1)$ on average |
| Binary search | $O(\log n)$ |
|  | $O(n^2)$ |
```

One row per prompt, with its option. A row with an empty prompt adds an option that matches nothing. Several prompts may
share an option. Write `\|` for a `|` inside a cell. The compiler shuffles the options.

### fill

```markdown
### Blanks
After 2, 4 and 9 the mean is [[5 ± 0.001]]; the next update divides by [[n + 1 | n+1]].
```

Each `[[…]]` is a blank holding its accepted answers, separated by `|`, written like `### Answer`. A blank cannot sit
inside `$math$`: put it in the sentence around the formula. For blanks in code, make the section one fenced code block:

````markdown
### Blanks
```python
total = [[0]]
for x in xs:
    total [[+=]] x
```
````

### essay

```markdown
## Why caches help
difficulty: medium
words: 30-250
pass_score: 0.75

In a short paragraph, explain why a CPU cache makes programs faster.

### Rubric
1. Says main memory is much slower than the cache.
2. Mentions locality.
```

Claude grades against the rubric when the board has an API key; otherwise learners compare their answer with it and
mark it themselves. `words: 150-400` (or `150-`, `-400`) bounds the length; `pass_score` is 0 to 1 (default 0.7).

### checklist

```markdown
### Checklist
- Read the hints for an exercise you solved
- Benchmark a solution and improve it once
```

### lesson

No sections: the statement is the reading. `difficulty` is optional.

### code (Python, with tests)

````markdown
## Two sum
id: two_sum
difficulty: easy

Return the indices of the two numbers in `nums` that add up to `target`.

### Starter
```python
def solve(nums, target):
    pass
```

### Solution
```python
def solve(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```

### Tests
```python
solve([2, 7, 11, 15], 9) == [0, 1]  # example
solve([3, 3], 6) == [0, 1]          # duplicates
solve([1, 5, 9, 14], 23)            # no == : the solution computes the expected value
```
````

- The starter defines `solve(...)`. Its parameters are the inputs of every test (or set `params: nums, target`).
- One test per line: `solve(args) == expected  # name`. Arguments are literals: numbers, strings, lists, dicts, tuples,
  `True`, `False`, `None`; no names or expressions such as `list(range(10))`.
  A long call may run over several lines (up to 200 lines or 64 KB per test). Without `== expected`, the reference solution gives the expected value, so
  that test needs a `### Solution`.
- Learners see the statement, hints and starter, never the tests or the solution. A machine checks before publishing
  that the starter fails and the solution passes. Results compare with a small tolerance for floats.

### code (any grader)

For C++, CUDA, custom checks or data files, write the grader as files:

````markdown
### Starter
```python
def solve(nums): ...
```
```cpp
long long solve(const std::vector<int>& nums) { return 0; }
```

### Solution
```python
def solve(nums): return sum(nums)
```

### Files
```python file=problem.py
from gymkit.spec import Problem
...
```
```cpp file=cpp/harness.cpp
...
```
````

A fenced block in `### Starter` or `### Solution` goes to that language's file (the fence names the language; the gym's
`langs:` must include it). `file=<path>` in a fence's info string puts it at that path instead: under `starter/` or
`solution/` in those sections, under the exercise's directory in `### Files`. `problem.py` defines the grader; see
`gymkit/spec.py` and `python -m gymhub.cli gym export gyms/ml-coding` for examples.

## Ids

Choices, items, prompts, options and blanks get ids automatically. To keep an id stable across edits (saved answers refer
to it), end the text with `{#id}`: `- [x] Paris {#paris}`, `[[5 ± 0.1 {#mean}]]`.

## Checking

`gym check` reports every problem it finds as `file:line: exercise: message`; `--json` gives
`{"ok", "gym", "exercises": [{"id", "kind", "line"}], "errors": [{"line", "exercise", "message"}]}`. Nothing runs during
a check. `gym validate` also runs each code exercise's tests on this computer, and the board runs the same checks on a
machine before a version goes live.
