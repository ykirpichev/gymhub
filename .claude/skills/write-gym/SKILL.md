---
name: write-gym
description: Write or change a human-gym gym (a course of lessons, questions and coding exercises) as one Gym Markdown file (.gym.md), check it, run its tests and publish it. Use when asked to create a gym, course, quiz, exercises or practice problems for the board, or to edit an existing gym's content.
---

# Writing a gym

A gym is one Markdown file, `<id>.gym.md`, compiled into a package by `gymhub/gymmd.py`. Read the language reference
first: `python -m gymhub.cli gym guide` (the same text as `gymhub/GYM_MARKDOWN.md`).

## Steps

1. **Start.** For a new gym, run `python -m gymhub.cli gym new <id>.gym.md`. To change an existing gym, export it with
   `python -m gymhub.cli gym export gyms/<id> -o <id>.gym.md` and edit the file. `examples/every-kind.gym.md` has one
   exercise of every kind.
2. **Plan the gym before writing it.** Write down who it is for. Group exercises into tracks of 4–10 that build on
   each other, each with a one-sentence summary. Open a track with a short lesson when the learner needs ideas first.
   Then go from recall (choice, answer, match, order, fill) to application (code, essay).
3. **Write.** Follow these rules:
   - Keep each statement self-contained: give every number, definition and input format the learner needs.
   - Every `choice`, `answer`, `order`, `match` and `fill` gets an `### Explanation` that teaches, not just the answer.
   - Make wrong choices plausible, and use mistakes people really make.
   - For typed numbers, give a tolerance (`0.25 ± 0.005`), and accept equivalent spellings of text answers.
   - In code exercises, write 4–10 tests: the example from the statement, edge cases (empty, one item, duplicates,
     negatives) and one larger input. Always write a `### Solution`.
   - Set `minutes` and `concepts` on each exercise, and choose ids you will keep.
4. **Check.** Run `python -m gymhub.cli gym check <file> --json` and fix every error at its `line`. Repeat until
   `"ok": true`.
5. **Run the tests.** Run `python -m gymhub.cli gym validate <file>`. Every starter must fail its
   tests and every solution must pass them. If a solution fails, fix the solution or the expected value. Don't
   weaken the test.
6. **Publish.** Only do this when asked. Run `python -m gymhub.cli gym push <file> --wait` with `GYMHUB_BOARD` and
   `GYMHUB_TOKEN` set (an API token from the board's Gyms page). Boards behind IAP don't accept CLI uploads; tell the
   user to upload the file on the Gyms page instead. New gyms start private.

## Don't

- Don't write answers into the statement or hints, and don't list `### Order` items in any order but the right one.
  The compiler shuffles them.
- Don't use `#` or `##` inside a statement, and don't put a `[[blank]]` inside `$math$`.
- Don't use names or expressions in tests (`list(range(10))`); write the literal values. For generated or very large
  inputs, write a grader in `### Files` (`file=problem.py`, see `gymkit/spec.py`).
- Don't change an existing exercise's `id`, or the `{#id}` of its items, unless asked. Learners' progress refers to
  them.
