# Contributing to Human Gym

Thanks for helping. Human Gym takes three kinds of contributions: new gym content (lessons, quizzes and exercises), fixes and features in the code, and reports of bugs or gaps. All of them start on GitHub.

## Before you start

- **Bugs and ideas:** open an issue with the template. Say what is wrong or missing today, and name the files, pages or commands involved.
- **Larger changes** (a new package, a change to the job or store format, anything touching how other people's code is run): open an issue first so the design can be agreed before you write it.
- **Security problems:** do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Set up

```bash
git clone https://github.com/ykirpichev/gymhub
cd gymhub
./setup.sh && source .venv/bin/activate
```

Or install by hand (CPU-only PyTorch is enough for everything except CUDA exercises):

```bash
pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
pip install -r deploy/requirements-board.txt
pip install -e ".[hub]" pytest pytest-xdist httpx "psycopg[binary]"
```

Python 3.10 or newer is required. CI runs Python 3.12.

## Run the tests

```bash
python -m pytest -q -n auto        # everything, in parallel, as CI runs it
python -m pytest -q tests/test_pool.py   # one module
python -m gymhub.packages validate       # every gym package in gyms/
```

Tests that need CUDA skip themselves. `tests/test_ui.py` opens the board in headless Chromium and needs Node and Playwright. The pool's Postgres tests run when `POOL_TEST_POSTGRES` points at a database (see `CLAUDE.md`).

Tests must be safe to run in parallel: use `tmp_path` and free ports, never fixed paths or ports.

To try the board locally, without any cloud account:

```bash
GYMHUB_STORE=sqlite:.cache/gymhub/dev.db GYMHUB_AUTH=dev uvicorn gymhub.board.app:app --port 8080
python -m gymhub.runner &
```

`GYMHUB_AUTH=dev` turns off sign-in and makes every request an admin. Use it only on localhost.

## Adding a gym or exercises

The easiest way to write a gym is one Gym Markdown file (`.gym.md`). The format is described in [`gymhub/GYM_MARKDOWN.md`](gymhub/GYM_MARKDOWN.md).

```bash
python -m gymhub.cli gym new my-gym.gym.md          # start from a template
python -m gymhub.cli gym check my-gym.gym.md        # compile it; errors come with line numbers
python -m gymhub.cli gym validate my-gym.gym.md     # also run the reference solutions against the tests
```

Gyms live in their own repository, [gymhub-gyms](https://github.com/ykirpichev/gymhub-gyms), as packages under `<id>/`; send changes to gyms there. Here they appear under `gyms/<id>/`, fetched at the commit `gyms.lock` pins (`python -m gymhub.gymsfetch`), or linked to your clone of the gyms repository. Each exercise has public parts (`problem.json`, `PROBLEM.md`, `HINTS.md`, `starter/`) and private parts (`solution/`, `problem.py`). Learners and the board only ever see the public parts, so keep answers out of them. `python -m gymhub.packages validate gyms/<id>` must pass before a gym is merged.

Do not commit your own solutions or `progress.json`: `gyms/*/solutions/` is for your local attempts only.

If you work with Claude Code, the `write-gym` skill in `.claude/skills/` walks through the same steps.

## Code layout

The repository holds four packages. `CLAUDE.md` describes each in more detail.

| Package | What it is |
|---|---|
| `gpu_gym/` | GPU kernel exercises (CUDA, PyTorch, Triton) and the `gpu-gym` CLI |
| `gymkit/` | Python and C++ grader runtime shared by the other gyms |
| `gymhub/` | The web board, the runner daemon and the gym package formats |
| `pool/` | The compute pool: a server and machine agents that run jobs in containers |

Two rules hold the design together, and changes that break them will not be merged:

- **The board never executes user code.** It only writes job documents to the store. Runners and pool machines do the grading.
- **`pool/` never imports from `gymhub/`.** gymhub may depend on the pool, never the reverse.

To change the pool's database schema, append a migration in `pool/migrations.py`. Never edit one that has shipped.

## Pull requests

- Keep a pull request to one change. Link its issue (`Closes #N`, or `Part of #N`).
- Fill in the template: what changed and why, and how you checked it (which tests, what you tried by hand and where, and what you did not check).
- Add or update tests with the code. CI runs the full suite and the gym validation on every pull request, and both must pass.
- Update `CHANGELOG.md` under *Unreleased* when the change is something a user or self-hoster would notice.
- Update the README, `docs/` or `CLAUDE.md` when you change a command, an environment variable or a default.

## Licence

Human Gym is released under the [MIT License](LICENSE). By opening a pull request you agree that your contribution is released under the same licence.
