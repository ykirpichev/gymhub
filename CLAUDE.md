# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
./setup.sh && source .venv/bin/activate
gpu-gym test playground --lang all   # verify the toolchain
```

### Install (development)
```bash
pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
pip install -r deploy/requirements-board.txt
pip install -e ".[hub]" pytest pytest-xdist httpx "psycopg[binary]"
```

### Tests
```bash
python -m pytest -q -n auto                      # all tests, one worker per core (pytest-xdist; as CI runs them)
python -m pytest -q                              # all tests, serially
python -m pytest -q tests/test_pool.py           # single module
python -m pytest -q tests/test_ui.py              # board pages in headless Chromium (needs node + playwright)
POOL_TEST_POSTGRES=postgresql://postgres:pool@localhost:5432/postgres python -m pytest -q -n auto tests/test_pool.py tests/test_pool_migrations.py
```

CI uses CPU-only PyTorch and SQLite (no CUDA, no Firestore). Tests that need CUDA skip themselves. Tests must be safe to run in parallel (`-n auto`): use `tmp_path` and free ports, never fixed paths or ports. The pool's Postgres tests are a separate CI job.

### Gym packages
```bash
python -m gymhub.packages validate              # validate all gyms in gyms/
python -m gymhub.packages validate gyms/llm     # validate one gym
python -m gymhub.packages pack gyms/<id>        # produce a .tar.gz bundle
```

### Gym Markdown (one file per gym; see `gymhub/GYM_MARKDOWN.md` and the `write-gym` skill)
```bash
python -m gymhub.cli gym new my-gym.gym.md               # template
python -m gymhub.cli gym check my-gym.gym.md --json      # compile and check; errors with line numbers
python -m gymhub.cli gym validate my-gym.gym.md   # also run the tests
python -m gymhub.cli gym export gyms/<id>                # any package as Gym Markdown
```

### Local board (no cloud)
```bash
gymhub-board                                     # board + runner in one command (gymhub/local.py), http://localhost:8080
GYMHUB_STORE=sqlite:.cache/gymhub/dev.db GYMHUB_AUTH=dev uvicorn gymhub.board.app:app --port 8080
python -m gymhub.runner &                        # separate terminal: picks up jobs from the store; with a sqlite store it also publishes the courses (gymhub/courses.py)
python -m gymhub.cli gym seed                    # publish the courses to GYMHUB_STORE without a runner
```

### GPU gym CLI
```bash
gpu-gym test <exercise> [--lang cuda|pytorch|triton|all]
gpu-gym bench <exercise>
gpu-gym watch <exercise>
gpu-gym compile-db    # regenerate compile_commands.json after changing a CUDA signature
```

### Compute pool
```bash
pool server --data ~/pool-data --port 8765
pool run --gpus 1 --image nvcr.io/nvidia/pytorch:25.01-py3 -- python train.py
```

## Architecture

This repo contains four independent packages (all installed as one with `pip install -e ".[hub]"`):

### `gpu_gym/` — GPU kernel exercises (CUDA / PyTorch / Triton)
Entry point `gpu-gym`. Exercises live in `gyms/gpu/problems/<slug>/` (problem.py, PROBLEM.md, HINTS.md, starter/). User solutions go in `gyms/gpu/solutions/<slug>/` (paths in `gpu_gym/paths.py`). `gpu_gym/runner.py` compiles and runs CUDA with ctypes, times it, and compares to the PyTorch reference in `problem.py`. Progress stored in `gyms/gpu/progress.json` (`GPU_GYM_PROGRESS_FILE` overrides it).

### `gymkit/` — Python and C++ grader runtime
Shared by `llm-gym` and `algo-gym`. `gymkit/gym.py` loads a gym from a directory, `gymkit/runner.py` loads the solution module and runs test cases defined in `problem.py`. `gymkit/cases.py` defines the JSON tensor-spec format for custom test cases. C++ solutions are compiled by `gymkit/cpp.py`.

### `gymhub/` — Web board + runner daemon
- **`gymhub/board/app.py`**: FastAPI app. Never executes user code — it only writes job docs to the store. Mounts `gymhub/board/workers.py` (worker-token API), `gymhub/board/gyms.py`, and `gymhub/board/authoring.py`.
- **`gymhub/store.py`**: Two-backend document store. `FirestoreStore` for production (the board on Cloud Run uses a service account; the installed runner uses a worker token and the board's `/v1/worker` API, `gymhub/runner/backends.py` `ApiBackend`; direct store access is for local dev). `SqliteStore` for local dev and tests. Selected by `GYMHUB_STORE=firestore:<project>` or `GYMHUB_STORE=sqlite:<path>`.
- **`gymhub/runner/pooled.py`**: With `GYMHUB_EXECUTOR=pool`, isolated jobs are graded as pool jobs (repo-shaped `/work`: graders + answer-free gym slice + submission + progress) on any pool machine. With `GYMHUB_DISPATCH_ONLY=1` (or `gymhub-runner --dispatch-only`) the runner is a dispatcher: every job, owner's too, all actions and validation, goes to the pool and nothing executes locally. Imports `pool.client` (gymhub may depend on pool, never the reverse). The board's Compute page lists pool machines and hands out machine codes via `gymhub/board/poolmachines.py` (`GYMHUB_POOL_SERVER`, `GYMHUB_POOL_TOKEN`); a board that cannot reach the pool server shows the `pool_report` the dispatcher sends in its heartbeat (`gymhub/poolview.py`).
- **`gymhub/runner/`**: Runner daemon (`gymhub-runner` entry point). `runner/agent.py` polls the store for jobs, delegates to `runner/adapters.py` (one `GymAdapter` per gym), and sends other users' jobs to the compute pool (`runner/pooled.py`); it never runs them itself. `runner/capabilities.py` reports what the machine can run.
- **`gymhub/gymmd.py`**: Gym Markdown, a whole gym in one `.gym.md` file, compiled into a package (public/private split, shuffles, Python graders from `solve(...) == expected` tests). The board accepts `.gym.md` uploads; it only parses them.
- **`gymhub/packages.py`**: Gym package format. Validates and packs `gym.json` + exercises. `gymhub/gymversions.py` manages uploaded versions; workers validate each before publishing.
- **`gymhub/routing.py`**: Decides which machines can run which jobs based on `requires` (GPU, nvcc, g++) and gym membership.
- **`gymhub/store.py` collections**: `gyms/`, `problems/`, `progress/`, `jobs/`, `runners/`.

Auth modes (`GYMHUB_AUTH`): `iap` (Identity-Aware Proxy), `app` (in-app Google OAuth), `local` (name and password kept by the board, accounts from invite links; `gymhub-board --lan`), `dev` (no auth, local only).

### `pool/` — Compute pool (independent of gymhub)
Runs containers on machines you register. No imports from gymhub — designed to eventually move to its own repo.
- **`pool/server.py`**: FastAPI server. Pools, roles (admin/member/viewer), quotas, artifact storage. SQLite (`--data`) or Postgres (`--db postgresql://...`).
- **`pool/agent.py`**: Machine-side daemon. Outbound only. Caches chunks of artifacts (content-addressed, 64 MiB each), measures network speed every 6 h. Runs jobs in Docker with dropped capabilities and no network.
- **`pool/scheduler.py`**: Placement: fair-share by GPU-hours, then `--priority`. Checks model fits (full/offload/stage) and optionally network cost (`--flops`/`--dtype`/`--bytes-out`). Prefers machines that already hold the inputs.
- **`pool/artifacts.py`**: Content-addressed chunk store. Manifests + chunks. Deduplication is per-uploader (never reveals what others stored).
- **`pool/migrations.py`**: Numbered schema migrations (SQLite and Postgres), applied at server start and recorded in `schema_version`. Migration 1 is the pre-migration schema. To change the schema, append a migration; never edit a shipped one.
- **`pool/metrics.py`**: Prometheus text for `GET /metrics` (server admin token or `--metrics-token`).
- **`pool/spec.py`**: Job spec normalization and validation. Source of truth for fields, limits, and defaults.
- **`pool/models.py`**: Model metadata (reads HuggingFace safetensors headers for parameter counts, MoE topology, bytes).
- **`pool/peers.py`**: Machine-to-machine traffic. The server is the directory (LAN address + peer port per agent); agents measure links (islands), serve chunks to each other against server-signed tickets, and gang ranks meet through the server's rendezvous (`gang_members` table, `scheduler.rendezvous`).

### Gym package format (`gyms/<id>/`)
```
gym.json             manifest: id, title, runtime (gymkit|gpu_gym), langs, tracks, requires
problems/<slug>/
  problem.json       public: title, difficulty, summary, kind
  PROBLEM.md         public: statement
  HINTS.md           public: progressive hints
  starter/<file>     public: blank solution per language
  solution/<file>    private: reference solution (used to validate the package)
  problem.py         private: grader — defines Problem(params, reference, cases, bench_case, ...)
```

The board and learners receive only the public parts. Machines that grade receive the full package. The GPU gym is an ordinary package at `gyms/gpu/`.

### Key env vars
| Variable | Purpose |
|---|---|
| `GYMHUB_STORE` | `sqlite:<path>` or `firestore:<project>` |
| `GYMHUB_AUTH` | `iap`, `app`, or `dev` |
| `GYMHUB_EXECUTOR` | `pool` (default): other users' code is graded on the compute pool; `local`: never use the pool |
| `GYMHUB_TRUST_JOBS` | `1`: a direct-mode runner runs isolated jobs itself, unisolated (local dev and tests only) |
| `GYMHUB_LOG_FORMAT` | `json` (Cloud Logging lines; the default when `K_SERVICE` is set on Cloud Run) or `plain` |
| `GYMHUB_UI_CHECKS` | `required`: fail (not skip) the headless-browser page checks when Playwright is missing (CI) |
| `GYMKIT_PROGRESS_FILE` | Override progress.json location (board uses scratch copies) |
| `POOL_TEST_POSTGRES` | Postgres DSN for pool integration tests |
