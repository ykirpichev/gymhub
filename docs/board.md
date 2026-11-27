# Using the board

The board is a web app that lists every gym, has an in-browser editor, and shows your progress. It never runs code itself: it stores jobs, and machines you add pick them up and grade them (see [Self-hosting](self-hosting.md)). This page covers what learners and gym authors do on it.

## Gyms and their command-line tools

| Gym | CLI | Exercises |
| --- | --- | --- |
| GPU gym | `gpu-gym` | CUDA / PyTorch / Triton kernels ([GPU gym](gpu-gym.md)) |
| LLM gym | `llm-gym` | `gyms/llm`: RMSNorm, RoPE, attention, KV cache, BPE, sampling, ... in PyTorch |
| Algo gym | `algo-gym` | `gyms/algo`: intervals, stacks, heaps, tries, LRU, union-find, graphs and DP in Python (the first nine also in C++), with time limits |

## Run, submit and test cases

**Run vs Submit.** *Run* (Ctrl+Enter) tries the code in the editor on every test case and prints, for each one, the inputs, your output, the expected output, and whatever your code printed. It keeps going after a failure and records nothing: the code goes to a scratch copy, not your solution file, and the job is not listed in history. *Submit* (Ctrl+Shift+Enter) saves the file, runs the full tests, and records progress. On the command line: `gpu-gym run <exercise> [--file path]`, `llm-gym run ...`, `algo-gym run ... --lang cpp`.

**Try in browser.** Python exercises also have *Try in browser* (*Try* on a phone): it runs the exercise's built-in example cases on the code in the editor right in your browser, with [Pyodide](https://pyodide.org) and NumPy, so it gives instant feedback with no runner online, even on a phone. Nothing is recorded, there are no time limits (a run that takes over 20 seconds is stopped), and an exercise that checks answers its own way may accept an answer that differs there. *Submit* grades. The first try downloads Python into the browser (about 10 MB, then cached). The examples are the same inputs and expected outputs Run shows; the board never sends a grader to the browser. A machine that has the grader builds them when it publishes or validates the gym (`gymkit/examples.py`), keeping only cases the reference solution passes there, so exercises whose grader drives the solution its own way (a class, say), PyTorch exercises and C++ have none.

**Custom cases.** *Cases* (or the IDE's CASES tab) holds your own test cases for Run, prefilled from the problem's example. One JSON object per case, every argument by name; tensors are `{"tensor": [[1, 2], [3, 4]]}`, `{"randn": [64, 128], "seed": 1}`, `{"randint": [16], "low": 0, "high": 10}`, `{"zeros": [4]}`, `{"empty": [32]}` and so on (full format in `gymkit/cases.py`). The reference computes the expected output. Tick *Only my cases* to skip the built-in ones. On the command line: `gpu-gym run vector_add --cases my_cases.json --only-custom` (same flags for `llm-gym` and `algo-gym`).

**Browser IDE (optional).** Exercises open in a simple editor page by default. *IDE view* switches to a VS Code-style workspace (remembered per browser; *Simple view* switches back): explorer with every exercise and its files, problem and hints beside the code, tabs, output panel, status bar, Ctrl+P quick open, and completions for CUDA, C++, PyTorch, Triton and Python. Nothing to install. Locally, VS Code / Cursor work too: Ctrl+Shift+B submits the open exercise file in any gym; *Run Task → gym: run this file* is the debug run.

The web editor and your local files are the same files. Submitting from the board writes `solutions/...` first (the previous version is backed up under `.cache/gymhub/backups/`). Edits saved locally show up on the board within a few seconds.

## Gyms

### Adding a gym

Every gym is a **package** (`gymhub/packages.py` describes the format):

```
gyms/<id>/gym.json                  manifest: title, runtime (gymkit | gpu_gym), requires, langs, tracks
gyms/<id>/problems/<slug>/
  problem.json  PROBLEM.md  HINTS.md  starter/<file per language>     public: shown on the board
  problem.py  cpp/harness.cpp  cpp/reference.cpp  ...                  grader: only machines get it
```

Copy an exercise from `gyms/llm` or `gyms/algo`, list it in a track, then check the package with `python -m gymhub.packages validate gyms/<id>`. The runner publishes valid gyms on start. The GPU gym is `gyms/gpu`. Each gym's `solutions/` and `progress.json` are yours and gitignored: a fresh clone copies every `starter/` into `solutions/` on first run.

The easiest way to write a gym is one [Gym Markdown](../gymhub/GYM_MARKDOWN.md) file (`python -m gymhub.cli gym new my-gym.gym.md`). [`gyms/README.md`](../gyms/README.md) describes the bundled courses and every exercise kind.

### Uploading a gym (no repo access needed)

A gym does not have to live in this repo. Anyone allowed to run code can upload one at the bottom of the board's **Gyms** page, and each upload becomes a new **version**:

1. Start from `examples/gyms/hello`. Give the gym its own `id` in `gym.json`, and add a reference solution for each exercise in `problems/<slug>/solution/<file>`. Machines use it to prove the exercise can be solved; learners never see it.
2. Pack the gym with `python -m gymhub.packages pack path/to/gym`, which checks it first, then upload the `.tar.gz` on the Admin page.
3. The board checks the manifest and every `problem.json`. It never runs the gym's code itself.
4. A machine then validates the version on the compute pool:
   - every grader loads;
   - every **starter fails** its tests;
   - every **reference solution passes** them.
5. If every check passes, the version is published, and learners see only that version. If any check fails, the version is kept with its report and never reaches learners.

Jobs in an uploaded gym use the version that was live when they were submitted. Everyone's code and progress stay on the board, including the owner's.

When a new version changes an exercise's tests, everyone who solved it on older tests sees **tests changed since you solved** until they submit again. A change to the statement or hints alone does not trigger this.

Bundles are stored in the project's `gs://<project>-gym-bundles` bucket, which `deploy/gcp.sh deploy` creates. For now, uploaded gyms use the gymkit runtime (Python and C++ graders).

### Sharing gyms

Each gym has a visibility, set by its owner in the gym's **Manage** panel:

| visibility | who sees it |
|---|---|
| private | the owner and the gym's editors (new uploads start here) |
| shared | also the people it is shared with, by email or through its join link |
| unlisted | anyone on the board who has the link; it is not listed in Explore |
| public | everyone on the board; it is listed in **Explore**, with tags and a "needs NVIDIA GPU" badge |

Roles:
- **Owners** change these settings. Admins can act as the owner of any gym.
- **Editors** can upload new versions and see the gym's stats.
- **Learners** practise.

Gyms published from a machine's local files are public unless an admin changes that.

### Review and reports

**Review before public.** By default an admin approves members' gyms before they go public: choosing *Public* in **Manage** sends a request, listed under **Gyms waiting to go public** on the Admin page, and the gym keeps its current visibility until an admin approves. Admins' own gyms go public straight away. An admin can turn review off in **Access settings** on the Admin page; a board keeps whatever it last saved there. New versions of a gym that is already public are not reviewed again.

**Reports.** Anyone who can open a gym (other than its owner) can **Report** it from the gym's page: a reason (spam, offensive, harmful code, copyright, personal information, broken, something else) and optional details. One report per person and gym; reporting again replaces it. Only admins see reports, with the reporter's email, under **Reported gyms** on the Admin page, most reported first. For each gym an admin can:
- **Hide:** the gym becomes private and stays there. Its owner and editors still see it, with the admin's note. They cannot share it again; asking for *Public* goes to the admins, even with review off. An admin who approves that request, or makes the gym shared, unlisted or public themselves, lifts the hold.
- **Remove:** delete it for everyone, as if its owner had (learners keep their progress). Gyms published from a machine's local files can only be hidden.
- **Dismiss:** close the reports and leave the gym as it is.

The owner can **allow forks**. A fork copies the live version into a new private gym owned by the person forking it, who gets the graders and hidden tests too. The fork is validated again before it is published.

Owners and editors see **Progress** per exercise. It shows aggregate counts only, never names:
- how many people started and solved it;
- the solve rate;
- the average number of tries for those who solved it, and for those still stuck;
- the number of hints used.

### Writing a gym without files

- **In the browser:** on the Gyms page, choose **Write a gym in the browser**. Each exercise is defined by:
  - a statement and optional hints;
  - the parameters of `solve(...)`;
  - a starter and an optional reference solution;
  - tests, where each cell is a JSON value: inputs, and the expected output (blank means "compute it with the reference").

  **Publish a version** turns the draft into a normal package, validated like any upload. For an editor-made gym, **Edit in the browser** in its Manage panel opens the draft again. The editor makes Python gyms; for C++ or custom graders, upload a package instead.
- **From a terminal:**
  1. Create an API token under "Publish from a terminal or CI" on the Gyms page.
  2. Run `GYMHUB_BOARD=https://<board> GYMHUB_TOKEN=gp_… python -m gymhub.cli gym push path/to/gym --wait`.

  To run the machine's checks on your own computer first, use `python -m gymhub.cli gym validate path/to/gym`. Tokens can only upload versions and read their status.
- **From GitHub:**
  - **Public repository:** in the gym's Manage panel, connect the repository, the folder that holds `gym.json`, and a ref such as `refs/heads/main` or `refs/tags/v*`. Then add the webhook shown there. Each matching push publishes a version, recorded with its commit.
  - **Private repository:** run `gymhub.cli gym push --wait` in GitHub Actions, with the token stored as a secret.

Tokens and webhooks need the board in in-app sign-in mode (`AUTH_MODE=app`), because IAP blocks every request that doesn't come from a browser.

### Exercises that are not code

An exercise's `problem.json` can set `"kind"` to `answer`, `choice`, `essay` or `checklist` instead of `code`. The board grades these kinds itself, so no machine is needed. The private answer key lives in `answer.json` next to the statement; learners never receive it.

| kind | public (`problem.json`) | private (`answer.json`) |
|---|---|---|
| answer | `placeholder` | `{"answers": ["Paris"], "match": "text" or "number", "tolerance": 1e-6}` |
| choice | `choices: [{"id", "text"}]`, `multiple` | `{"correct": ["b"], "explanation": "…"}` |
| essay | `min_words`, `max_words` | `{"rubric": "…", "pass_score": 0.7}` |
| checklist | `items: ["…"]` | none; learners tick items off themselves |

How essays are graded depends on whether the board has an Anthropic API key:
- **With a key:** Claude (`claude-sonnet-5-5`, or `GYMHUB_ESSAY_MODEL`) grades essays against the rubric and gives per-criterion feedback. To set this up, run `deploy/gcp.sh anthropic-key`, then `deploy/gcp.sh deploy`.
- **Without a key:** after submitting, learners see the rubric and mark the essay done themselves.

`examples/gyms/mixed` has one exercise of every kind.

### AI tutor

On a code exercise, the **Tutor** tab offers three actions:
- **Explain my last failure:** uses your last failed submission's output and code.
- **Review my code:** correctness, complexity, and GPU efficiency for kernels.
- **Another hint:** for when the authored hints run out.

The tutor sees only what the learner sees, never grader files, and it teaches rather than solves. It runs on Claude (`claude-sonnet-5-5`, or `GYMHUB_TUTOR_MODEL`) and is available once the board has an API key (`deploy/gcp.sh anthropic-key`).

Each member has a monthly budget, `GYMHUB_TUTOR_MONTHLY_USD`, which defaults to $2. The board charges each request from the API's reported token usage, at the price of the model that answered, and admins have no limit. Owners can turn the tutor off for a gym in its Manage panel.

### Monthly AI cap

The board as a whole has a monthly cap on what it spends on Claude, for the tutor and essay grading together, admins included. It defaults to $20 (`GYMHUB_AI_MONTHLY_USD`), and admins can change it, or set 0 for no cap, under **AI spend** on the Admin page, which also shows this month's spend. Once the cap is reached, the tutor says it is paused until the 1st (UTC) and essays are self-marked against the rubric. Requests already running still finish, so the total can pass the cap by a few cents. The cap is the board's own count, so keep a budget alert on the Anthropic account as well.

### Profiling a kernel (bench --profile)

`gpu-gym bench <exercise> --profile` runs one extra call of your solution under Nsight Compute. On the board, tick **profile** next to Bench to do the same.

Only the kernels inside your solution call are measured. For the slowest ones it shows:
- duration, launch shape and registers per thread;
- achieved occupancy;
- how busy the SMs and memory are;
- L2 bandwidth;
- the main warp stall reason, with advice;
- a verdict: memory-, compute- or latency-bound.

GPU performance counters are root-only by default. To allow profiling on a machine, run once:

```
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

### Spaced review

Solved exercises come back on Home under **Review** 1, 7 and 30 days after you first solved them. That includes exercises solved from a terminal.

**Re-solve from scratch** opens the starter. Submitting runs the tests on a scratch copy, so your saved solution, and for the owner the file on disk, is never touched. A pass schedules the next review; after the third, the exercise leaves the queue.

### Leaderboards

A gym's owner can turn on **Leaderboards** in its Manage panel. Each exercise then ranks every person's best bench median, separately per language and per GPU.

Only benchmarks run on **trusted** machines count:
- a direct-mode runner, which holds the project's credentials;
- a machine owned by the gym's owner;
- a machine owned by an admin.

Results from members' own machines never appear, because they could report anything.

### Notifications

Under **Notifications** on the Compute page, you can let the board notify you in the browser in two cases:
- when a job you started finishes while the board is in a background tab;
- when one of your machines goes offline.

Each setting is stored with your account, and the browser asks for permission once. Email digests aren't available, because the project has no mail service.

### Practising from your own editor

These commands keep VS Code, Cursor or any editor in sync with the board:

```
python -m gymhub.cli login --board https://<board> --token gp_…   # token: Gyms page, "Publish from a terminal or CI"
python -m gymhub.cli pull algo                                     # ./algo/<exercise>/solution.py, PROBLEM.md
python -m gymhub.cli watch algo                                    # every save is pushed to the board
python -m gymhub.cli run algo/dijkstra/solution.py                 # every case with output, nothing recorded
python -m gymhub.cli submit algo/dijkstra/solution.py              # full tests, progress recorded
```

What you save locally shows up on the board, and code saved on the board comes back with the next `pull`. A pull never overwrites edits you haven't pushed unless you add `--force`.

Tokens work only for these exercise and job endpoints and for publishing gyms; settings, sharing and admin stay browser-only. Like the other token features, this needs the board in `AUTH_MODE=app`.
