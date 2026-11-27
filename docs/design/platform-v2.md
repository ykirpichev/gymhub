# Human gym v2: many users, their own compute, their own gyms

Status: built. Phases 1–3 (multi-user core, bring your own compute, gyms as packages) have shipped, and so has most of Phase 4 (non-coding gyms, the learning loop, staging). What is left is on the Phase 4 milestone. The compute pool that grew out of this design is issue #45. The text below is the original proposal, kept for its reasoning.

Today the system serves one person. The board runs on Cloud Run behind IAP for one Google account. One runner on one machine executes every job with the owner's full permissions. The catalog is read from local files, and solutions and progress live as files on that machine. This document describes the changes needed so that:

1. every user can connect their own compute (workers), and the design holds up at scale;
2. every user can create gyms and keep them private or share them;
3. the start page is organized around what a user wants to do next;
4. a list of further improvements, ranked.

The last section lists the decisions that need an owner's answer before building.

---

## 0. What changes, in one picture

```
            browser (board UI)
                   │  HTTPS, user session
                   ▼
   ┌──────────────── Cloud Run: API ────────────────┐
   │ auth · gyms · problems · workspaces · jobs     │
   │ worker API (lease / heartbeat / logs / result) │
   └──────┬───────────────────────┬─────────────────┘
          │                       │
      Firestore              Cloud Storage
      users, gyms, ACLs,     gym bundles,
      jobs (leases),         final job logs,
      progress, live logs    artifacts
          ▲
          │  HTTPS long-poll, outbound only, per-worker token
   ┌──────┴───────┐   ┌──────────────┐   ┌──────────────┐
   │ worker: your │   │ worker: Ann's│   │ platform CPU │
   │ DGX Spark    │   │ RTX 4090 box │   │ pool (opt.)  │
   │ sandboxed    │   │ sandboxed    │   │ Cloud Run    │
   └──────────────┘   └──────────────┘   └──────────────┘
```

The main shifts:

| Today | v2 |
| --- | --- |
| The source of truth for solutions and progress is files on the runner machine | Solutions, drafts and progress live in the cloud, per user. Workers are stateless executors. |
| The gym catalog is published by the runner from its local repo | Gyms are versioned packages stored in the cloud and published by their authors |
| The runner talks to Firestore with the owner's credentials | Workers talk only to a worker API, each with its own revocable token |
| Jobs run as the owner, with no isolation | Every job runs in a container with no network and limited resources, with the GPU passed through |
| IAP lets in one email | App-level sign-in (Google or GitHub) for invited users, with roles |

---

## 1. Users and sign-in

- **Sign-in:** use Firebase Authentication / Identity Platform with Google and GitHub sign-in. The API verifies the ID token on every request. Cloud Run becomes publicly reachable, and the app enforces access.
  - Why not keep IAP: IAP on a personal project needs an OAuth client in "testing" mode, which is capped at 100 test users, and every new user means an IAM change. It works well for one person, poorly for a community.
- **Admission:** start invite-only. An admin creates invite links, or allows email domains. Add a waitlist page later. Opening sign-up is one configuration switch.
- **Roles:** `admin` (platform), plus per-gym roles (owner, editor, learner). Workers have an owner.
- **Quotas per user:** jobs per hour, total job minutes per day, and gyms owned. Quotas protect shared resources and the API.

## 2. Data model (Firestore)

Decision: **stay on Firestore**. It already runs the board, needs no server to manage, and has a free tier. The design keeps every query on a single field or a declared composite index, and keeps the job queue transactional:

```
users/{uid}                       email, name, role, quotas, created_at
invites/{code}                    created_by, email?, expires_at, used_by
gyms/{gymId}                      owner, title, description, visibility, current_version, forked_from, tags, member_uids[]
gyms/{gymId}/versions/{v}         bundle_uri, manifest, status (validating|published|failed), created_at
gyms/{gymId}/problems/{key}       title, difficulty, track, order, kind, langs, statement, hints, starters   (current version)
users/{uid}/workspaces/{key__lang} code, updated_at                       -- the user's saved solution
users/{uid}/progress/{gymId}      per-problem state (same shape as progress.json today)
attempts/{id}                     uid, gym, problem_key, lang, action, status, ok, metrics, worker_id, created_at, log_uri
workers/{id}                      owner, name, token_hash, capabilities, labels, granted_uids[], granted_gyms[], status, last_seen
jobs/{id}                         attempt_id, uid, requirements, pool, status, lease_owner, lease_expires_at, tries, created_at
```

- **Visibility checks** use array fields such as `member_uids` and `granted_uids`, so each is one `array-contains` query.
- **The job queue** stays in Firestore:
  - a worker leases a job in a transaction (`status == queued` and `pool in [...]`, ordered by `created_at`, limit a few, then a conditional update);
  - a heartbeat extends `lease_expires_at`;
  - a sweeper returns expired leases to the queue.
  - Firestore handles this comfortably for hundreds of workers. If dispatch ever becomes the bottleneck, it can move to Pub/Sub pull subscriptions without changing the worker protocol.
- **Live logs** are append-only chunks in `attempts/{id}/logs/{seq}`, streamed to the browser with SSE from the API. Final logs go to GCS.
- `problem_key` is `gymId/slug`, so progress survives new gym versions. If a newer version changes a problem's tests, the user's solve is shown as "solved on v3, tests changed in v4".

## 3. Workers: bring your own compute

### 3.1 Registration (one command, no cloud credentials)

1. In the board, go to **Compute → Add worker**. The user picks a name and gets a one-time code that expires in 10 minutes.
2. On their machine, the user runs one command:
   ```
   curl -fsSL https://<board>/install.sh | sh -s -- --code ABCD-1234
   ```
   The script installs `gymhub-worker` (a single static binary or a pipx package) and checks Docker or Podman and the NVIDIA Container Toolkit. It exchanges the code for a **worker token** that belongs to that worker only. It then installs a systemd user service, or launchd on macOS.
3. The worker probes its capabilities and reports them:
   - OS and architecture; CPU cores and RAM;
   - each GPU's model, compute capability, memory, driver and CUDA version, and whether MIG is enabled;
   - available runtimes (docker/podman, gVisor).
4. The board shows the worker online. The user can rename, pause, drain, revoke or delete it. Revoking kills the token.

The worker token is an opaque random secret; the server stores only its hash. The worker exchanges it for a short-lived JWT, valid for one hour. The worker holds no Google Cloud credentials and can do only four things: lease a job, send heartbeats, stream logs, and report results.

### 3.2 Worker protocol (outbound only, works behind NAT)

```
POST /v1/worker/lease      {capabilities, free_slots}   → 200 {job, lease_ttl} | 204 (long-poll, up to 30 s)
POST /v1/worker/heartbeat  {job_id?, progress}          → extends the lease; answers {cancel: true} when cancelled
POST /v1/worker/logs       {job_id, seq, chunk}         → append (also fanned out live to the browser)
POST /v1/worker/complete   {job_id, ok, exit, metrics, summary}
```

- **Leases:** a lease lasts 60 s and heartbeats extend it. If a worker disappears, the lease expires and the job goes back to the queue, at most `max_tries` times. Completing is idempotent, keyed by job id and lease id.
- **Scale:** long-polling from thousands of workers is cheap on Cloud Run, which handles concurrent async requests. Each lease call is one indexed Firestore query plus a transaction. Around 10k workers, move dispatch to Pub/Sub pull subscriptions per pool, using attribute filters per capability class. The protocol stays the same.
- **Job payload:**
  - the submission;
  - the grader bundle reference (a signed GCS URL plus a content hash, which the worker caches);
  - the runtime image digest;
  - limits (time, memory, CPU, GPU);
  - the action (run, submit or bench).
  The worker verifies the hash before running anything.

### 3.3 Routing: who runs what, where

Each gym version declares requirements, and a job inherits them, for example:
```yaml
requires:
  gpu: { vendor: nvidia, min_compute_capability: "8.0", min_memory_gb: 8 }
  arch: [x86_64, aarch64]
  runtime: cuda-12
```
The scheduler matches a job to a worker when **all** of these hold:
1. **Capabilities:** the worker's capabilities satisfy the job's requirements.
2. **Permission:** the submitter may use the worker. By default only the owner can. Owners can share with specific users, with everyone in a gym ("my students"), or with a group.
3. **Preference:** the user's preferred worker when online. Otherwise any eligible worker, ordered by queue length and then by "same hardware as your last benchmark".

**Benchmarks record the worker's hardware.** Best times and any leaderboard are tracked per hardware class (for example "NVIDIA GB10" or "RTX 4090"), so times are never compared across different GPUs.

**Exclusivity:** bench jobs take a GPU exclusively. Test and run jobs can share a GPU only if the worker owner allows it; default is one job per GPU.

### 3.4 Live output

The worker sends log chunks about once a second. The API appends them to Firestore (`attempts/{id}/logs`), and the browser follows over **Server-Sent Events** instead of polling. Final logs go to GCS (`logs/<attempt>.txt`), and the attempt row stores the URI. Run-mode logs are deleted after 15 minutes, as today.

### 3.5 Sandboxing (required before any sharing, recommended always)

Every job runs in a fresh container:
- `--network=none` and a read-only image;
- the working directory on tmpfs, with only the submission and grader bundle mounted;
- a non-root user, `no-new-privileges`, the default seccomp profile, a pids limit, memory and CPU limits, and a wall-clock timeout;
- GPUs passed through with the NVIDIA Container Toolkit, using `--gpus device=N` for the leased GPU only.
- **Untrusted code** (workers shared beyond their owner): run under **gVisor** (`runsc`, whose `nvproxy` supports CUDA), or on MIG slices on data-center GPUs. Reset the GPU between jobs from different users.
- **Runtime images:** each gym declares a runtime image. The platform publishes base images (`gymhub/python`, `gymhub/cuda-torch`, `gymhub/cpp`), and gym authors can extend them. Images are pinned by digest and cached on the worker.

**Hidden tests:** a learner can read any grader bundle that runs on their own machine. That's fine for practice gyms. For gyms with leaderboards or certificates, grading must run on **trusted workers** (the gym owner's or the platform's), and the gym says so (`grading: trusted`).

### 3.6 Platform workers (optional, for adoption)

Users without a GPU should still be able to start. Two options:
- a small **CPU pool on Cloud Run Jobs** for gyms that need no GPU (algo, Python LLM-tokenizer tasks), metered per user quota;
- on-demand GPU workers (Cloud Run with L4, or spot VMs) later, if there's a budget.

## 4. Gyms: anyone can create, private or shared

### 4.1 A gym is a versioned package

```
my-gym/
  gym.json              id, title, description, runtime, requires, langs, tracks (visibility and grading come with #15)
  problems/<slug>/
    problem.json        title, difficulty, summary, kind (code | answer | choice | essay | checklist)
    PROBLEM.md          statement (public)
    HINTS.md            progressive hints (public, revealed one at a time)
    starter/…           starter files per language (public)
    grader/…            reference solution, hidden tests, problem.py / harness (private: workers only)
```

Built in #13 (`gymhub/packages.py`); JSON rather than YAML so the board needs no extra parser. This is a generalization of what `gymkit` already does. The three existing gyms become three packages:
- **llm and algo:** already in this shape.
- **gpu:** the `gpu_gym` checker becomes the grader of the `cuda-torch` runtime.

The browser only ever receives the public parts.

### 4.2 Authoring paths

1. **Web editor** (simple gyms): create a gym, then add problems with a Markdown statement, starter code, and either input/expected-output tests or a reference function. Good for teachers and quick personal sets.
2. **Git repo** (serious authors): connect a GitHub repo, and each push to a branch or tag publishes a version. This is how the current gyms would be maintained.
3. **CLI:** `gymhub gym push ./my-gym`, then `gymhub gym validate` locally.

**Publishing pipeline:**
1. Upload the bundle and check its manifest and schema.
2. Run a **validation job** on an eligible worker: every problem's reference solution must pass its own tests, and every starter must fail them.
3. Mark the version published.

The version learners see changes only on publish. Learners' progress is keyed by `problem_key`, so it survives new versions.

### 4.3 Visibility and sharing

| Visibility | Who sees it | Who can submit |
| --- | --- | --- |
| private | owner and invited editors | the same people |
| shared | plus specific users, groups, or anyone with an invite link | the same people |
| unlisted | anyone with the link (after sign-in) | the same people |
| public | listed in **Explore**; optionally reviewed by an admin first | everyone signed in |

- **Fork:** copy a public or shared gym into your own private gym to adapt it. The fork keeps a "forked from" link.
- **Editors** can change a gym; **learners** can only practice.
- **Owner stats:** per problem, solve rate and the attempts where people get stuck.

### 4.4 Non-coding gyms

The `kind` field in `problem.yaml` chooses the answer widget and the grader:

| kind | UI | grader |
| --- | --- | --- |
| code | editor (simple or IDE view), Run / Submit / Bench | harness on a worker |
| answer | text or number box | exact, regex or numeric tolerance; runs on the API, no worker needed |
| choice | radio buttons or checkboxes | answer key, on the API |
| essay | text area | rubric + LLM grader on a worker or the API; shows feedback |
| checklist | tick items, add notes | self-reported; progress only |

Everything else is shared: statement, hints, history, progress and tracks. Only `code` (and essays graded by a local model) needs a worker.

## 5. Start page and navigation

Today the start page mixes gym cards, raw compute tables and a job log. Proposed:

**Top navigation:** `Home · My gyms · Explore · Compute · Create`, with the user menu on the right. A small status chip shows "2 workers online" or "no compute connected". The raw tables move off the home page.

**Home** answers "what should I do now?":
1. **Continue:** a large card for the exercise touched most recently, showing the gym, title, language and last result, with a *Resume* button.
2. **Up next:** one suggested exercise per enrolled gym (the next unsolved one in track order), each with its difficulty and a *Start* button.
3. **Progress:** solved this week, current streak, and a small activity heatmap. Recent personal bests ("matmul 0.84 → 0.61 ms").
4. **Your gyms:** compact cards with progress rings, plus *Explore gyms* and *Create a gym*.
5. **Empty states and onboarding** for new users:
   1. pick gyms to join;
   2. connect compute (or use the shared CPU pool);
   3. solve a warm-up exercise.
   Each step is explained in one sentence.

**Gym page:** a header with description, progress and a *Continue* button. Tracks are shown as a path, each a row of exercises with status dots. Filters: unsolved, by language, by difficulty. Owner tools (edit, share, stats) appear for owners.

**Compute page:** the user's workers (status, GPU, current job, queue), *Add worker*, sharing settings, and recent jobs with filters. Everything the home page shows about compute today moves here.

**Exercise page:** unchanged; the simple view is the default and the IDE view is opt-in.

## 6. More improvements (ranked)

1. **Custom input in Run mode:** add your own test case (args as JSON) next to the built-in ones. This is the most-requested debugging aid on similar platforms.
2. **Submission history with code:** every submit stores its code; view the diff against the previous submit; restore any version.
3. **AI tutor (optional, with a budget):**
   - *Explain this failure* explains the failing case and the output;
   - *Review my solution* covers correctness, complexity and GPU efficiency;
   - generated hints after the authored ones run out.
   Uses the Claude API with per-user monthly limits; gym owners can turn it off.
4. **GPU profiling for bench:** an optional Nsight Compute summary (achieved occupancy, DRAM throughput, the warp stall reason that dominates) next to the bench numbers.
5. **Spaced review:** resurface solved exercises after 1/7/30 days; *Re-solve from scratch* resets the starter in a scratch workspace.
6. **Leaderboards per hardware class** (opt-in per gym; requires trusted grading).
7. **CLI sync for local editors:** `gymhub pull` / `gymhub push` and `gymhub watch` keep a local folder in sync with the cloud workspace. Today's VS Code and Cursor flow keeps working without local files being the source of truth.
8. **Notifications:** browser notifications and an optional email when a long bench finishes or a worker goes offline.
9. **Engineering hygiene:**
   - CI on GitHub Actions: unit tests for gymkit, the harnesses and the API, plus the headless-browser checks used during development;
   - a staging environment, with Cloud Build deploying `main` to staging and tags to production;
   - structured logs and metrics: queue wait, job duration, worker utilization, error rate;
   - budget alerts;
   - a Firestore TTL policy for old jobs and attempts.
10. **Abuse and safety:**
    - rate limits and quotas;
    - a size cap on submissions and outputs;
    - an audit log for sharing changes and token issuance;
    - CSP headers;
    - dependency pinning;
    - a reviewed list of runtime images.

## 7. Plan

| Phase | Scope | Outcome |
| --- | --- | --- |
| **1. Multi-user core** | Sign-in and invites; Firestore schema; workspaces and progress per user in the cloud; worker API with tokens; containerized execution; the DGX Spark becomes "worker #1" (owner-only); new Home page and navigation | Invited users can practice the existing gyms on the owner's compute, safely |
| **2. Bring your own compute** | Installer and registration codes, capability probing, routing, Compute page, sharing a worker with users or gyms, SSE logs | Each user runs on their own GPU or CPU; owners can lend workers |
| **3. Gyms as packages** | `gym.yaml` / `problem.yaml`, bundles in GCS, validation jobs, visibility and sharing, Explore, fork, web editor for simple gyms, Git-based publishing; migrate gpu/llm/algo | Anyone can create and share gyms |
| **4. Beyond code and polish** | Answer, choice, essay and checklist kinds; custom Run input; submission history; AI tutor; platform CPU pool; leaderboards | Non-coding gyms; better learning loop |

Each phase ships on its own, and the system stays usable between phases.

## 8. Decisions needed

1. ~~Scale target / database~~: decided, stay on Firestore.
2. **Sign-up:** invite-only, or open with quotas?
3. **Shared workers:** may a worker ever run code from people its owner doesn't know (a community pool), or only from explicitly granted users and gyms? This decides whether gVisor is needed in Phase 2 or can wait.
4. **Platform compute:** pay for a CPU pool so users without a GPU can start? GPU too?
5. **Competitive features:** are leaderboards or certificates in scope? If so, trusted grading is required.
6. **AI tutor:** in scope, and who pays for it (the platform, or each user's own API key)?
