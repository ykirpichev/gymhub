# Self-hosting the board

Run the board on your laptop with no cloud at all, or deploy it to Google Cloud for yourself and the people you invite. Either way the board never executes code. Machines you add pick up jobs and report results: your own jobs can be graded on your own machine, and everyone else's are graded in locked-down containers on the [compute pool](pool.md).

```
browser ──sign-in──> board (Cloud Run, or uvicorn locally) ──> store (Firestore or SQLite) <── runner / dispatcher ──> your machine or the compute pool
```

## Run it all locally (no cloud)

`gymhub-board` (after `./setup.sh && source .venv/bin/activate`) does all of this in one command: it checks what this machine can grade (`gymhub-board --check` only checks), then starts the board on http://localhost:8080 and a runner with the settings below. Variables you set yourself win; `--port` picks another port and `--no-runner` starts only the board. It refuses to listen on anything but localhost while there is no sign-in (to host it for others, see [Host it for a group](#host-it-for-a-group-without-google)). By hand:

```bash
export GYMHUB_STORE=sqlite:.cache/gymhub/dev.db
GYMHUB_TRUST_JOBS=1 python -m gymhub.runner &              # TRUST: a board only you use (see "Other users' code")
GYMHUB_AUTH=dev uvicorn gymhub.board.app:app --port 8080   # http://localhost:8080
```

No GPU or PyTorch is needed to start. Every gym is listed; one whose graders need something this machine lacks (the GPU and LLM gyms need PyTorch and an NVIDIA GPU) says so on its page, and its runs wait until a machine that has it connects. With a local store the runner also publishes the courses (`gyms/README.md`: ML coding, ML fundamentals, LLM fundamentals) at start, validating them first, and again when their files change (`--no-courses` to skip; `python -m gymhub.cli gym seed` does it without a runner). ML coding's exercises are graded like any uploaded gym's, so they need `GYMHUB_TRUST_JOBS=1` or a pool.

## Host it for a group, without Google

`gymhub-board --lan` runs the same board for a group: a study group on your network, a team, friends on your Tailscale network. People sign in with a name and a password the board keeps (`GYMHUB_AUTH=local`), so nobody needs a Google account and there is nothing to set up in a cloud console.

```bash
gymhub-board --lan                 # listens on every address, port 8080
```

On first start it prints a one-time link, valid for a day, that creates your admin account (named after your login, or `--owner <name>`). Open it, pick a password, and you are the board's owner: your own runs are graded on this machine, as with `gymhub-board`. Then invite people from the **Admin** page: each invite link lets one person (or as many as you set) create an account. An invite can be made for one name, and with role *admin*. Someone who forgot their password gets a new one from **Password reset link** next to their name on the Admin page; if you lose yours, `gymhub-board --lan --owner-link` prints a reset link. Anyone can change their password under **Your data**.

What else to know:

- **Names and passwords.** Names are 2 to 32 lowercase letters, digits, dots, dashes and underscores. Passwords need 8 characters and are stored as scrypt hashes. After 10 wrong passwords for a name in 15 minutes, sign-in for that name pauses. Sessions last 30 days; the key that signs them is kept beside the store (`.cache/gymhub/session.key`, mode 600), and deleting it signs everyone out.
- **Members' code.** Members can browse, edit and save. They can run code once an admin turns on *Members can run code* (Admin, Access settings), and even then their code is not run on your machine as you: it waits for a [compute pool](pool.md) machine, where it runs in a container with no network. For a group you trust completely, start with `GYMHUB_TRUST_JOBS=1 gymhub-board --lan` and their jobs run here, under your user, with no isolation.
- **Plain HTTP.** The board speaks HTTP. On a network you don't trust, reach it through Tailscale or an SSH tunnel (below), which encrypt the traffic, rather than over the open LAN.
- **Doing it by hand.** `gymhub-board --lan` sets `GYMHUB_AUTH=local`, `GYMHUB_ADMIN_EMAILS=<owner>` (the names of admins) and `GYMHUB_SESSION_KEY`. Any board can use the same mode with those variables, `--host` picks the address, and `--lan` is not needed when you use only an SSH tunnel: `GYMHUB_AUTH=local gymhub-board` keeps listening on 127.0.0.1.

### A GPU box with no screen

When the GPU is in a headless machine (a server, a DGX Spark, a Jetson), run the board there and open it from your laptop.

**SSH tunnel**, just for you: start `gymhub-board` on the GPU box as usual (no `--lan` needed: it stays on 127.0.0.1 there), and on your laptop:

```bash
ssh -N -L 8080:localhost:8080 you@gpu-box   # then open http://localhost:8080 on the laptop
```

Each person who has SSH access to the box can do the same; give them accounts with `GYMHUB_AUTH=local gymhub-board`, since everyone coming through a tunnel looks like localhost to the board.

**Tailscale**, for a group: install [Tailscale](https://tailscale.com/download) on the GPU box and on each person's device, and share the box with them from the Tailscale admin console. Then on the box:

```bash
gymhub-board --lan --host "$(tailscale ip -4)"   # reachable only over Tailscale, not the local network
```

People open `http://<box's Tailscale name or 100.x address>:8080`. For HTTPS and a stable name, keep the board on localhost and let Tailscale serve it: `GYMHUB_AUTH=local gymhub-board` and `tailscale serve --bg 8080` (it prints `https://<box>.<tailnet>.ts.net`). Invite links use the address you opened the Admin page at, so create them from the address your group will use.

## Deploy to Google Cloud (once)

```bash
gcloud auth login
gcloud billing accounts list
deploy/gcp.sh create <BILLING_ACCOUNT_ID>   # new project gymhub-xxxxxxxx, Firestore, service account
deploy/gcp.sh deploy                        # board on Cloud Run, IAP allows only your account
deploy/gcp.sh iap-client ~/Downloads/client_secret_*.json   # personal accounts only, see below
deploy/gcp.sh runner                        # how to give the runner a worker token (no Google login on the machine)
deploy/install-runner.sh                    # runner as a systemd user service, in worker-token mode
```

**IAP on a personal Google account** (no Workspace organization) needs its own OAuth client. Until it has one, the board answers `502 Empty Google Account OAuth client ID(s)/secret(s)` to everyone. In the Cloud Console for the project, under **Google Auth Platform**:

1. **Branding**: app name, your email as support and developer contact. **Audience**: External, leave it in *Testing*, add your Gmail as the only test user.
2. **Clients → Create client**: Web application. Download the JSON.
3. Edit the client and add the redirect URI `https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect`.
4. `deploy/gcp.sh iap-client <downloaded.json>`, then delete the JSON.

`deploy/gcp.sh status` prints the board URL. `deploy/gcp.sh budget 5` (or `BUDGET_USD=5` in `deploy/gcp.env`) sets a $5/month budget for the project; the billing account's admins get emails at 50%, 90% and 100% of actual spend and when the forecast passes 100% (`BUDGET_ALERT_CHANNEL` adds a Cloud Monitoring notification channel). Re-running changes the amount. `deploy/gcp.sh ttl` lets Firestore delete scratch documents (run jobs, quota counters, registration codes) once their `expire_at` passes. Board images are cleaned up automatically, keeping the 5 newest.

**Cost.** At personal or small-group scale this stays within Google Cloud's free tiers:
- The board scales to zero.
- Every machine costs about one Firestore read per poll: every 15 s when idle, every second while someone uses the board.

## Sign-in and inviting people

`AUTH_MODE` in `deploy/gcp.env` picks how the board checks who you are (a board on your own machine can instead use name-and-password sign-in, `local`: see [Host it for a group](#host-it-for-a-group-without-google)):

- `iap`: Identity-Aware Proxy signs people in with Google, and the board applies the same user model as `app`: `OWNER_EMAIL` is the admin, everyone else needs an invite link or an allowed domain. With `IAP_ACCESS=owner` (the default) IAP admits only your account. With `IAP_ACCESS=everyone` IAP admits any Google account and the board enforces the invites. This mode needs no console setup.
- `app`: in-app *Sign in with Google*. `OWNER_EMAIL` is the first admin. Everyone else needs an invite link, or an email at a domain you allow; both are managed on the **Admin** page. Members can browse, edit and save; they can run code once an admin turns on *Members can run code*. Their jobs then run on the compute pool, in its containers (see *Other users' code* below).

One-time console steps before switching to `app`. Both are under **Google Auth Platform** for the project:

1. **Clients:** open the OAuth client (its ID is `GOOGLE_CLIENT_ID` in `deploy/gcp.env`). Under *Authorized JavaScript origins*, add both board URLs, `https://gymhub-board-<hash>-uc.a.run.app` and `https://gymhub-board-<project-number>.us-central1.run.app`, plus `http://localhost:8080` for local testing.
2. **Audience:** while the app is in *Testing*, only the listed test users can sign in. Click **Publish app** so invited people can. It only asks for name, email and profile, so Google does not need to review it.

**The signed-out page.** In `app` mode, a visitor who is not signed in sees the sign-in card and, below it, a short landing page: the public gyms on the board with their exercise counts (from `GET /api/showcase`, which needs no sign-in and never names gym owners), how the board works, and a link to the source (`GYMHUB_SOURCE_URL`, by default this repository). Behind IAP visitors never reach it, because IAP asks them to sign in first, so a board open to the public should use `app`.

Then set `AUTH_MODE=app` and run `deploy/gcp.sh deploy`. The session-signing key is created in Secret Manager on the first deploy; rotating it signs everyone out. To roll back, set `AUTH_MODE=iap` and deploy again. Logs: `journalctl --user -u gymhub-runner -f`. To redeploy after changing `gymhub/board`, run `deploy/gcp.sh deploy` again.

## Privacy policy, terms and members' data

The board serves a privacy policy at `/privacy` and terms of use at `/terms`, linked from the sign-in
page and the footer. The text is a **draft** (`gymhub/board/legal/*.html`, marked as such on the page):
it is not legal advice, so have it checked against how you run your board, then delete its draft note.
Name yourself with `GYMHUB_OPERATOR` (who runs the board) and `GYMHUB_CONTACT_EMAIL` (where privacy
questions go); `GYMHUB_LEGAL_UPDATED` sets the "Last updated" date.

Members manage their own data on **Your data** (user menu): they can download everything the board holds
about them as JSON, and delete their account. Deletion is immediate and also deletes the gyms they own,
for everyone, and disconnects their machines (`gymhub/accounts.py` lists what happens to each
collection). Accounts on `GYMHUB_ADMIN_EMAILS` cannot delete themselves there, since they would be
recreated as admins at their next sign-in. Cloud Logging keeps request logs for its own retention period.

## Adding your own machine

On the board's **Compute** page, *Add a machine* gives a one-time code (valid for 10 minutes) and a command to run on that computer:

```bash
curl -fsSL https://<board>/install.sh | sh -s -- --code ABCD-1234
```

The installer, which needs no root:
1. downloads the worker bundle (gym code and exercise definitions, never anyone's solutions or progress) into `~/gymhub-worker`;
2. creates a Python environment, with CUDA PyTorch when an NVIDIA GPU is present;
3. registers the machine and stores its token in `~/.config/gymhub/worker-token`;
4. starts the `gymhub-worker` systemd user service.

Every machine is private by default: it runs only its owner's jobs, a machine or pool bridge added by an **admin** too. To let others use it, click **Share…** next to it (or **Share the pool…** in *Machines that grade* for the runner that connects the board to the pool) and pick people by email or *Everyone on this board*. Your own jobs run directly; other people's run only on the compute pool, in its containers with no network. Until a machine is shared with them, other people's jobs wait in the queue and the board tells them why. That includes validating a gym version someone else uploaded. Remove a machine from the same page. Behind IAP the command points at the worker service (`<service>-workers`, see *Runner modes*), so the computer needs no Google login.

## Runner modes

- **Worker token** (default, what `deploy/install-runner.sh` installs): the runner talks only to the board's worker API (`/v1/worker/*`) with a token made on the **Compute** page. The machine holds no cloud credentials. The runner polls about every second while someone has used the board in the last 15 minutes, and every 10 s otherwise.
  ```bash
  install -D -m 600 /dev/stdin ~/.config/gymhub/worker-token   # paste the token, then Ctrl+D
  echo 'RUNNER_API=https://<worker-url>' >> deploy/gcp.env && deploy/install-runner.sh
  gcloud auth application-default revoke                       # the runner does not need it
  ```
  With `AUTH_MODE=iap`, machines cannot pass IAP, so `deploy/gcp.sh deploy` deploys the same image a second time as `<service>-workers`: no IAP, only `/v1/...` and `/install.sh` (everything else is a 404), and every worker call needs a valid worker token. `deploy/gcp.sh status` prints its URL; that is `RUNNER_API`. With `AUTH_MODE=app`, `RUNNER_API` is the board's own URL.
- **Direct** (`RUNNER_DIRECT=1 deploy/install-runner.sh`, or `GYMHUB_STORE=... python -m gymhub.runner`): the runner talks to Firestore with this machine's application-default login, which can read and write the whole database. For local development (`GYMHUB_STORE=sqlite:...`) and machines you trust with the project.

**Revoking a machine** (Compute page) ends its token at once. The runner's next call is refused: it kills the job it is running (its process group, or its pool job), exits with status 78, and the systemd unit does not restart on that status (`RestartPreventExitStatus=78`). On the board the machine's running job goes back to the queue, or ends with an error if it has already had its 3 tries; a job that was being cancelled is closed.

**What a job inherits.** Every job the runner starts runs without the runner's credential variables (`GOOGLE_*`, `CLOUDSDK_*`, `AWS_*`, `GYMHUB_WORKER_TOKEN`, `SSH_AUTH_SOCK`, anything ending in `_TOKEN`, `_SECRET`, `_PASSWORD`, ...), and with `CLOUDSDK_CONFIG` pointing at an empty folder. The runner removes the token from its own environment, keeps the token file at mode 600 and makes itself non-dumpable, so a job cannot read the token from `/proc`. Those jobs run as the same user, so they could still read the token file itself; that is why only the owner's own jobs run on the machine (or anything with `GYMHUB_TRUST_JOBS=1`, for local development). Other users' jobs run on the compute pool, which never sees your home directory.

Either way a job is leased for 90 s and the lease is renewed while it runs. If the machine disappears, the job goes back to the queue, up to 3 tries.

## Other users' code

A runner never runs another user's code on its own machine. Isolated jobs (other users' code, and any job of an uploaded gym) are graded on the compute pool (`gymhub/runner/pooled.py`, see [Grading on the compute pool](#grading-on-the-compute-pool)), in self-contained grader images (`deploy/grader/`) with no network, a read-only root filesystem, every capability dropped and only the GPUs the job was given. Pool mode is the runner's default (`GYMHUB_EXECUTOR=pool`); it needs a pool member to submit as (`pool login`, or `POOL_SERVER` and `POOL_TOKEN`).

A runner without a pool member grades only its owner's jobs: it reports no sandbox, so the board never leases it anyone else's (`gymhub/board/workers.py`). A runner in direct mode, where no board stands in between to say whose job it is, leases no isolated job it cannot send to the pool, unless `GYMHUB_TRUST_JOBS=1` (local development and tests only: those jobs then run as your user).

The host-mount sandbox that used to run these jobs on the runner's machine (`gymhub/runner/sandbox.py`, `deploy/Dockerfile.sandbox`) is gone: it mounted the host's venv into an image that had to match the host's Ubuntu and Python, and failed on any machine where they differed. Runners from before still report their sandbox, and the board still lets them use it until they are updated.

## Grading on the compute pool

In pool mode (`GYMHUB_EXECUTOR=pool`, the default), the gymhub runner grades isolated jobs (other users' code, and any job of an uploaded gym) as pool jobs (`gymhub/runner/pooled.py`). The board and its worker API don't change: the runner still leases board jobs, streams their output and reports results, but the grading runs on any pool machine that fits. A pool job gets the graders, the gym as a slice without answer keys and reference solutions, the submitted file and the user's progress, which comes back updated. Only a public gym's jobs go to any machine of the pool: a job of any other gym runs only on pool machines owned by the pool member the runner submits as (`resources.machines: own`), and when none is online it is graded on the runner itself if it is the owner's own, or left to another worker.

**Run on.** The exercise toolbar has a *Run on* choice: *Auto* (any machine that can take the job), or one machine the learner could be routed to anyway: a board runner, or a pool machine of a bridge (pinned with `resources.machine`). The choice is kept per gym in the browser. A job waits in the queue while its machine is offline. The output says where a job ran (`Ran on jetson (Orin) · via spark-runner`), and so do the Runs and Compute pages.

The job runs in a self-contained grader image (`deploy/grader/`, built and pushed to GHCR by `.github/workflows/grader-images.yml`), so a pool machine needs no repo checkout and no venv. The graders are inside the image at `/opt/gym`; the job's `/work` carries only the gym slice and the submission. The image follows the machine's labels: `grader:cpu` (amd64 + arm64) for a job without a GPU, `grader:cuda` (amd64 + arm64, CUDA 13: x86 cards and DGX Spark) for a GPU job, and `grader:l4t-r36` on a Jetson (`gpu=tegra`, JetPack 6). With an image that is not a grader image (`GYMHUB_POOL_IMAGE`/`GYMHUB_POOL_IMAGES`), `/work` also carries `gymkit/` and `gpu_gym/`.

| Variable | Purpose |
|---|---|
| `GYMHUB_EXECUTOR=pool` | grade isolated jobs on the pool (the default; `local` never uses the pool, so the runner takes only its owner's jobs) |
| `GYMHUB_GRADER_IMAGE` | where the grader images live (default `ghcr.io/ykirpichev/gymhub/grader`) |
| `GYMHUB_GRADER_VERSION` | pin the grader images to a build (`<short sha>`; default: the floating tags) |
| `GYMHUB_POOL_IMAGE` | override: the image for pool jobs (Python with the graders' dependencies, and nvcc for CUDA) |
| `GYMHUB_POOL_IMAGES` | override: per-platform images, for example `gpu=tegra=<jetson image>` |
| `GYMHUB_POOL_ACTIONS` | actions sent to the pool (default `run,test,bench`); a benchmark's best time is recorded under the GPU model of the pool machine that measured it |
| `GYMHUB_POOL`, `POOL_SERVER`, `POOL_TOKEN` | the pool and member to submit as (default: `~/.config/pool/member.json`) |

**Dispatcher (worker v2).** `python -m gymhub.runner --api <board> --dispatch-only` (or `GYMHUB_EXECUTOR=pool` with `GYMHUB_DISPATCH_ONLY=1`) sends every job to the pool: the owner's own jobs too, every action including bench, and the validation of uploaded gym versions. It executes nothing on its own machine, reports no GPU, tools or sandbox of its own, and the board leases it only jobs it can send to the pool, so it can run next to the board or the pool server. Saving, resetting and hints of the owner's jobs stay file edits on its machine. It has a container image of its own, and `deploy/pool/compose.controller.yml` runs it next to the pool server: see [the pool's Docker section](pool.md#docker-the-default).

**Pool machines on the board.** With `GYMHUB_POOL_SERVER` (the pool server's URL) and `GYMHUB_POOL_TOKEN` (a pool member token; `GYMHUB_POOL` picks the pool) set for the board, the Compute page lists the pool's machines (GPUs, labels, state, owner, last seen), and admins get **Add a machine**, which asks the pool server for a machine code and shows its install command. Machines added this way belong to the board's pool member; give the dispatcher the same member, so that gyms that are not public may run on them. Board runners (`gymhub-runner`) that still execute jobs themselves are marked "old runner"; `deploy/install-runner.sh` prints a deprecation notice.

## CI and operations

- **Tests:** `.github/workflows/tests.yml` runs the package validator and the full test suite on every push and pull request. It uses CPU-only PyTorch and SQLite instead of Firestore.
- **Job logs:** every finished job logs one JSON line (`event: job_finished`, with queue wait, run time, status and machine). In Cloud Logging, filter on `jsonPayload.event="job_finished"`.
- **Health panel:** the Admin page's **Health** panel summarises the last 24 hours: queue wait and run time (median and p90), error rate, and each machine's jobs and busy time.
- **Staging:** `deploy/gcp.sh deploy staging` deploys `gymhub-board-staging`. It uses the same project and Firestore database (the free tier covers only the default database), but every collection is prefixed `staging-` and bundles live under `staging/` in the bundle bucket, so it never sees production data. IAP lets in only your account. After the first staging deploy, run `deploy/gcp.sh iap-client <client.json>` once more so staging gets the sign-in client. Staging jobs need a runner with the same prefix:
  ```
  GYMHUB_WORKER_TOKEN=<staging token> .venv/bin/python -m gymhub.runner --api https://<gymhub-board-staging-workers URL>
  ```
- **Auto-deploy:** `deploy/gcp.sh github-ci` (once) lets `.github/workflows/deploy.yml` deploy through Workload Identity Federation, with no key stored anywhere. Only workflows on this repository's `main` or on a tag `v*` may use it, and the deployer may only build and deploy the board: it cannot change buckets, secrets or who may sign in. Every push to main whose tests pass deploys staging; production is deployed when you push a tag `v*` whose commit passed the tests on main (`git tag v1.2.3 && git push origin v1.2.3`), from the Actions tab (**deploy**, then **Run workflow** with target `production`), or with `deploy/gcp.sh deploy` as before. If you ran `github-ci` before tag deploys existed, run it again so the deployer accepts tags. Until `github-ci` has run, the deploy workflow is skipped.
