# Security policy

Human Gym runs code that other people wrote: every submission to a gym is a program, and the compute pool runs containers on machines that their owners register. Security reports are taken seriously and are welcome.

## Reporting a vulnerability

**Please do not open a public issue, pull request or discussion for a security problem.**

Report it privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** ([direct link](https://github.com/ykirpichev/human-gym/security/advisories/new)). Only the maintainers can see the report.

Please include:

- what an attacker can do, and what they need first (an account on a board, a pool member token, a machine in the pool, an uploaded gym, ...);
- the steps or a proof of concept, and the commit or version you tested;
- any fix you have in mind.

What to expect:

- an acknowledgement within 3 working days;
- an assessment, and a plan or a reason for declining, within 14 days;
- a fix released as soon as it is ready, with an advisory that credits you unless you ask not to be named.

Please give us a reasonable time to fix the problem before you disclose it publicly, and do not test against boards, pools or machines that you do not run yourself.

## Supported versions

Human Gym is before 1.0. Fixes land on `main` and in the next release; older releases are not patched. Run the latest release, or `main`.

## Security model

Knowing what Human Gym is meant to protect helps you tell a vulnerability from expected behaviour. The README and `docs/` describe each part in more detail.

### What should hold

- **The board never runs user code.** The web board (`gymhub/board/`) only parses uploads, including `.gym.md` files, and writes job documents to the store. A way to make the board execute submitted code is a vulnerability.
- **Other people's code runs only in the pool sandbox.** A runner grades its owner's own jobs on its machine. Other users' code, and every job of an uploaded gym, is graded on the compute pool in a grader image with no network, a read-only root filesystem, every capability dropped, `no-new-privileges`, an unprivileged user, limits on memory, CPUs, processes and `/tmp`, and only the GPUs the job was given. A job that escapes this, reaches the network, reads the host's files or another job's data, or gets the runner to run it outside the pool is a vulnerability.
- **Jobs do not get credentials.** Runners start jobs without their credential variables (`GOOGLE_*`, `AWS_*`, tokens, secrets, passwords, the SSH agent) and keep the worker token out of reach of `/proc`.
- **Answers stay private.** Learners and the board see only the public parts of a gym; reference solutions and graders go only to machines that grade. Pool jobs get a slice of the gym without answer keys. Getting a gym's private parts through the board or a job's output is a vulnerability.
- **Private gyms stay private.** A job of a non-public gym runs only on pool machines owned by the pool member who submits it.
- **Access control holds.** Pool roles (admin, member, viewer), quotas, one-time machine codes, worker tokens, and the board's sign-in (`iap` or `app`) are meant to be enforced. Bypassing them is a vulnerability.
- **Pool agents open no ports to the outside.** Agents connect out to the server; the peer server accepts only private-range addresses, and chunks are served only against tickets signed by the server.
- **Deduplication reveals nothing.** The artifact store deduplicates per uploader, so one member cannot learn what another stored.

### Trusted by design (not vulnerabilities)

- **A machine's owner sees everything a job brings** to their machine. The pool does not hide job data from the machines that run it. Only submit to pools whose machines you trust with your data.
- **The pool's LAN and its members are trusted.** The pool server speaks plain HTTP and is meant to stay on a LAN or behind HTTPS (a reverse proxy or a tunnel such as Tailscale). Exposing it to the internet over plain HTTP is a misconfiguration.
- **`--network peers` is a prototype.** Gang containers with it can reach anything the host can. It runs only on machines whose owner opted in with `--allow-peer-network`. `--network island` is the isolated option.
- **Agent auto-update trusts the pool server.** `pool agent update --auto` (off by default) lets whoever runs the server change code that runs outside the sandbox on your machine.
- **Development switches turn protections off.** `GYMHUB_AUTH=dev` disables sign-in and makes every request an admin; `GYMHUB_TRUST_JOBS=1` runs isolated jobs on the runner itself as your user; a runner in direct mode (`GYMHUB_STORE=firestore:...`) can read and write the whole database. They are for local development and tests only.
- **Your own jobs run as you.** The owner's own submissions are graded on the owner's runner without a sandbox, as they would be if run by hand.
- **Benchmarks from members' own machines** can report anything, so they never appear on leaderboards; only results from trusted machines (a direct-mode runner, or a machine of the gym's owner or an admin) count.

If you are not sure which side of this line a problem falls on, report it privately anyway.
