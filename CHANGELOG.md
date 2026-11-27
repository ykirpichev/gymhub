# Changelog

All notable changes to Human Gym are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Until 1.0, minor versions may include breaking changes to the job, store and package formats; they will be called out under **Changed**.

## [Unreleased]

### Added

- **Board:** sign-in without Google for a group: `gymhub-board --lan` (or `GYMHUB_AUTH=local`) gives people a name and a password kept by the board, with accounts created only from invite links, password reset links from the Admin page, and the board listening on a LAN or Tailscale address. docs/self-hosting.md, "Host it for a group, without Google", also covers `ssh -L` and Tailscale for a headless GPU box.
- **Pool:** Docker is the default way to run both halves of the pool: images for the pool server and agent (`pool`) and the board's dispatcher (`dispatcher`), compose files for the controller and for workers (CPU, x86/DGX Spark GPUs, Jetson), and `install.sh` now runs the agent in its own container (`--host-agent` keeps the systemd unit). See docs/pool.md, "Docker".
- **Board:** *Try in browser* runs a Python exercise's example cases on the code in the editor in the browser itself (Pyodide), with no runner and nothing recorded; Submit still grades. Workers publish the examples (inputs and expected outputs, never graders) with the gym.
- **Moderation:** members' gyms need an admin's approval before they go public (on by default; *Access settings* turns it off), and anyone can report a gym. Admins hide, remove or dismiss reported gyms from the Admin page.
- **Board:** a landing page for signed-out visitors under the sign-in card: the board's public gyms with exercise counts, how it works, and a link to the source.

### Changed

- **Compute:** machines are private by default, an admin's machines and pool bridge too: they run only their owner's jobs until shared. *Share the pool…* on the Compute page (or *Share…* next to a machine) shares with people by email or with everyone. Other people's jobs wait with a note saying why. Boards where members ran code on the admin's pool need the admin to share it once.

## [0.1.0] - unreleased

The first public release.

### Added

- **Gyms:** GPU kernel exercises in CUDA, PyTorch and Triton (`gpu-gym`), with tests, benchmarks and editor autocomplete; algorithm, LLM, ML coding and ML fundamentals gyms on the shared `gymkit` grader runtime (Python and C++).
- **Gym Markdown:** a whole gym in one `.gym.md` file, written by people or agents, compiled into a package with a public/private split; `gymhub.cli gym new`, `check`, `validate` and `export`.
- **Board:** a web board to browse gyms, edit and submit solutions, see progress and benchmarks; works on phones; private gyms and sharing; gym upload with validation on a worker before publishing. Runs locally on SQLite or on Cloud Run with Firestore, with IAP, in-app Google sign-in, or no auth for local use.
- **Runner:** `gymhub-runner` leases jobs from the board through a worker-token API; a dispatch-only mode sends every job to the compute pool; one-command worker installer with Docker, the NVIDIA container toolkit and a system service.
- **Compute pool:** `pool` server and agents that run jobs in Docker on machines you register: fair-share scheduling by GPU-hours, model-fit and network-cost placement, content-addressed artifact cache, gangs (torchrun, Ray), refs, queues, actors and rounds, island networking, trust checks with replicated tasks, schema migrations on SQLite or Postgres, Prometheus `/metrics`, and agent self-update (opt-in).
- **Security:** other users' code is graded only on the pool, in self-contained grader images with no network, a read-only root filesystem and every capability dropped. See [SECURITY.md](SECURITY.md).
- CONTRIBUTING.md, SECURITY.md and this changelog.

[Unreleased]: https://github.com/ykirpichev/human-gym/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ykirpichev/human-gym/releases/tag/v0.1.0
