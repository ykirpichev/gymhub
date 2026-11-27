# Worker v2: one agent per host, installed by one command

Status: proposal (2026-11-24). Tracking issue: #137.
Builds on `docs/design/platform-v2.md` §3 (workers) and #45 (the compute pool).

## 0. Summary

Today a host runs one of two daemons: the gymhub runner or the pool agent. Each has its own sandbox
and its own installer, and both installers assume Docker, the NVIDIA container toolkit, Python and
systemd lingering are already set up.

Worker v2:

- **One agent per host** (the pool agent). gymhub reaches hosts only through the pool.
- **Docker** (Docker CE + NVIDIA Container Toolkit) runs every job. The installer installs both when
  they are missing.
- **One command** to add a machine, and one to remove it.
- **Nothing on the hosts or the pool server is reachable from outside the network.**

## 1. Threat model

- **Trusted:** everyone on the local network, the machine owners, and the people they give pool
  access to. No protection between them beyond what Docker already gives a job. Isolation between
  users on the LAN is out of scope.
- **Untrusted:** the internet. Concretely:
  - nobody outside the LAN can connect to an agent, its peer port, or the pool server;
  - a job cannot open connections out of the machine (`--network none`, as today), so code that
    arrives through the board cannot be used to reach or attack anything;
  - what a host downloads from outside (the installer, the agent, images) comes over HTTPS and is
    checked against a checksum or digest.

Out of scope: gVisor, separate users for the agent and for jobs, trust tiers, image signing, GPU
memory wiping. These are worth adding again only if machines start running code from people the
owner does not know.

## 2. Host layout

```
 board (Cloud Run) ◀── HTTPS, outbound ── gymhub dispatcher ── pool server (LAN only)
                                                                  ▲ HTTP(S), outbound
 ┌──────────────── host (LAN) ───────────────────────────────────┼────────┐
 │  pool-agent  (system service, in the docker group)  ───────────┘        │
 │     │ docker run --network none --read-only --cap-drop ALL               │
 │     │   --gpus device=<leased> --memory … --pids-limit …                 │
 │     ▼                                                                    │
 │  job container  (grader image, gym slice + submission mounted)          │
 └─────────────────────────────────────────────────────────────────────────┘
```

- The agent runs as a **system** systemd unit (`User=<owner>` or a `pool` user, in the `docker`
  group), so it survives logout and reboot without `loginctl enable-linger`.
- Containers keep today's hardening (`pool/agent.py` `DockerRuntime`): no network, read-only root,
  every capability dropped, `no-new-privileges`, memory/CPU/pids/`/tmp` limits. One fix: only the
  leased GPUs (`--gpus device=N`), never `all` (the gymhub sandbox passes `all` today).
- Gangs keep `network: island` and `network: peers`. Both stay inside the LAN (§4).

## 3. Installation

Shown by `pool machine add` and by the board's "Add a machine" button:

```
curl -fsSL https://<server>/install.sh | sudo sh -s -- --code ABCD-1234
```

Steps, one line of output each:

1. **Preflight:** OS, arch, GPU and driver, Jetson or not, disk free. Stops with a clear message on
   anything unsupported (for example, no NVIDIA driver).
2. **Docker:** installs Docker CE from Docker's apt repository if there is no Docker. Installs
   `nvidia-container-toolkit` from NVIDIA's repository and configures Docker to use it.
3. **Agent:** downloads the agent package from the server and checks its sha256 (the server prints
   it with the install command). Unpacks it to `/opt/pool`.
4. **Register** with the code, and write `/etc/pool/agent.json` (0600).
5. **Service:** `pool-agent.service`, a system unit, enabled and started.
6. **Smoke test:** runs a tiny container with the leased GPU and no network, and checks three things:
   the GPU works, the network is unreachable, and the root filesystem is read-only. Prints the result.
7. **Warm up** (in the background): pulls the grader image for this arch.
8. **Prints** the machine's name, where to see it, and how to pause or remove it.

The script is idempotent: running it again repairs or upgrades. Owner commands:

```
pool-agent status | pause | resume | uninstall
```

`uninstall` stops the service, unregisters the machine, and removes the unit, `/opt/pool` and
`/etc/pool`. It leaves Docker installed.

Supported first: Ubuntu 22.04/24.04 and Debian 12 on x86_64, the DGX Spark, and Jetson (JetPack 6/7).
A `--no-sudo` mode keeps today's user-unit install for machines without sudo.

## 4. Network exposure

- **Agents** open no ports to the outside. They connect out to the pool server. The peer port
  (`pool/peers.py`) binds to the LAN address only and refuses connections from outside private
  address ranges (RFC 1918 and ULA). Gang ports published by `network: island` do the same.
- **Pool server** binds to the LAN by default. If it has to be reachable from outside (machines in
  other places), it goes behind HTTPS (a reverse proxy or a tunnel such as Tailscale), and the
  install command is given only with an `https://` URL.
- **Board** (Cloud Run) talks to the LAN only through the gymhub dispatcher's outbound connections.
  No inbound path to the LAN.
- **Jobs:** `--network none`. `island` reaches only the gang's peers; `peers` uses the host network,
  which is acceptable on a trusted LAN, but such jobs are allowed only from pool members, never from
  board-submitted gym code.

## 5. Images

Done: the host-mount sandbox is removed, and other users' code runs only on the pool. Before that,
`gymhub/runner/sandbox.py` mounted the host's `.venv` and `/usr/local/cuda` into an image built on
the host. That image had to match the host's Ubuntu and Python, and it was the main reason a new
machine needed a repo checkout.

Instead, CI builds **self-contained grader images** with the grader code inside and pushes them by
digest:

- `grader:<ver>-cpu`: Python, numpy, g++;
- `grader:<ver>-cuda`: amd64 and arm64 (Spark), with torch, triton, nvcc;
- `grader:<ver>-l4t-r36`: Jetson.

The pool picks the image from the machine's profile (arch, Jetson, CUDA major), replacing
`GYMHUB_POOL_IMAGES`. Only the gym slice (without answers) and the submission are mounted.

## 6. gymhub on the pool

- The gymhub runner becomes a **dispatcher**: it leases board jobs and submits pool jobs
  (`gymhub/runner/pooled.py`, #88). It never executes code, so it can run on the pool server.
- The board's Compute page lists pool machines. "Add a machine" gives the install command.
- Old runners keep working until they are switched. `deploy/install-runner.sh` prints a deprecation
  notice.

## 7. Rollout

1. Images (§5). They unblock installs with no repo checkout.
2. Installer (§3).
3. Network exposure (§4).
4. gymhub on the pool (§6).

## Issues

| # | Issue | Section |
|---|---|---|
| #132 | Self-contained grader images per arch (no host venv or CUDA mounts) | §5 |
| #133 | One-command installer: Docker, NVIDIA toolkit, system service, smoke test, uninstall | §3 |
| #128 | Nothing reachable from outside the LAN; jobs only see their leased GPUs | §2, §4 |
| #135 | gymhub uses pool agents only | §6 |

Closed as out of scope for this threat model: #127, #129, #130 (folded into #133), #131, #134.
Related: #45, #52, #88.
