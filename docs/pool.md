# Compute pool

`pool` runs containers on machines that you and your friends own: register a machine with one command, then run a script on any of them and get its logs and output files back. It implements milestones M1 and M2 of [issue #45](https://github.com/ykirpichev/human-gym/issues/45), which builds towards MoE training, RL and batch inference on commodity machines. The package does not import gymhub, so it can later move to its own repository.

```
pool server --data ~/pool-data --port 8765            # prints a one-time admin code on first start
pool login http://<server>:8765 --code <code> --name ann
pool machine add --name spark                          # prints: curl -fsSL <server>/install.sh | sudo sh -s -- --code <code> --name spark
pool machine ls

pool-agent status | pause | resume                     # on the machine; sudo pool-agent uninstall removes it

cd my-project
pool run --gpus 1 --image nvcr.io/nvidia/pytorch:25.01-py3 --data ./data -- python train.py
pool ps | pool show <job> | pool logs -f <job> | pool get <job> | pool cancel <job>
pool submit job.yaml -f                                # the same from a spec file (see pool/spec.py)

pool model add qwen3.6-35b-a3b@fp8 ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/<rev>
pool run --gpus 1 --model base=qwen3.6-35b-a3b@fp8 --fit base:full:8 --flops 1e15 --dtype bf16 -- python serve.py
```

- **Adding a machine:** the installer (`pool/installer.py`, served at `/install.sh`) checks the OS, GPU and disk, installs Docker CE and the NVIDIA container toolkit if missing (Ubuntu/Debian), checks the agent package's sha256, unpacks it to `/opt/pool`, registers into `/etc/pool/agent.json` (0600), starts the agent in its own container, `pool-agent` (see [Docker](#docker-the-default)), as the user who ran sudo, and runs a smoke test container (no network, read-only root, GPU). Running it again repairs or upgrades. `--host-agent` runs the agent on the host's Python as the system unit `pool-agent.service` instead (the user then joins the `docker` group), and `--no-sudo` installs a user unit. The server refuses to hand out an installer over plain http on a public address.
- **Running a job:** `pool run` uploads the current folder as the job's code, mounted read-only at `/work`, the working directory. Each `--data` folder and `--model` is mounted read-only at `/inputs/<name>`. Whatever the job writes to `/outputs` (`$POOL_OUTPUTS`) comes back to `pool-outputs/<job>/`.
- **Artifacts and the cache:** folders travel as manifests of 64 MiB content-addressed chunks, and only the chunks the server is missing are uploaded, so sending an unchanged model again costs nothing. You only skip an upload for bytes you have uploaded yourself, so an upload never reveals what other members have stored. Each agent keeps the chunks and assembled folders it has fetched (`--cache-gb`, least recently used out first), so a model placed on a machine a second time is not downloaded again. The CLI remembers the hashes of files it has read.
- **Models and fits:** `pool model add` reads a Hugging Face folder's `config.json` and safetensors headers: parameters, active parameters for MoE models, bytes, layers, experts and top-k. Models are shared with the pool. `--fit base:full|offload|stage:k/n[:kv_gb]` makes placement check that the model fits:
  - `full`: all weights in GPU memory;
  - `offload`: active parameters on the GPU, the rest in RAM;
  - `stage:k/n`: k of n pipeline stages.

  On unified-memory machines such as the DGX Spark, GPU memory and RAM are counted once.
- **The network check:** give a job a cost hint (`--flops`, `--dtype`, `--bytes-out`). A machine is then refused when fetching the inputs it doesn't hold and sending the outputs back would take more than `alpha` (default 0.25) times the job's compute time at that machine's speed. `--bulk` skips the check. Among machines that pass, placement prefers fast machines that already hold the inputs. A job waits up to `affinity_wait_s` (default 60 s) for such a machine instead of going to a worse one.
- **Measurements:** agents measure their link to the server (round-trip time, download and upload speed) at start and every 6 hours. With `--bench-image <image with PyTorch>` they also measure TFLOP/s per dtype and host-to-GPU bandwidth once (`pool agent bench` redoes it). `--network-mbps` declares a link speed instead of measuring it, and `--limit-mbps` caps downloads.
- **Agent updates:** the server names its version of the agent in every heartbeat. `pool agent update` installs it on a machine (then restart the agent). `pool agent update --auto` also lets the agent update itself from then on: it waits until it runs no jobs, installs the new files and restarts in place. Auto-update is off by default, because it lets whoever runs the server change code that runs outside the sandbox. An agent running from a source checkout never overwrites itself. Each agent reports its own version (and whether auto-update is on, and its last failed self-update) in its profile: `pool machine ls` shows it in the AGENT column as the digest's first 12 characters and `current`, `outdated` or `unknown` (an agent too old to report it), `pool show` shows it for a job's machines, and `/metrics` has `pool_machine_agent_outdated`.
- **Why a job waits:** `pool show <job>` prints where a job ran and why: its score, the bytes it had to fetch, and the time to fetch them. For a queued job, it prints why each machine isn't running it: offline, busy, never fits (and why), link too slow, or over quota.
- **Pools and roles:** a server holds several pools, each with its own members, machines, models and settings. The first member is the server admin and creates pools (`pool pool create`). Pool admins invite people (`pool invite --pool P --role admin|member|viewer`) and change settings and quotas. Members run jobs and add machines; viewers only look. Use `pool use <pool>` to pick your default pool and `pool join --code` to join another.
- **Fair share and quotas:** the queue goes first to members with the least GPU time in the last 24 hours, then by `--priority`. Pool admins set quotas for the whole pool or per member:
  - `max_running_gpus`;
  - `max_running_jobs`;
  - `gpu_hours_per_day`;
  - `storage_gb`.

  For example, `pool pool set P quota.gpu_hours_per_day=10` or `pool member set bob --pool P quota.storage_gb=50`. Unpinned artifacts expire after `retention_days` (default 30) unless a queued or running job or a model uses them.
- **Use policies:** an owner decides who may use a machine, and when, with `pool machine policy`:
  - `--who pool|owner|ann,bob`;
  - `--hours 22:00-08:00` (local time);
  - `--idle-only`: only while nobody else uses the GPU;
  - owner-first ordering, on by default.

  When a machine stops being available, its running jobs get 30 s to stop and go back to the queue, without using up a retry.
- **The sandbox:** jobs run in Docker with no network, a read-only root filesystem, every capability dropped, `no-new-privileges`, an unprivileged uid, and limits on memory, CPUs, processes and `/tmp`. They get only the GPUs they were given. GPUs need the NVIDIA container toolkit. Job outputs are packed without following links, so a job cannot make the agent upload a file from the host. A machine's owner can see everything a job brings; the pool does not hide job data from them.
- **Bigger sandboxes:** `/dev/shm` defaults to 4 GB, `/tmp` to 8 GB and processes plus threads to 4,096. A job may ask for more with `--shm-gb`, `--tmp-gb` and `--pids` (`resources.shm_gb`, `tmp_gb`, `pids`), up to 256 GB, 512 GB and 65,536. Everything else stays locked down. Ray, vLLM and RL trainers such as verl run within these limits: see `examples/pool/rl-spike/`.
- **Machines that disappear:** leases last 90 s and are renewed by heartbeats. When a machine stops reporting, its jobs rerun elsewhere (`--retries`, default 1). A restarted agent stops the containers it left behind.
- **What the agent needs:** the agent and the CLI use only the Python standard library (3.10+), plus Docker. The server needs FastAPI and uvicorn. It keeps its state in SQLite under `--data`, or in Postgres with `--db postgresql://...` (needs `psycopg`), so several server processes can share one database. Members and machines authenticate with tokens from one-time codes, and agents make outbound connections only.
- **Upgrading the server:** the database schema is a list of numbered migrations (`pool/migrations.py`). The server applies the ones a database lacks when it starts and records them in `schema_version`; it refuses a database from a newer server. Databases made before migrations existed upgrade in place.
- **Network exposure:** only the outside network is untrusted; the LAN and pool members are trusted. Keep the server on the LAN: `pool server --host <LAN address>` (it warns at startup when it listens on a public address, or on every interface of a machine with a public one). It speaks plain HTTP, so if machines elsewhere must reach it, put it behind HTTPS: a reverse proxy such as Caddy or nginx with `--public-url https://...`, or a tunnel such as Tailscale. Never forward the port as plain HTTP. Agents open no ports to the outside: the peer server binds to the machine's LAN address and refuses connections from outside private ranges (RFC 1918, 100.64.0.0/10 for Tailscale, IPv6 ULA, link-local, loopback), and logs them; `network: island` publishes its ports only on the LAN address and lets in only the gang's peers.
- **Metrics:** `GET /metrics` serves Prometheus metrics: jobs by state, queue depth and the age of the oldest queued job, task queues, machines online, GPUs online and busy, heartbeat age per machine, GPU-hours per pool, and artifact and chunk bytes. It takes a server admin's token, or a scrape-only token set with `pool server --metrics-token` (or `POOL_METRICS_TOKEN`).

## Docker (the default)

Both halves of the pool run in containers: the **controller** (the pool server, plus the board's dispatcher) and every **worker** (the agent). Jobs were always containers; now the agent that starts them is one too, so a machine needs Docker (and the NVIDIA container toolkit for GPUs) and nothing else: no Python, no systemd unit. Running on the host stays possible: `install.sh --host-agent`, or `pool server` / `pool agent run` from a checkout.

| Image | What | Built from |
|---|---|---|
| `ghcr.io/ykirpichev/human-gym/pool` | pool server (`server`) and agent (`agent`, the default command) | `deploy/pool/Dockerfile` |
| `ghcr.io/ykirpichev/human-gym/dispatcher` | the board's dispatcher (`gymhub-runner --dispatch-only`) | `deploy/pool/Dockerfile.dispatcher` |

Both are multi-arch (amd64, arm64: x86 machines, DGX Spark and Jetson) and published by `.github/workflows/pool-images.yml` as `latest` and `<short sha>`. Add `--build` to the compose commands below to build them from your checkout instead.

**The controller** (one machine on the LAN; `deploy/pool/compose.controller.yml`):

```bash
cp deploy/pool/pool.env.example deploy/pool/.env        # set POOL_PUBLIC_URL=http://<LAN address>:8765
docker compose -f deploy/pool/compose.controller.yml up -d
docker compose -f deploy/pool/compose.controller.yml logs pool-server    # the first admin's one-time code
pool login http://<LAN address>:8765 --code <code> --name ann            # from your computer: you are the admin
pool invite --role member                                                # a code for the dispatcher
docker compose -f deploy/pool/compose.controller.yml run --rm dispatcher pool login http://pool-server:8765 --code <code> --name dispatcher
```

The server keeps its state in the `pool-data` volume (SQLite and chunks; `--db postgresql://...` works as before). The dispatcher waits, saying what it lacks, until it has a pool member and a board: set `GYMHUB_API` (the board's worker URL) and `GYMHUB_WORKER_TOKEN` (a worker token from the Compute page) in `deploy/pool/.env` and run `up -d` again. For a board on the same machine, `GYMHUB_STORE=sqlite:/state/board.db` instead. Skip the dispatcher (`up -d pool-server`) if you use the pool only from the `pool` CLI.

**A worker** (every machine with a GPU, the controller's too). The command from `pool machine add` (or the board's *Add a machine*) does it all:

```bash
curl -fsSL http://<server>:8765/install.sh | sudo sh -s -- --code <code>
sudo pool-agent status | pause | resume | logs -f | uninstall
```

Or with compose, on a machine that already has Docker and the NVIDIA container toolkit:

```bash
export POOL_SERVER=http://<server>:8765 POOL_CODE=<code>
docker compose -f deploy/pool/compose.agent.yml up -d                                                     # no GPU
docker compose -f deploy/pool/compose.agent.yml -f deploy/pool/compose.agent.gpu.yml up -d                 # x86 cards, DGX Spark
docker compose -f deploy/pool/compose.agent.yml -f deploy/pool/compose.agent.jetson.yml up -d              # Jetson (JetPack 6)
docker exec pool-agent pool-agent status
```

How the agent's container works:

- **Sibling containers.** The agent drives the host's Docker through the mounted `/var/run/docker.sock`, so jobs are the host's containers with the same sandbox as before. Access to that socket is root on the host: it is the same trust a host agent in the `docker` group had.
- **Same paths inside and out.** `/var/lib/pool` (the work folder, the input cache and the jobs' sockets, `POOL_SOCKET_DIR`) is mounted at the same path, because the host's Docker mounts job folders from the host. `/etc/pool/agent.json` is the same file a host install uses, so switching between the two keeps the machine's registration.
- **Not root.** The entrypoint (`deploy/pool/entrypoint.sh`) drops to `POOL_UID` (the installer passes the uid of the user who ran sudo; compose defaults to 10001) with the socket's group. Jobs run as that uid, as before.
- **Host network and PIDs.** Peers reach the agent's peer port, it reports the LAN address it reaches the server from, and it can tell the owner's own GPU processes from its jobs'.
- **GPUs.** On x86 and the Spark the agent gets the GPUs with `utility` only, to read them with `nvidia-smi`; jobs get theirs from the host's Docker as before (`--gpus device=N`). On a Jetson the NVIDIA runtime mounts the GPU's device nodes but no `nvidia-smi` (CSV mode), so the agent reports the integrated GPU from `/etc/nv_tegra_release` and `/dev/nvmap` (`pool/profile.py`), and labels the machine `gpu=tegra` as before.
- **Updates.** The installer mounts the server's own agent package (`/opt/pool`) over the image's copy, so a machine runs the agent its server ships. To update, run the installer again: it fetches the server's package and the latest image and restarts the container (jobs get their 90 s grace period and requeue). Auto-update (`pool agent update --auto`) does not apply to a container. With compose: `docker compose pull && docker compose up -d`.
- **Island networking** works unchanged: its helper container runs the host's own `ip` and `iptables`.

## Machines on one LAN: peers, islands and gangs (prototype)

The server is also the pool's directory. Each agent runs a small peer server (`--peer-port`, default 8766) and reports the address it reaches the server from. Agents then find each other through the server and talk directly:

- **Links and islands:** every agent measures its round-trip time and download speed from each peer within a minute of seeing it, and again every 6 hours. `pool machine links` lists them. Machines with at least 10 Gbit/s and under 1 ms between them form an island; a Wi‑Fi LAN gives one island per machine.
- **Chunks from peers:** before downloading a chunk from the server, an agent asks the server which online peers hold it, fastest link first. The server checks that the agent may read the artifact and returns a ticket per chunk, signed with the pool's peer key. The peer serves the chunk only against that ticket, and the receiver checks its hash, so a peer can't hand out wrong bytes. When no peer has it, or a peer fails, the chunk comes from the server.
- **Gangs:** `pool run --gang N` runs N ranks of a job, one per machine. Each rank waits on its machine until every rank has one (the rendezvous), then starts with `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `POOL_PEERS` and `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME`, so `torch.distributed` works unchanged. If a rank fails, the gang fails. If a rank's machine disappears, the whole gang goes back to the queue. A rank that waits 10 minutes for the others hands the gang back, without using a retry.
- **Runtimes:** a gang spec may set `runtime: torchrun` (the agent runs each rank under `torchrun` with the gang's rendezvous, one process per GPU) or `runtime: ray` (rank 0 starts a Ray head, waits for every rank, runs the command as the driver and stops Ray; the other ranks join and stay until it does). RL frameworks built on Ray (verl, OpenRLHF, SkyRL) then run unchanged inside one island. The image needs `torchrun` or `ray`.
- **Island networking:** `--network island` gives each rank of a gang its own network namespace. It reaches the other ranks and nothing else: no internet, no DNS, no other machine on the LAN, and none of its own host's services.
  - Every rank of a gang gets the same block of 64 TCP ports (`POOL_PORT_MIN`..`POOL_PORT_MAX`; `MASTER_PORT` is the first). The block is published 1:1 on the machine's LAN address, and ports that programs pick themselves come from it too, so gloo, NCCL and Ray work unchanged.
  - The machine's LAN address is the rank's own address inside the namespace (`POOL_NODE_IP`, on interface `pool0`), so programs bind to and advertise it as they would with host networking.
  - With `runtime: ray`, every Ray port is pinned inside the block.
  - Machines need Docker and the host's `ip` and `iptables`. No owner opt-in is needed, because a rank reaches less than with `peers`. A short-lived helper container sets the firewall rules (details in `pool/island.py`).
  - Ranks must be on different machines, which gangs already guarantee.
- **Peer networking:** `--network peers` gives a gang's containers the host's network, so the ranks reach each other. It runs only on machines whose owner allows it (`pool agent run --allow-peer-network`, remembered). Every other job still has no network at all. This is a prototype: the containers can reach anything the host can, not just their peers.
- **Mixed machines:** agents label themselves `arch` and `gpu` (`tegra`, `unified`, `discrete` or `none`). `--label gpu=tegra` places a job on a Jetson. `--image-for gpu=tegra=<image>` (`images` in a spec) swaps in a per-platform image, because a CUDA image built for a GB10 or an x86 card does not run on a Jetson.

```bash
pool agent run --allow-peer-network                    # on each machine, once
pool machine links
pool run --gang 2 --network peers --gpus 1 --mem 4 \
    --image nvcr.io/nvidia/vllm:26.04-py3 --image-for gpu=tegra=<jetson image> \
    --data corpus=./corpus -- python diloco.py --rounds 10 --round-seconds 30
```

`examples/pool/diloco/` trains one model on a DGX Spark and a Jetson over Wi‑Fi with DiLoCo, and its README has the measured results.

## Refs and queues

Long-running work coordinates through two server-side primitives (`pool/coord.py`):

- **Refs** are named, versioned pointers to artifacts, shared with the pool (`grpo/policy`). Publishing is compare-and-swap: it names the version it builds on and fails with 409 if someone published meanwhile (`--force` overrides). Every version is kept.
- **Queues** hold small tasks with inline JSON inputs, idempotent by key. Jobs lease tasks; the agent's heartbeat keeps the leases alive, and a lapsed lease puts the task back. A task that runs well past its queue's p90 is offered to a second worker, and the first result wins. A queue tied to a ref with `max_lag` requeues (or drops) results computed with a version too far behind.

```
pool ref publish grpo/policy ./lora-v0          # v1
pool queue create rollouts --ref grpo/policy --max-lag 2
pool queue add rollouts prompts.jsonl           # {"inputs": {...}, "key": "..."} per line
pool queue results rollouts -f                  # accepted results as JSON lines
```

A job has no network, so it reaches refs and queues through its agent: a unix socket at `$POOL_SOCKET`, and a stdlib helper module in `$POOL_LIB` (`pool/job.py`):

```python
import os, sys; sys.path.insert(0, os.environ["POOL_LIB"]); import pool_job
path, version = pool_job.fetch_ref("grpo/policy")       # under /refs, through the machine's cache
for task in pool_job.lease("rollouts", n=4, wait=10):
    pool_job.result("rollouts", task["id"], {"reward": 1.0}, computed_with=version)
pool_job.publish_ref("grpo/policy", "lora-v2", parent=version)  # a folder under /outputs
```

## Actors, checkpoints and subscriptions

An **actor** is a named group of identical long-lived jobs (replicas), typically workers that keep a model loaded and lease tasks from a queue (`pool/actors.py`). The server keeps it between `--min` and `--max` replicas: with `--queue`, about one replica per `--per-replica` waiting tasks; without, `--min`. A replica that ends or loses its machine is replaced; scaling down removes only replicas still waiting. Five replicas failing within a minute of starting mark the actor *failing*, and it stops relaunching until you scale or recreate it.

```
pool actor create rollers roller.yaml --min 1 --max 16 --queue rollouts --per-replica 4
pool actor ls
pool actor scale rollers --max 4
pool actor stop rollers
```

A job saves its `/checkpoint` folder with `pool_job.save_checkpoint()`, and the agent saves it when the job is preempted. When the job runs again (its machine was lost, or it was preempted), the agent restores the newest checkpoint first and sets `POOL_RESUMED=1`.

Small GPU jobs can share one GPU: `resources: {gpus: 1, share_gpu: true, gpu_memory_gb: 12}` reserves 12 GB of a GPU that other shared jobs may also use, while a job without `share_gpu` still gets a GPU to itself. That is how a trainer and its rollers fit on a single DGX Spark (`examples/pool/async-grpo`). `examples/pool/verl-grpo` runs the same recipe with verl's own fully-async trainer: a pool queue stands in for verl's message queue, a pool ref for its weight sync, and a pool actor of vLLM rollers for its rollouter.

`resources: {machines: own}` keeps a job (its code and inputs) on machines you own: other members' machines pass it over, and `pool show <job>` says why it waits. `resources: {machine: spark}` pins a job to one machine, by name (with `machines: own`, only if it is yours).

A spec may list `subscribe: [ref names]`. When a new version of such a ref is published, the agent on every machine running a subscribed job fetches it into its cache right away (from peers when they have it), so `fetch_ref` is a local copy.

## Rounds

A **round** synchronises workers on a ref without them needing a link to each other (`pool/rounds.py`), so it can span islands. It opens on the ref's current version; workers train from it and contribute a folder with a weight (`pool_job.contribute`). The round is ready once `--quorum` contributions are in plus `--grace` seconds, or at `--deadline` with at least `--min`. A reducer job claims it (`pool_job.claim_round`), merges the contributions (`pool_reducers.weighted_mean` averages `.npy` and `.safetensors` files) and finishes it (`pool_job.finish_round`), which publishes the ref's next version on the round's base; with `--repeat` the next round opens. Late work is dropped, or with `--late next` carried into the next round with its weight scaled by `--decay`.

```
pool round open diloco/weights --quorum 3 --min 2 --deadline 300 --late next --repeat
pool round status diloco/weights
```

A round that spans islands can reduce **island first** (`--reduce tree`; `flat`, one reducer for every contribution, stays the default). Its contributions stay on the machines that made them (held: the server gets only the manifest), and when the round is ready the server groups them by island and plans one stage per island: the machine of the island that collects the others' updates fastest over the measured links merges them into a partial, fetching them machine to machine over the peer swarm; an island with one contribution forwards it. The server sets each partial's weight to its island's total (late work times `--decay`), so the final reducer's weighted mean of the partials equals the flat one, and only one partial per island crosses the slow link instead of one update per member. Workers run `pool_reducers.reduce_round(ref)` after contributing, which claims whatever stage their machine may take (the planned machine at once, another of its island after `--handoff` seconds, any machine after twice that); a reducer whose job stops or whose machine goes offline hands its stage on, and an island whose partial is not in by the deadline counts its contributions as missing against `--min`. With `robust=True` each island stage checks its members before forming the partial and reports outliers; the final stage sees every partial's inputs and can recompute a share of them (`verify`). Not for gossip or torchft rounds, which reduce their own way.

```
pool round open diloco/weights --quorum 16 --min 12 --deadline 600 --reduce tree --repeat
```

Workers that do have a link to each other, however slow (islands of one pool, `--network peers`), can reduce in-band with **torchft** instead (`--reducer torchft --lighthouse http://host:port`). `pool_reducers.TorchftRounds` does it from a job: the first job to open the round hosts torchft's lighthouse and writes its address into the round, every job gets a `torchft.Manager` pointed at it (the round's min is torchft's `min_replica_size`, its deadline the quorum timeout), torchft's DiLoCo all-reduces the pseudo-gradients directly between the islands, and each outer step is recorded as a pool round: who contributed with what weight, published as the ref's next version. `examples/pool/hierarchical-diloco/` is Demo C1 in small with it.

## The checked tier

For a pool wider than one team, `pool pool set default trust=checked` turns on checks (`pool/trust.py`). Verifiers report on machines: a queue created with `--replicate 0.05` runs that share of its tasks on two machines and accepts a result only when they agree (a third breaks a tie); a trainer or reducer reports its own checks with `pool_job.report(machine, ok, kind, detail)`, for example rewards or logprobs it recomputed (`examples/pool/async-grpo/trainer.py` does both), or updates that `pool_reducers.robust_mean` dropped as outliers. After `exclude_after` failed checks (default 2) a machine is excluded: it gets no more tasks, its results and round contributions are refused, and `pool machine ls` shows it as excluded until an admin runs `pool machine trust NAME`.

## Board jobs: private until you share

When the Human Gym board grades on the pool, its dispatcher submits every board job as one pool member, the board owner's. Whose board jobs reach your machines is decided on the board, not here: the dispatcher takes only its owner's jobs until the owner shares it (**Share the pool…** on the board's Compute page, by email or with everyone; see docs/self-hosting.md). Jobs for a gym that is not public stay on the dispatcher member's own machines (`resources.machines: own`) even then. To keep a machine to your own pool jobs as well, run `pool machine policy NAME --who owner`.

Not yet: iroh-blobs (M5), and signed URLs for a cloud blob store.
