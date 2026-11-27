"""network: island, a gang rank's own network namespace that reaches only its peers (#45 section 6.4).

`network: peers` gives a gang the host's network: the ranks find each other, but a container can then
reach anything the host can. `network: island` keeps them apart from everything but each other:

- **A holder container** owns the rank's network namespace (like a Kubernetes pause container). It
  sits on Docker's default bridge and publishes the gang's block of TCP ports on the machine's LAN
  address, 1:1. Its ephemeral port range is the upper part of that block, so ports that programs
  pick themselves (gloo pairs, NCCL bootstrap, Ray workers) are published too.
- **The machine's LAN address is local inside the namespace**, on a dummy interface `pool0` (or on
  `lo` when the kernel has no dummy driver). Programs bind to and advertise the machine's real
  address, as they would with host networking; incoming connections on the block are rewritten to
  it and outgoing ones from it are rewritten to the container's bridge address (then the host's
  usual masquerade). The host's own services are not reachable: that address is the namespace's own.
- **Firewall rules inside the namespace**, set by a short-lived helper container with NET_ADMIN
  that runs the host's own `ip` and `iptables` (chroot into a read-only view of `/`):
  in, only from the peers and loopback; out, only to the peers' blocks (and replies to them) and
  loopback. No DNS, no internet, no other machine on the LAN. IPv6 is switched off. A peer outside
  private address ranges is never let in: islands stay inside the LAN (docs/design/worker-v2.md §4),
  and the block is published only on the machine's LAN address.
- **The job container joins the holder's namespace** (`--network container:<holder>`) with every
  capability dropped, so it cannot change the rules.

Every rank of a gang uses the same block (ranks are on different machines), chosen from the gang id.
The helper uses a fixed small image chosen by the agent (never the job's image), because it runs
with NET_ADMIN and sees the host's files.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import shutil
import subprocess

from pool.peers import private_address

PORT_BASE = 41000
BLOCK = 64                      # ports per gang: [0, FIXED) for services, the rest ephemeral
FIXED = 24
BLOCKS = (61000 - PORT_BASE) // BLOCK
IMAGE = "busybox:1.36"
IFACE = "pool0"

# Ray's ports inside the block (offsets); workers use [8, FIXED).
RAY_OFFSETS = {"node-manager-port": 1, "object-manager-port": 2, "ray-client-server-port": 3,
               "dashboard-agent-grpc-port": 4, "dashboard-agent-listen-port": 5, "metrics-export-port": 6,
               "runtime-env-agent-port": 7}


def port_block(gang_id: str) -> tuple[int, int]:
    """The gang's ports, the same on every machine: [lo, hi], inclusive."""
    index = int(hashlib.sha256(gang_id.encode()).hexdigest()[:8], 16) % BLOCKS
    lo = PORT_BASE + index * BLOCK
    return lo, lo + BLOCK - 1


def ipv4(addr: str) -> str:
    """A plain IPv4 address, or ValueError: addresses end up in a shell script."""
    return str(ipaddress.IPv4Address(addr))


def ray_args(lo: int, node_ip: str) -> str:
    """`ray start` flags that keep every Ray port inside the block (the head's GCS port is lo)."""
    flags = [f"--node-ip-address={ipv4(node_ip)}", "--include-dashboard=false",
             f"--min-worker-port={lo + 8}", f"--max-worker-port={lo + FIXED - 1}"]
    flags += [f"--{name}={lo + offset}" for name, offset in RAY_OFFSETS.items()]
    return " ".join(flags)


def job_env(addr: str, lo: int, hi: int, iface: str = IFACE) -> dict:
    """What a rank needs to bind and advertise the machine's address. Without the dummy interface
    the address sits on lo, which gloo and NCCL cannot pick by name, so they are left to choose."""
    env = {"POOL_NODE_IP": ipv4(addr), "POOL_PORT_MIN": str(lo), "POOL_PORT_MAX": str(hi)}
    if iface == IFACE:
        env.update({"GLOO_SOCKET_IFNAME": IFACE, "NCCL_SOCKET_IFNAME": IFACE})
    return env


def holder_name(job: dict) -> str:
    return f"pool-{job['id']}-{job['tries']}-net"


def holder_command(job: dict, machine_id: str, addr: str, lo: int, hi: int, image: str = IMAGE) -> list[str]:
    return ["docker", "run", "-d", "--rm", "--name", holder_name(job),
            "--label", f"pool.machine={machine_id}", "--label", f"pool.job={job['id']}", "--label", "pool.role=island",
            "--network", "bridge", "--publish", f"{ipv4(addr)}:{lo}-{hi}:{lo}-{hi}/tcp",
            "--sysctl", f"net.ipv4.ip_local_port_range={lo + FIXED} {hi}",
            "--sysctl", "net.ipv6.conf.all.disable_ipv6=1",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--memory", "16m", "--pids-limit", "8", image, "sleep", "2147483647"]


def rules_script(addr: str, container_ip: str, lo: int, hi: int, peers: list[str]) -> str:
    """The commands run in the holder's namespace, with the host's `ip` and `iptables`."""
    addr, container_ip = ipv4(addr), ipv4(container_ip)
    peers = sorted({p for p in map(ipv4, peers) if private_address(p)} - {addr})
    ports = f"{int(lo)}:{int(hi)}"
    lines = [
        "set -e",
        f"if ip link add {IFACE} type dummy 2>/dev/null; then ip link set {IFACE} up; ip addr add {addr}/32 dev {IFACE};"
        f" else ip addr add {addr}/32 dev lo; echo iface=lo; fi",
        f"iptables -t nat -A PREROUTING -d {container_ip} -p tcp --dport {ports} -j DNAT --to-destination {addr}",
        f"iptables -t nat -A POSTROUTING -o eth0 -s {addr} -j SNAT --to-source {container_ip}",
        "iptables -A INPUT -i lo -j ACCEPT",
        *[f"iptables -A INPUT -s {peer} -p tcp -j ACCEPT" for peer in peers],
        "iptables -P INPUT DROP",
        "iptables -A OUTPUT -o lo -j ACCEPT",
        *[f"iptables -A OUTPUT -d {peer} -p tcp --dport {ports} -j ACCEPT" for peer in peers],
        *[f"iptables -A OUTPUT -d {peer} -p tcp -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT" for peer in peers],
        "iptables -P OUTPUT DROP",
        "iptables -P FORWARD DROP",
        "echo rules=ok",
    ]
    return "\n".join(lines)


def helper_command(holder: str, script: str, image: str = IMAGE) -> list[str]:
    return ["docker", "run", "--rm", "--network", f"container:{holder}", "--cap-drop", "ALL",
            "--cap-add", "NET_ADMIN", "--cap-add", "NET_RAW", "--cap-add", "SYS_CHROOT",
            "--security-opt", "no-new-privileges", "--mount", "type=bind,src=/,dst=/host,readonly",
            image, "chroot", "/host", "/bin/sh", "-c", script]


def available() -> tuple[bool, str]:
    """Whether this machine can make islands: Docker, and the host's ip and iptables. An agent in its
    own container (deploy/pool/) cannot see the host's tools: the helper container, which runs them
    from the host's root, is the check (a missing tool fails the gang with its error)."""
    tools = ("docker",) if os.environ.get("POOL_IN_CONTAINER") == "1" else ("docker", "iptables", "ip")
    for tool in tools:
        if not shutil.which(tool) and not shutil.which(tool, path="/usr/sbin:/sbin"):
            return False, f"no {tool} on this machine"
    return True, ""


class IslandError(RuntimeError):
    pass


def _run(command: list[str], timeout: float = 120) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise IslandError(f"{command[0]} {command[1]} failed: {(result.stderr or result.stdout).strip()[-400:]}")
    return result.stdout


def setup(job: dict, machine_id: str, addr: str, peers: list[str], lo: int, hi: int, image: str = IMAGE) -> tuple[str, str]:
    """Start the rank's holder and set its rules; return the holder's name and the interface that
    holds the machine's address (IFACE, or "lo" without a dummy driver). Removes it on failure."""
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        _run(["docker", "pull", "-q", image], timeout=600)
    name = holder_name(job)
    teardown(job)
    _run(holder_command(job, machine_id, addr, lo, hi, image))
    try:
        container_ip = _run(["docker", "inspect", "-f", '{{(index .NetworkSettings.Networks "bridge").IPAddress}}', name]).strip()
        out = _run(helper_command(name, rules_script(addr, container_ip, lo, hi, peers), image))
        if "rules=ok" not in out:
            raise IslandError(f"the island rules did not apply: {out.strip()[-200:]}")
    except Exception:
        teardown(job)
        raise
    return name, "lo" if "iface=lo" in out else IFACE


def teardown(job: dict) -> None:
    subprocess.run(["docker", "rm", "-f", holder_name(job)], capture_output=True, timeout=60)
