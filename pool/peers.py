"""Machine-to-machine traffic inside a pool (#45 sections 5 and 6.4). Standard library only.

The server is the directory: every agent reports the address it reaches the server from (its LAN
address on a home or lab network) and the port of its small peer server. Agents then talk to each
other directly, without the server in the path:

  - link measurement: each agent times a transfer from every peer it can reach and reports
    {peer: Mbit/s, round trip}. Machines with >= 10 Gbit/s and < 1 ms between them form an island.
  - peer chunks: the server is the chunk tracker. Agents report the chunks they hold (with each
    heartbeat, and each chunk as soon as it lands: POST /v1/agent/chunks). Before downloading, an
    agent asks the server which online peers hold the chunks it needs; the server checks that the
    agent may read the artifact, offers holders only for that artifact's chunks (so the tracker
    never says what else a machine holds) and hands back a ticket per chunk. The peer serves the
    chunk only against a valid ticket. Swarm then fetches from all holders at once, rarest chunk
    first, and from the server what nobody offers, asking the tracker again while it works.
    Chunks are checked against their hash on arrival, so a peer cannot hand out wrong bytes.
  - transports: plain HTTP between peers (PeerServer), or, with the agent's --iroh, the iroh-blobs
    sidecar (IrohSidecar, pool/sidecar/iroh): QUIC with hole punching and relays, the same tickets,
    and HTTP as the fallback.
  - gangs: the jobs of a gang find each other through the server (rendezvous) and then talk
    directly, for example torch.distributed over gloo or NCCL.

Peer requests carry an HMAC made with the pool's peer key, which the server gives only to the
pool's agents. Peer traffic is plain HTTP on the LAN; integrity comes from content hashes, and
the bytes are the same chunks the server would send.

Only the outside network is untrusted (docs/design/worker-v2.md section 4): the peer server binds to
the machine's LAN address and refuses connections from addresses outside private ranges.
"""

import hashlib
import hmac
import ipaddress
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PORT = 8766
TICKET_SECONDS = 15 * 60
CLOCK_SKEW = 300
ISLAND_MBPS = 10_000
ISLAND_RTT_MS = 1.0
SPEEDTEST_MAX = 64 << 20
_BLOCK = os.urandom(1 << 20)


# ---- signatures ------------------------------------------------------------------------------------


def sign(key: str, *parts: str) -> str:
    return hmac.new(bytes.fromhex(key), ":".join(parts).encode(), hashlib.sha256).hexdigest()


def ticket(key: str, machine: str, sha: str, expires: float | None = None) -> str:
    """What the server hands an agent to fetch chunk `sha` from a peer: machine:expires:mac."""
    expires = int(expires or time.time() + TICKET_SECONDS)
    return f"{machine}:{expires}:{sign(key, 'chunk', machine, sha, str(expires))}"


def check_ticket(key: str, value: str, sha: str) -> bool:
    try:
        machine, expires, mac = value.split(":")
        return int(expires) >= time.time() and hmac.compare_digest(mac, sign(key, "chunk", machine, sha, expires))
    except (ValueError, TypeError):
        return False


def peer_header(key: str, machine: str) -> str:
    """Proof that the caller is an agent of this pool (for speed tests)."""
    stamp = str(int(time.time()))
    return f"{machine}:{stamp}:{sign(key, 'peer', machine, stamp)}"


def check_peer_header(key: str, value: str) -> str | None:
    try:
        machine, stamp, mac = value.split(":")
        if abs(time.time() - int(stamp)) > CLOCK_SKEW:
            return None
        return machine if hmac.compare_digest(mac, sign(key, "peer", machine, stamp)) else None
    except (ValueError, TypeError):
        return None


# ---- addresses -------------------------------------------------------------------------------------


def _route(host: str, port: int) -> tuple[str, str] | None:
    """(source, destination) of the route to host:port."""
    try:
        family, _, _, _, addr = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)[0]
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.connect(addr[:2])  # no packet is sent: this only picks the route
            return probe.getsockname()[0], addr[0]
    except OSError:
        return None


def _route_source(host: str, port: int) -> str | None:
    found = _route(host, port)
    return found[0] if found else None


def local_address(server_url: str) -> str | None:
    """The address this machine uses to reach the server: on a LAN, its LAN address. An agent on the
    server's own machine reaches it over one of its own addresses (loopback, or whichever address its
    name resolves to: mDNS may answer with a Docker bridge's), so it advertises its default-route
    address instead."""
    parsed = urllib.parse.urlparse(server_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    found = _route(parsed.hostname or "localhost", port)
    if found and not found[0].startswith("127.") and found[0] != "::1" and found[0] != found[1]:
        return found[0]
    return _route_source("192.0.2.1", 9) or (found[0] if found else None)  # TEST-NET-1: picks the default route


_CGNAT = ipaddress.ip_network("100.64.0.0/10")  # carrier-grade NAT, also Tailscale's addresses


def private_address(address: str | None) -> bool:
    """Whether `address` is inside the LAN: RFC 1918, loopback, link-local, IPv6 ULA (fc00::/7), or
    100.64.0.0/10 (Tailscale). Anything else, or something that is not an address, is outside."""
    try:
        ip = ipaddress.ip_address((address or "").split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_unspecified or ip.is_multicast:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or (ip.version == 4 and ip in _CGNAT)


def bind_address(address: str | None) -> str:
    """Where the peer server listens: the machine's LAN address when it has a usable IPv4 one, so
    the port is not open on its other interfaces; otherwise every interface (connections from
    outside private ranges are still refused)."""
    try:
        return str(ipaddress.IPv4Address(address or ""))
    except ValueError:
        return "0.0.0.0"


def interface_of(address: str | None) -> str | None:
    """The network interface that holds `address` (gloo and NCCL want its name), on Linux."""
    if not address:
        return None
    try:
        out = subprocess.run(["ip", "-o", "addr", "show"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[3].split("/")[0] == address:
            return parts[1]
    return None


def islands(machines: list[str], links: list[dict]) -> list[list[str]]:
    """Group machines whose links, both ways measured or one way, are island-grade. A machine with
    no such link is an island of one."""
    parent = {m: m for m in machines}

    def root(m: str) -> str:
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    for link in links:
        a, b = link["src"], link["dst"]
        if a in parent and b in parent and link["mbps"] >= ISLAND_MBPS and link["rtt_ms"] < ISLAND_RTT_MS:
            parent[root(a)] = root(b)
    groups: dict[str, list[str]] = {}
    for m in machines:
        groups.setdefault(root(m), []).append(m)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


# ---- the peer server ---------------------------------------------------------------------------------


class Pace:
    """One byte rate shared by everything the peer server sends: the owner's cap on what peers may
    take from this machine's uplink (--peer-limit-mbps)."""

    def __init__(self, mbps: float):
        self.rate = mbps * 1e6 / 8
        self.lock = threading.Lock()
        self.until = time.monotonic()

    def take(self, n: int) -> None:
        with self.lock:
            now = time.monotonic()
            self.until = max(self.until, now) + n / self.rate
            wait = self.until - now
        if wait > 0:
            time.sleep(wait)


class PeerServer:
    """A small HTTP server on the agent: chunks it holds (against a ticket) and speed tests.

    `key()` returns the pool's current peer key (None until the server sent it); `read_chunk(sha)`
    returns the chunk's bytes or None; `has_chunk(sha)` says whether it is here (HEAD).
    Connections from outside private address ranges are closed before a request is read, and logged
    with `log(message)`."""

    def __init__(self, port: int, key, read_chunk, host: str = "0.0.0.0", has_chunk=None, limit_mbps: float | None = None,
                 log=None):
        self.key = key
        self.log = log or (lambda message: print(time.strftime("%H:%M:%S"), message, file=sys.stderr, flush=True))
        self.refused = 0
        self.pace = Pace(limit_mbps) if limit_mbps else None
        self.read_chunk = read_chunk
        self.has_chunk = has_chunk or (lambda sha: read_chunk(sha) is not None)
        self.served_bytes = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                pass

            def _reply(self, status: int, body: bytes = b"", kind: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                view = memoryview(body)
                for start in range(0, len(body), 1 << 20):
                    piece = view[start:start + (1 << 20)]
                    if outer.pace is not None:
                        outer.pace.take(len(piece))
                    self.wfile.write(piece)

            def do_GET(self):
                key = outer.key()
                url = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(url.query)
                if url.path == "/peer/healthz":
                    return self._reply(200, b'{"ok":true}')
                if key is None:
                    return self._reply(503, b'{"detail":"not ready"}')
                if url.path == "/peer/speedtest":
                    if not check_peer_header(key, self.headers.get("X-Pool-Peer", "")):
                        return self._reply(403, b'{"detail":"not a peer of this pool"}')
                    size = max(1, min(int((query.get("bytes") or ["0"])[0] or 0), SPEEDTEST_MAX))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    left = size
                    while left:
                        piece = _BLOCK[: min(left, len(_BLOCK))]
                        if outer.pace is not None:
                            outer.pace.take(len(piece))
                        self.wfile.write(piece)
                        left -= len(piece)
                    return None
                if url.path.startswith("/peer/chunks/"):
                    sha = url.path.rsplit("/", 1)[-1]
                    if not check_ticket(key, self.headers.get("X-Pool-Ticket", ""), sha):
                        return self._reply(403, b'{"detail":"no valid ticket for this chunk"}')
                    data = outer.read_chunk(sha)
                    if data is None:
                        return self._reply(404, b'{"detail":"not here"}')
                    outer.served_bytes += len(data)
                    return self._reply(200, data, "application/octet-stream")
                return self._reply(404, b'{"detail":"no such path"}')

            def do_HEAD(self):
                """Whether a ticket is good for a chunk held here, without the bytes (the iroh
                sidecar checks every fetcher's ticket this way before serving a blob it holds)."""
                key = outer.key()
                url = urllib.parse.urlparse(self.path)
                status = 404
                if key is None:
                    status = 503
                elif url.path.startswith("/peer/chunks/"):
                    sha = url.path.rsplit("/", 1)[-1]
                    if not check_ticket(key, self.headers.get("X-Pool-Ticket", ""), sha):
                        status = 403
                    elif outer.has_chunk(sha):
                        status = 200
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

        class Server(ThreadingHTTPServer):
            def verify_request(self, request, client_address) -> bool:
                return outer.allowed(client_address[0])

        self.httpd = Server((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="peer-server")

    def allowed(self, source: str) -> bool:
        """Whether a connection from `source` may talk to this server: only from inside the LAN."""
        if private_address(source):
            return True
        self.refused += 1
        self.log(f"peer server: refused a connection from {source} (not a private address)")
        return False

    def start(self) -> "PeerServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ---- the client side ----------------------------------------------------------------------------------


def measure(peer: dict, key: str, machine: str, size: int = 16 << 20, timeout: float = 30) -> dict:
    """Round trip and download rate from one peer: {"rtt_ms", "mbps"}. Raises OSError when unreachable."""
    base = f"http://{peer['addr']}:{peer['port']}"
    rtt = []
    for _ in range(3):
        start = time.monotonic()
        with urllib.request.urlopen(base + "/peer/healthz", timeout=5) as reply:
            reply.read()
        rtt.append((time.monotonic() - start) * 1000)
    request = urllib.request.Request(f"{base}/peer/speedtest?bytes={size}", headers={"X-Pool-Peer": peer_header(key, machine)})
    start = time.monotonic()
    got = 0
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        while True:
            block = reply.read(1 << 20)
            if not block:
                break
            got += len(block)
    seconds = max(time.monotonic() - start, 1e-3)
    return {"rtt_ms": round(min(rtt), 2), "mbps": round(got * 8 / 1e6 / seconds, 1)}


def fetch_chunk(peer: dict, sha: str, ticket_value: str, dest: Path, timeout: float = 600) -> int:
    """Download one chunk from a peer into `dest`, checking its hash. Raises OSError or ValueError."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.{threading.get_ident()}.peer.part")
    request = urllib.request.Request(f"http://{peer['addr']}:{peer['port']}/peer/chunks/{sha}",
                                     headers={"X-Pool-Ticket": ticket_value})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply, open(tmp, "wb") as handle:
            while True:
                block = reply.read(1 << 20)
                if not block:
                    break
                digest.update(block)
                handle.write(block)
                size += len(block)
        if digest.hexdigest() != sha:
            raise ValueError(f"chunk {sha[:12]} from {peer.get('name', peer['addr'])} arrived corrupt")
        os.chmod(tmp, 0o444)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return size


# ---- the swarm: chunks from several peers at once ---------------------------------------------------

PER_PEER = 2  # chunks in flight from one peer: enough to fill its link, not enough to swamp it
PEER_FAILURES = 3  # a peer that fails this many chunks in a row is left out for the rest of the artifact
REFRESH_SECONDS = 2.0  # how often to ask the tracker again for chunks no peer offered


class Swarm:
    """Fetch an artifact's missing chunks from every peer that holds them, in parallel, like a
    torrent, and from the server what no peer offers.

    `sizes` {sha: bytes} are the chunks to fetch; `offers` {sha: {"ticket", "peers"}} come from the
    server's tracker (POST /v1/agent/peer-sources), peers best link first. Placement:
      - rarest first, ties broken at random, so machines fetching the same artifact at the same time
        start on different chunks and can trade them;
      - each chunk goes to the least busy peer that holds it, at most `per_peer` chunks in flight
        per peer, so two peers serve about twice as fast as one;
      - a chunk no usable peer offers comes from the server (`fetch_server`), in random order; while
        such chunks remain, the tracker is asked again (`refresh(shas)` -> offers) every
        `refresh_every` seconds, so chunks that other machines fetched meanwhile come from them;
      - a failed chunk is tried with its next holder, then the server; a peer that fails
        PEER_FAILURES chunks in a row is dropped.
    `fetch_peer(peer, sha, ticket)` and `fetch_server(sha)` return the bytes received; each landed
    chunk is passed to `announce(sha)` so the tracker can offer it to others at once."""

    def __init__(self, sizes: dict[str, int], offers: dict | None, fetch_peer, fetch_server, threads: int = 8,
                 per_peer: int = PER_PEER, refresh=None, refresh_every: float = REFRESH_SECONDS, announce=None,
                 server_slots: int | None = None, seed: int | None = None):
        self.sizes = dict(sizes)
        self.pending = set(self.sizes)
        self.fetch_peer, self.fetch_server = fetch_peer, fetch_server
        self.threads = max(1, threads)
        self.per_peer = max(1, per_peer)
        self.refresh, self.refresh_every, self.announce = refresh, refresh_every, announce
        self.server_slots = server_slots or self.threads
        self.rng = random.Random(seed)
        self.holders: dict[str, list[dict]] = {}
        self.tickets: dict[str, str] = {}
        self.busy: dict[str, int] = {}
        self.failures: dict[str, int] = {}
        self.tried: set[tuple[str, str]] = set()
        self.server_busy = 0
        self.refreshing = False
        self.refreshed_at = time.monotonic()
        self.cond = threading.Condition()
        self.error: BaseException | None = None
        self.peer_bytes: dict[str, int] = {}
        self.server_bytes = 0
        self.refreshes = 0
        self.merge(offers)

    def merge(self, offers: dict | None) -> None:
        for sha, offer in (offers or {}).items():
            if sha not in self.sizes or not isinstance(offer, dict) or not offer.get("peers") or not offer.get("ticket"):
                continue
            self.tickets[sha] = offer["ticket"]
            known = {p["id"] for p in self.holders.get(sha, [])}
            self.holders.setdefault(sha, []).extend(p for p in offer["peers"] if p.get("id") not in known)

    def _usable(self, sha: str) -> list[dict]:
        return [p for p in self.holders.get(sha, [])
                if self.failures.get(p["id"], 0) < PEER_FAILURES and (sha, p["id"]) not in self.tried]

    def _pick(self):
        best = None
        for sha in self.pending:
            usable = self._usable(sha)
            free = [p for p in usable if self.busy.get(p["id"], 0) < self.per_peer]
            if free:
                peer = min(free, key=lambda p: self.busy.get(p["id"], 0))  # min() keeps the tracker's order on ties
                rank = (len(usable), self.busy.get(peer["id"], 0), self.rng.random())
                if best is None or rank < best[0]:
                    best = (rank, sha, peer)
        if best:
            return "peer", best[1], best[2]
        orphans = [sha for sha in self.pending if not self._usable(sha)]
        if not orphans:
            return None  # every chunk left has a holder, all busy: wait for one
        if (self.refresh is not None and not self.refreshing
                and time.monotonic() - self.refreshed_at >= self.refresh_every):
            return "refresh", sorted(orphans), None
        if self.server_busy < self.server_slots:
            return "server", self.rng.choice(sorted(orphans)), None
        return None

    def _worker(self) -> None:
        while True:
            with self.cond:
                while True:
                    if self.error is not None or not self.pending:
                        return
                    task = self._pick()
                    if task is not None:
                        break
                    self.cond.wait(timeout=0.25)
                kind, what, peer = task
                if kind == "peer":
                    self.pending.discard(what)
                    self.busy[peer["id"]] = self.busy.get(peer["id"], 0) + 1
                elif kind == "server":
                    self.pending.discard(what)
                    self.server_busy += 1
                else:
                    self.refreshing = True
            if kind == "refresh":
                try:
                    offers = self.refresh(what)
                except Exception:  # the tracker is an optimisation: the server still has every chunk
                    offers = {}
                with self.cond:
                    self.merge(offers)
                    self.refreshes += 1
                    self.refreshing = False
                    self.refreshed_at = time.monotonic()
                    self.cond.notify_all()
                continue
            if kind == "peer":
                try:
                    got = self.fetch_peer(peer, what, self.tickets[what])
                except (OSError, ValueError):
                    with self.cond:
                        self.busy[peer["id"]] -= 1
                        self.failures[peer["id"]] = self.failures.get(peer["id"], 0) + 1
                        self.tried.add((what, peer["id"]))
                        self.pending.add(what)
                        self.cond.notify_all()
                    continue
                with self.cond:
                    self.busy[peer["id"]] -= 1
                    self.failures[peer["id"]] = 0
                    name = peer.get("name", peer["id"])
                    self.peer_bytes[name] = self.peer_bytes.get(name, 0) + got
                    self.cond.notify_all()
            else:
                try:
                    got = self.fetch_server(what)
                except BaseException as exc:
                    with self.cond:
                        self.server_busy -= 1
                        self.error = self.error or exc
                        self.cond.notify_all()
                    return
                with self.cond:
                    self.server_busy -= 1
                    self.server_bytes += got
                    self.cond.notify_all()
            if self.announce is not None:
                try:
                    self.announce(what)
                except Exception:  # the next heartbeat reports it anyway
                    pass

    def run(self) -> dict:
        """Fetch everything; raises what the server raised. Returns {"bytes", "server", "peers", "seconds", "refreshes"}."""
        started = time.monotonic()
        workers = [threading.Thread(target=self._worker, daemon=True, name=f"swarm-{i}") for i in range(self.threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if self.error is not None:
            raise self.error
        return {"bytes": self.server_bytes + sum(self.peer_bytes.values()), "server": self.server_bytes,
                "peers": dict(self.peer_bytes), "seconds": time.monotonic() - started, "refreshes": self.refreshes}


# ---- the iroh-blobs sidecar (optional transport) ------------------------------------------------------

SIDECAR_PORT = 8767
SIDECAR_IMAGE = "pool-iroh"


class SidecarError(OSError):
    pass


class IrohSidecar:
    """The agent's side of pool/sidecar/iroh, a Rust binary built on iroh and iroh-blobs (the agent
    stays standard library only). Between machines it moves chunks over QUIC, with hole punching and
    relays, BLAKE3-verified as they stream; the agent still checks each chunk's sha256.

    Authorization is the same as over HTTP: the fetcher's sidecar hands the holder's sidecar the
    server's ticket, and the holder's sidecar asks its own agent's peer server (on loopback, HEAD or
    GET /peer/chunks/<sha>) to check it, so the pool key never leaves the agent. Peers find each
    other's sidecar through the tracker: the agent advertises {"id", "addr"} (its iroh endpoint
    address) in its peer profile.

    Control protocol, one JSON line per TCP connection on loopback (see the sidecar's main.rs):
    status -> {"ok", "id", "addr"}; fetch {peer, sha, ticket} -> {"ok", "size"} and then the bytes."""

    def __init__(self, port: int = SIDECAR_PORT, host: str = "127.0.0.1", process=None, container: str | None = None):
        self.host, self.port = host, port
        self.process, self.container = process, container

    @classmethod
    def launch(cls, agent_port: int, port: int = SIDECAR_PORT, binary: str | None = None, image: str | None = None,
               lan: bool = False, bind: str | None = None, wait: float = 20.0, agent_host: str = "127.0.0.1") -> "IrohSidecar":
        """Start the sidecar: `binary` (or $POOL_IROH_BIN, or pool-iroh on PATH), else the container
        `image` with the host's network. `agent_host` is where the agent's peer server listens (its LAN
        address when it binds there). Raises SidecarError when neither comes up."""
        if agent_host in ("0.0.0.0", ""):
            agent_host = "127.0.0.1"
        extra = ["--control", f"127.0.0.1:{port}", "--agent", f"{agent_host}:{agent_port}"]
        if lan:
            extra.append("--lan")
        if bind:
            extra += ["--bind", bind]
        try:
            cls(port).status(timeout=1)
            raise SidecarError(f"something already answers on control port {port} (a sidecar left from an earlier agent?)")
        except SidecarError:
            raise
        except OSError:
            pass  # free, as it should be
        binary = binary or os.environ.get("POOL_IROH_BIN") or shutil.which("pool-iroh")
        if binary:
            if sys.platform.startswith("linux"):
                extra += ["--parent", str(os.getpid())]  # it exits with the agent, even after a SIGKILL
            process = subprocess.Popen([binary, *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True)
            sidecar = cls(port, process=process)
        elif image:
            name = f"pool-iroh-{port}"
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
            started = subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--network", "host",
                                      "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
                                      image, *extra], capture_output=True, text=True, timeout=120)
            if started.returncode != 0:
                raise SidecarError(f"docker run {image}: {started.stderr.strip()[-300:]}")
            sidecar = cls(port, container=name)
        else:
            raise SidecarError("no pool-iroh binary and no sidecar image")
        deadline = time.monotonic() + wait
        while True:
            try:
                sidecar.status()
                return sidecar
            except OSError as exc:
                if time.monotonic() > deadline or (sidecar.process is not None and sidecar.process.poll() is not None):
                    sidecar.stop()
                    raise SidecarError(f"the sidecar did not come up: {exc}") from None
                time.sleep(0.2)

    def _open(self, request: dict, timeout: float):
        conn = socket.create_connection((self.host, self.port), timeout=timeout)
        try:
            conn.sendall(json.dumps(request).encode() + b"\n")
            reader = conn.makefile("rb")
            line = reader.readline(1 << 16)
        except OSError:
            conn.close()
            raise
        try:
            reply = json.loads(line or b"{}")
        except json.JSONDecodeError:
            reply = {}
        if not reply.get("ok"):
            reader.close()
            conn.close()
            raise SidecarError(reply.get("error") or "no reply from the sidecar")
        return conn, reader, reply

    def status(self, timeout: float = 5) -> dict:
        conn, reader, reply = self._open({"op": "status"}, timeout)
        reader.close()
        conn.close()
        return {"id": reply["id"], "addr": reply["addr"]}

    def fetch(self, peer: dict, sha: str, ticket_value: str, dest: Path, timeout: float = 600) -> int:
        """One chunk from `peer`'s sidecar (peer["iroh"]["addr"]) into `dest`, checking its sha256."""
        iroh = peer.get("iroh") or {}
        if not iroh.get("addr"):
            raise SidecarError(f"{peer.get('name')} has no sidecar")
        conn, reader, reply = self._open({"op": "fetch", "peer": iroh["addr"], "sha": sha, "ticket": ticket_value}, timeout)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".{os.getpid()}.{threading.get_ident()}.iroh.part")
        digest = hashlib.sha256()
        left = size = int(reply.get("size", 0))
        try:
            with open(tmp, "wb") as handle:
                while left:
                    block = reader.read(min(left, 1 << 20))
                    if not block:
                        raise SidecarError("the sidecar stopped mid-chunk")
                    digest.update(block)
                    handle.write(block)
                    left -= len(block)
            if digest.hexdigest() != sha:
                raise ValueError(f"chunk {sha[:12]} from {peer.get('name', '?')} (iroh) arrived corrupt")
            os.chmod(tmp, 0o444)
            tmp.replace(dest)
        finally:
            reader.close()
            conn.close()
            tmp.unlink(missing_ok=True)
        return size

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True, timeout=30)
