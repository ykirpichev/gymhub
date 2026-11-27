"""Swarm transfer benchmark (#89): one machine fetches a multi-chunk artifact from 1, 2, 3 peers.

Everything runs on this machine, as separate processes: a pool server, seed agents that already
hold the artifact, and a fresh fetcher agent each round. Each seed caps what peers may take from it
(--peer-limit-mbps, a simulated uplink), so the fetch rate shows how the swarm adds peers up.

    python examples/pool/swarm/bench.py --root /tmp/swarm-bench --chunks 16 --cap 1000 [--iroh BIN]

Ports: the server on --port (default 8774), peer servers on the next ports, iroh control ports
(loopback) at +100. Jobs run as plain processes (--unsafe-no-container): they only print.
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FETCHED = re.compile(r"\[pool\] fetched (?P<what>.+?) in (?P<s>[\d.]+) s \((?P<mbps>\d+) Mbit/s\)(?P<via>.*)")


def pool(root: Path, who: str, *args: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
    env = {**os.environ, "POOL_CONFIG_DIR": str(root / f"cfg-{who}"), "PYTHONPATH": str(REPO)}
    env.pop("POOL_SERVER", None)
    env.pop("POOL_TOKEN", None)
    out = subprocess.run([sys.executable, "-m", "pool.cli", *args], env=env, capture_output=True, text=True, **kw)
    if check and out.returncode != 0:
        raise SystemExit(f"pool {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out


class Bench:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root)
        self.url = f"http://127.0.0.1:{args.port}"
        self.procs: dict[str, subprocess.Popen] = {}

    def start_server(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        log = open(self.root / "server.log", "w")
        self.procs["server"] = subprocess.Popen(
            [sys.executable, "-m", "pool.cli", "server", "--data", str(self.root / "server"), "--port", str(self.args.port),
             "--host", "127.0.0.1"], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(REPO)})
        code = None
        for _ in range(200):
            text = (self.root / "server.log").read_text()
            found = re.search(r"--code (\S+)", text)
            if found:
                code = found.group(1)
                break
            time.sleep(0.1)
        if not code:
            raise SystemExit("the server did not start:\n" + (self.root / "server.log").read_text())
        pool(self.root, "member", "login", self.url, "--code", code, "--name", "bench")

    def make_data(self) -> None:
        data = self.root / "data"
        data.mkdir()
        with open(data / "blob.bin", "wb") as handle:
            for _ in range(self.args.chunks):
                handle.write(os.urandom(64 << 20))

    def register(self, name: str) -> None:
        reply = pool(self.root, "member", "machine", "add", "--name", name).stdout
        code = re.search(r"--code (\S+)", reply).group(1)
        pool(self.root, name, "agent", "register", "--server", self.url, "--code", code, "--name", name, "--gpus", "none")

    def start_agent(self, name: str, peer_port: int, cap: float | None, iroh: bool) -> None:
        cmd = [sys.executable, "-m", "pool.cli", "agent", "run", "--unsafe-no-container", "--label", f"host={name}",
               "--peer-port", str(peer_port), "--peer-addr", "127.0.0.1", "--work", str(self.root / f"work-{name}"),
               "--cache-gb", "20", "--max-jobs", "1"]
        if cap:
            cmd += ["--peer-limit-mbps", str(cap)]
        if iroh:
            cmd += ["--iroh", "--iroh-binary", self.args.iroh, "--iroh-port", str(peer_port + 100), "--iroh-lan"]
        log = open(self.root / f"agent-{name}.log", "a")
        env = {**os.environ, "POOL_CONFIG_DIR": str(self.root / f"cfg-{name}"), "PYTHONPATH": str(REPO)}
        self.procs[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        for _ in range(300):
            if "peer server on" in (self.root / f"agent-{name}.log").read_text() and \
                    (not iroh or "iroh sidecar" in (self.root / f"agent-{name}.log").read_text()):
                break
            time.sleep(0.1)
        time.sleep(1.5)  # its first heartbeat (profile, cache) reaches the server

    def stop_agent(self, name: str) -> None:
        proc = self.procs.pop(name, None)
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)

    def run_on(self, name: str) -> dict:
        out = pool(self.root, "member", "run", "--label", f"host={name}", "--data", str(self.root / "data"), "--no-code",
                   "--no-get", "--", "python3", "-c", "print('ok')", timeout=1800)
        text = out.stdout + out.stderr
        found = FETCHED.search(text)
        if not found:
            return {"text": text}
        via = dict((who, part) for part, who in re.findall(r", ([\d.]+ \w+) from peer ([\w.-]+)", found.group("via")))
        return {"seconds": float(found.group("s")), "mbps": int(found.group("mbps")), "what": found.group("what"),
                "via": via, "server": "the rest from the server" in found.group("via")}

    def measure(self, seeds: list[str], label: str, iroh: bool) -> dict:
        shutil.rmtree(self.root / "work-fetch", ignore_errors=True)
        self.start_agent("fetch", self.args.port + 4, None, iroh)
        result = self.run_on("fetch")
        self.stop_agent("fetch")
        print(f"{label:28s} {len(seeds)} peer(s): {result.get('what', '?')} in {result.get('seconds', '?')} s, "
              f"{result.get('mbps', '?')} Mbit/s; via {result.get('via')}{' + server' if result.get('server') else ''}", flush=True)
        if "seconds" not in result:
            print(result["text"][-2000:])
        return result

    def close(self) -> None:
        for name in list(self.procs):
            if name != "server":
                self.stop_agent(name)
        server = self.procs.pop("server", None)
        if server:
            server.terminate()
            server.wait(timeout=20)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/tmp/swarm-bench")
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--chunks", type=int, default=16, help="64 MiB chunks in the artifact")
    parser.add_argument("--cap", type=float, default=1000, help="each seed's peer upload cap, Mbit/s (0: none)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iroh", help="the pool-iroh binary: also measure over the iroh sidecars")
    args = parser.parse_args()
    bench = Bench(args)
    seeds = [f"seed{i + 1}" for i in range(args.seeds)]
    try:
        bench.start_server()
        bench.make_data()
        for name in seeds + ["fetch"]:
            bench.register(name)
        results = []
        for transport in ["http"] + (["iroh"] if args.iroh else []):
            iroh = transport == "iroh"
            for index, seed in enumerate(seeds):
                if iroh:  # fresh sidecars each round: none still holds chunks imported for the last fetcher
                    for running in seeds[:index]:
                        bench.stop_agent(running)
                        bench.start_agent(running, args.port + 1 + seeds.index(running), args.cap or None, iroh)
                bench.start_agent(seed, args.port + 1 + index, args.cap or None, iroh)
                if transport == "http":  # the first pass fills the seeds (from the server, then from each other)
                    warm = bench.run_on(seed)
                    print(f"  warmed {seed}: {warm.get('what')} in {warm.get('seconds')} s via {warm.get('via')}", flush=True)
                time.sleep(2)
                results.append((transport, index + 1, bench.measure(seeds[: index + 1], f"{transport}, cap {args.cap:g} Mbit/s", iroh)))
            for seed in seeds:
                bench.stop_agent(seed)
        print("\ntransport peers  seconds  Mbit/s")
        for transport, n, result in results:
            print(f"{transport:9s} {n:5d}  {result.get('seconds', '?'):>7}  {result.get('mbps', '?'):>6}")
    finally:
        bench.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
