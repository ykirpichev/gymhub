"""network: island, a gang rank that reaches only its peers (pool/island.py).

Units always run. POOL_TEST_ISLAND=1 also runs the Docker test, which starts real holder and helper
containers and checks what a job can reach (it needs Docker, the host's ip and iptables, and the
busybox:1.36 and python:3.12-slim images or the network to pull them).
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from pool import island, scheduler, spec as specs
from pool.agent import DockerRuntime
from tests.test_pool import FREE, _flag, _gang, _peer_machine, _status, db, db_url  # noqa: F401  (fixtures)


def test_port_blocks_are_stable_and_inside_the_range():
    lo, hi = island.port_block("gabc123")
    assert (lo, hi) == island.port_block("gabc123")
    assert hi - lo + 1 == island.BLOCK and island.PORT_BASE <= lo and hi < 61000
    assert len({island.port_block(f"g{i:06x}") for i in range(200)}) > 120  # spread over all blocks (~148 expected)


def test_ray_ports_stay_in_the_block():
    lo, hi = island.port_block("gabc123")
    args = island.ray_args(lo, "10.0.0.164").split()
    ports = [int(a.split("=")[1]) for a in args if a.split("=")[0].endswith("-port")]
    assert ports and all(lo < p < lo + island.FIXED for p in ports)
    assert "--node-ip-address=10.0.0.164" in args and "--include-dashboard=false" in args


def test_addresses_are_checked_before_they_reach_a_shell():
    for bad in ("10.0.0.1; reboot", "$(id)", "fe80::1", "example.com", ""):
        with pytest.raises(ValueError):
            island.rules_script(bad, "172.17.0.2", 41000, 41063, [])
        with pytest.raises(ValueError):
            island.rules_script("10.0.0.164", "172.17.0.2", 41000, 41063, [bad])


def test_rules_admit_only_the_peers():
    script = island.rules_script("10.0.0.164", "172.17.0.2", 41000, 41063, ["10.0.0.100", "10.0.0.164", "10.0.0.100"])
    lines = script.splitlines()
    assert "iptables -A INPUT -s 10.0.0.100 -p tcp -j ACCEPT" in lines
    assert "iptables -A OUTPUT -d 10.0.0.100 -p tcp --dport 41000:41063 -j ACCEPT" in lines
    assert not any("-s 10.0.0.164 -p tcp -j ACCEPT" in line for line in lines)  # not itself
    assert sum("INPUT -s 10.0.0.100" in line for line in lines) == 1  # each peer once
    assert lines.index("iptables -P INPUT DROP") > lines.index("iptables -A INPUT -s 10.0.0.100 -p tcp -j ACCEPT")
    assert "iptables -P OUTPUT DROP" in lines and "iptables -P FORWARD DROP" in lines
    assert "DNAT --to-destination 10.0.0.164" in script and "SNAT --to-source 172.17.0.2" in script


def test_holder_and_helper_commands():
    job = {"id": "j1", "tries": 2}
    holder = island.holder_command(job, "a1", "10.0.0.164", 41000, 41063)
    assert _flag(holder, "--publish") == "10.0.0.164:41000-41063:41000-41063/tcp"
    assert f"net.ipv4.ip_local_port_range={41000 + island.FIXED} 41063" in holder
    assert "net.ipv6.conf.all.disable_ipv6=1" in holder and _flag(holder, "--cap-drop") == "ALL"
    assert "pool.machine=a1" in holder  # the agent's start-up cleanup removes it
    helper = island.helper_command("pool-j1-2-net", "echo")
    assert _flag(helper, "--network") == "container:pool-j1-2-net" and island.IMAGE in helper
    assert "type=bind,src=/,dst=/host,readonly" in helper


def test_job_env_names_the_interface_only_when_it_exists():
    assert island.job_env("10.0.0.164", 41000, 41063)["GLOO_SOCKET_IFNAME"] == island.IFACE
    on_lo = island.job_env("10.0.0.164", 41000, 41063, iface="lo")
    assert "GLOO_SOCKET_IFNAME" not in on_lo and on_lo["POOL_NODE_IP"] == "10.0.0.164"


def test_island_spec_needs_a_gang_and_an_island_capable_machine():
    with pytest.raises(specs.SpecError):
        specs.normalize({"image": "x", "command": ["x"], "network": "island"})
    spec = specs.normalize({"image": "x", "command": ["x"], "gang": {"size": 2}, "network": "island"})
    plain = {"id": "a", "owner": "ann", "labels": {}, "policy": {}, "profile": {"gpus": [], "allow_images": ["*"]}}
    old = {**plain, "profile": {**plain["profile"], "peer": {"addr": "10.0.0.2", "port": 8766}}}
    no_tools = {**plain, "profile": {**plain["profile"], "peer": {"addr": "10.0.0.2", "port": 8766, "island": False,
                                                                     "island_why": "no iptables on this machine"}}}
    able = {**plain, "profile": {**plain["profile"], "peer": {"addr": "10.0.0.2", "port": 8766, "island": True}}}
    why: list = []
    assert scheduler.fits(spec, old, FREE, why=why) is None and "agent too old" in why[0]
    why = []
    assert scheduler.fits(spec, no_tools, FREE, why=why) is None and "iptables" in why[0]
    assert scheduler.fits(spec, able, FREE) == []


def test_island_rendezvous_hands_out_the_gangs_ports(db):
    r0, r1 = _gang(db)
    for job_id in (r0, r1):
        spec = json.loads(db.one("select spec from jobs where id=?", job_id)["spec"])
        spec["network"] = "island"
        db.run("update jobs set spec=? where id=?", json.dumps(spec), job_id)
    a, b = _peer_machine(db, "a"), _peer_machine(db, "bb")
    for machine in (a, b):
        machine["profile"]["peer"]["island"] = True
        db.run("update machines set profile=? where id=?", json.dumps(machine["profile"]), machine["id"])
    assert scheduler.claim(db, a, a["free"])["id"] == r0
    assert scheduler.claim(db, b, b["free"])["id"] == r1
    meet = scheduler.rendezvous(db, _status(db, r0))
    lo, hi = island.port_block("gabc123")
    assert meet["ports"] == [lo, hi] and meet["master_port"] == lo


def test_an_island_job_joins_its_holders_namespace():
    spec = specs.normalize({"image": "img", "command": ["x"], "gang": {"size": 2}, "network": "island"})
    job = {"id": "j1", "tries": 1, "gpus": [], "spec": spec, "image": "img", "island_holder": "pool-j1-1-net"}
    args = DockerRuntime("a1").command(job, {"inputs": {}, "outputs": Path("/tmp/o"), "checkpoint": Path("/tmp/c")}, {}, 4)
    assert _flag(args, "--network") == "container:pool-j1-1-net" and _flag(args, "--cap-drop") == "ALL"
    without = DockerRuntime("a1").command({**job, "island_holder": None}, {"inputs": {}, "outputs": Path("/tmp/o"),
                                                                           "checkpoint": Path("/tmp/c")}, {}, 4)
    assert _flag(without, "--network") == "none"  # never the host's network by accident


PROBE = r"""
import socket, sys
def reach(host, port):
    s = socket.socket(); s.settimeout(2)
    try:
        s.connect((host, port)); return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "blocked"
    finally:
        s.close()
peer, addr, lo = sys.argv[1], sys.argv[2], int(sys.argv[3])
print("peer-block", reach(peer, lo + 10))
print("peer-other", reach(peer, 22))
print("internet", reach("1.1.1.1", 443))
try:
    socket.getaddrinfo("example.com", 80); print("dns open")
except OSError:
    print("dns blocked")
print("own-host", reach(addr, 8765))
"""


@pytest.mark.skipif(os.environ.get("POOL_TEST_ISLAND") != "1", reason="needs Docker and iptables (POOL_TEST_ISLAND=1)")
def test_a_real_island_reaches_only_its_peer(tmp_path):
    import socket
    import threading

    gateway = "172.17.0.1"  # stands in for a peer machine: the host, seen from the default bridge
    addr = "192.0.2.10"     # a documentation address stands in for this machine's LAN address
    lo, hi = island.port_block("gtest")
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((gateway, lo + 10))
    listener.listen()
    threading.Thread(target=lambda: listener.accept(), daemon=True).start()
    job = {"id": "jtest", "tries": 1}
    try:
        # Publishing on 192.0.2.10 would fail (not a host address), and the gateway's ports are the
        # listener's: publish on loopback instead (this test checks what the job reaches, not inbound).
        command = island.holder_command(job, "test", "127.0.0.1", lo, hi)
        island.teardown(job)
        subprocess.run(command, check=True, capture_output=True)
        ip = subprocess.run(["docker", "inspect", "-f", '{{(index .NetworkSettings.Networks "bridge").IPAddress}}',
                             island.holder_name(job)], check=True, capture_output=True, text=True).stdout.strip()
        out = subprocess.run(island.helper_command(island.holder_name(job), island.rules_script(addr, ip, lo, hi, [gateway])),
                             capture_output=True, text=True)
        assert "rules=ok" in out.stdout, out.stderr
        probe = subprocess.run(["docker", "run", "--rm", "--network", f"container:{island.holder_name(job)}", "--cap-drop", "ALL",
                                "--user", "1000:1000", "--read-only", "python:3.12-slim", "python", "-c", PROBE,
                                gateway, addr, str(lo)], capture_output=True, text=True, timeout=120)
        result = dict(line.split() for line in probe.stdout.splitlines())
        assert result == {"peer-block": "open", "peer-other": "blocked", "internet": "blocked", "dns": "blocked",
                          "own-host": "refused"}, probe.stdout + probe.stderr
    finally:
        island.teardown(job)
        listener.close()


def test_an_agent_next_to_the_server_advertises_its_default_route_address(monkeypatch):
    from pool import peers

    routes = {"spark.local": ("172.19.0.1", "172.19.0.1"),   # mDNS answered with a Docker bridge of this machine
              "192.0.2.1": ("10.0.0.164", "192.0.2.1"),      # the default route
              "10.0.0.5": ("10.0.0.164", "10.0.0.5")}        # a server elsewhere on the LAN
    monkeypatch.setattr(peers, "_route", lambda host, port: routes.get(host))
    assert peers.local_address("http://spark.local:8765") == "10.0.0.164"
    assert peers.local_address("http://10.0.0.5:8765") == "10.0.0.164"
