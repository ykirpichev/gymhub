"""M5 chunk tracker and swarm (#89): the server tracks which machines hold which chunks, agents fetch
an artifact's chunks from several peers at once and the rest from the server, and the optional
iroh-blobs sidecar (pool/sidecar/iroh) moves them between machines."""

import os
import shutil
import threading
import time
import urllib.error
import urllib.request

import pytest

from pool import cli, peers
from pool.client import Client
from tests.test_pool import TRAIN, db_url, pool_env, small_chunks, start_agent, wait_for, write  # noqa: F401  (fixtures)

SHAS = [f"{i:064x}" for i in range(12)]


def offer(*names, ticket="t"):
    return {"ticket": ticket, "peers": [{"id": n, "name": n} for n in names]}


def slow(seconds, log=None, fail=()):
    """A fetcher that takes `seconds` per chunk, records who served what, and fails for `fail` peers."""
    lock = threading.Lock()

    def fetch(peer, sha, ticket):
        if peer["id"] in fail:
            raise OSError("unreachable")
        time.sleep(seconds)
        with lock:
            if log is not None:
                log.append((peer["id"], sha))
        return 100

    return fetch


def no_server(sha):
    raise AssertionError(f"{sha} should have come from a peer")


def test_swarm_spreads_chunks_over_every_holder():
    log = []
    swarm = peers.Swarm({sha: 100 for sha in SHAS}, {sha: offer("a", "b", "c") for sha in SHAS}, slow(0.05, log),
                        no_server, threads=8, per_peer=2, seed=1)
    stats = swarm.run()
    assert stats["bytes"] == 1200 and stats["server"] == 0
    assert {peer for peer, _ in log} == {"a", "b", "c"}
    assert all(3 <= stats["peers"][p] // 100 <= 5 for p in "abc")  # about a third each
    assert stats["seconds"] < 0.05 * 12 / 2  # faster than one peer alone could serve them


def test_more_peers_fetch_faster():
    def timed(holders):
        swarm = peers.Swarm({sha: 100 for sha in SHAS}, {sha: offer(*holders) for sha in SHAS}, slow(0.04),
                            no_server, threads=8, per_peer=2, seed=2)
        return swarm.run()["seconds"]

    one, three = timed(["a"]), timed(["a", "b", "c"])
    assert three < one / 2


def test_swarm_takes_rarest_chunks_first_and_unoffered_ones_from_the_server():
    log, served = [], []
    offers = {sha: offer("a", "b") for sha in SHAS[:6]}
    offers.update({sha: offer("a") for sha in SHAS[6:9]})  # SHAS[9:] nobody holds

    def server(sha):
        served.append(sha)
        return 100

    swarm = peers.Swarm({sha: 100 for sha in SHAS}, offers, slow(0.02, log), server, threads=1, per_peer=1, seed=3)
    stats = swarm.run()
    assert sorted(served) == SHAS[9:] and stats["server"] == 300
    fetched = [sha for _, sha in log]
    assert set(fetched[:3]) == set(SHAS[6:9])  # one holder each: before the chunks two peers hold


def test_a_failing_peer_falls_back_to_the_next_holder_then_the_server():
    log, served = [], []

    def server(sha):
        served.append(sha)
        return 100

    offers = {sha: offer("bad", "good") for sha in SHAS[:4]}
    offers.update({sha: offer("bad") for sha in SHAS[4:6]})
    swarm = peers.Swarm({sha: 100 for sha in SHAS[:6]}, offers, slow(0.01, log, fail={"bad"}), server, threads=2, seed=4)
    stats = swarm.run()
    assert stats["bytes"] == 600 and {p for p, _ in log} == {"good"}
    assert sorted(served) == SHAS[4:6]
    assert swarm.failures["bad"] >= peers.PEER_FAILURES or len(swarm.tried) >= 2


def test_swarm_asks_the_tracker_again_for_unoffered_chunks():
    """Two machines fetching the same artifact at the same time: a chunk the tracker did not offer at
    first comes from the peer that has fetched it meanwhile, not from the server."""
    log, served, asked = [], [], []

    def server(sha):
        served.append(sha)
        time.sleep(0.2)
        return 100

    def refresh(shas):
        asked.append(list(shas))
        return {sha: offer("late") for sha in shas}

    swarm = peers.Swarm({sha: 100 for sha in SHAS[:6]}, {}, slow(0.01, log), server, threads=2, refresh=refresh,
                        refresh_every=0.05, server_slots=1, seed=5)
    stats = swarm.run()
    assert asked and stats["refreshes"] >= 1
    assert len(served) < 6 and stats["peers"].get("late", 0) > 0


def test_server_errors_propagate():
    def server(sha):
        raise RuntimeError("server down")

    with pytest.raises(RuntimeError, match="server down"):
        peers.Swarm({sha: 1 for sha in SHAS[:3]}, {}, slow(0), server, threads=2).run()


def test_announced_chunks_are_offered_only_for_readable_artifacts(pool_env, small_chunks):
    db = pool_env["db"]
    one = start_agent(pool_env, "box1")
    two = start_agent(pool_env, "box2")
    wait_for(lambda: one.peer_key and two.peer_key)
    data = write(pool_env["tmp"] / "data", {"numbers.txt": "\n".join(str(i) for i in range(500))})
    artifact = Client.member().upload_folder(data, kind="data")["artifact"]
    chunks = [row["chunk"] for row in db.all("select chunk from artifact_chunks where artifact=?", artifact)]
    assert len(chunks) > 1
    reply = one.client.call("POST", "/v1/agent/chunks", {"add": chunks + ["f" * 64, "junk"]})
    assert reply["added"] == len(chunks)  # unknown chunks are not tracked
    with pytest.raises(Exception):  # box2 runs no job with this artifact: it may not ask
        two.client.call("POST", "/v1/agent/peer-sources", {"artifact": artifact, "chunks": chunks})
    db.run("insert into jobs (id, pool, owner, spec, status, machine, created_at) select 'jx', pool, owner, ?, 'running', id, 0 "
           "from machines where name='box2'", '{"image": "x", "command": ["x"], "inputs": [{"artifact": "%s"}]}' % artifact)
    sources = two.client.call("POST", "/v1/agent/peer-sources", {"artifact": artifact, "chunks": chunks})["sources"]
    assert set(sources) == set(chunks) and all(s["peers"][0]["name"] == "box1" for s in sources.values())
    one.client.call("POST", "/v1/agent/chunks", {"remove": chunks[:1]})
    sources = two.client.call("POST", "/v1/agent/peer-sources", {"artifact": artifact, "chunks": chunks})["sources"]
    assert chunks[0] not in sources
    db.run("delete from jobs where id='jx'")


def test_peer_server_checks_tickets_on_head(pool_env, small_chunks):
    agent = start_agent(pool_env, "box1")
    key = wait_for(lambda: agent.peer_key)
    base = f"http://127.0.0.1:{agent.peer_server.port}/peer/chunks/{'a' * 64}"

    def head(ticket):
        request = urllib.request.Request(base, method="HEAD", headers={"X-Pool-Ticket": ticket})
        try:
            with urllib.request.urlopen(request, timeout=5) as reply:
                return reply.status
        except urllib.error.HTTPError as exc:
            return exc.code

    assert head("nope") == 403
    assert head(peers.ticket(key, "m", "a" * 64)) == 404  # a good ticket, but the chunk is not here


def test_a_machine_fetches_from_two_peers_at_once(pool_env, capsys, small_chunks):
    data = write(pool_env["tmp"] / "data", {"numbers.txt": "\n".join(str(i) for i in range(6000))})
    code = write(pool_env["tmp"] / "code", {"train.py": TRAIN})
    boxes = [start_agent(pool_env, name) for name in ("box1", "box2", "box3")]
    run = ["run", "--code", str(code), "--data", str(data), "--no-get", "--", "python3", "train.py"]
    for index, name in enumerate(("box1", "box2")):  # box1 from the server, then box2 (from box1)
        for other in ("box1", "box2", "box3"):
            assert cli.main(["machine", "drain", other] + ([] if other != name else ["--off"])) == 0
        assert cli.main(run) == 0
        assert f"on {name}" in capsys.readouterr().out
    wait_for(lambda: pool_env["db"].one("select count(distinct machine) as n from chunk_locations")["n"] == 2)
    for other in ("box1", "box2", "box3"):
        assert cli.main(["machine", "drain", other] + ([] if other != "box3" else ["--off"])) == 0
    assert cli.main(run) == 0
    out = capsys.readouterr().out
    assert "on box3" in out and "from peer box1" in out and "from peer box2" in out and "sum 17997000" in out
    assert "the rest from the server" not in out
    assert boxes[0].peer_server.served_bytes > 0 and boxes[1].peer_server.served_bytes > 0


IROH = os.environ.get("POOL_IROH_BIN") or shutil.which("pool-iroh")


@pytest.mark.skipif(not IROH, reason="needs the pool-iroh sidecar (cargo build in pool/sidecar/iroh; POOL_IROH_BIN)")
def test_chunks_move_through_the_iroh_sidecars(pool_env, capsys, small_chunks):
    from tests.test_pool import free_port

    data = write(pool_env["tmp"] / "data", {"numbers.txt": "\n".join(str(i) for i in range(3000))})
    code = write(pool_env["tmp"] / "code", {"train.py": TRAIN})
    one = start_agent(pool_env, "box1", iroh={"binary": IROH, "port": free_port(), "lan": True})
    two = start_agent(pool_env, "box2", iroh={"binary": IROH, "port": free_port(), "lan": True})
    wait_for(lambda: one.sidecar and two.sidecar)
    run = ["run", "--code", str(code), "--data", str(data), "--no-get", "--", "python3", "train.py"]
    assert cli.main(["machine", "drain", "box2"]) == 0
    assert cli.main(run) == 0
    assert cli.main(["machine", "drain", "box1"]) == 0
    assert cli.main(["machine", "drain", "box2", "--off"]) == 0
    wait_for(lambda: pool_env["db"].one("select count(*) as n from chunk_locations")["n"] > 0)
    assert cli.main(run) == 0
    out = capsys.readouterr().out
    assert "on box2" in out and "from peer box1" in out and "sum 4498500" in out
    assert two.peer_bytes["iroh"] > 0 and two.peer_bytes["http"] == 0


@pytest.mark.skipif(not IROH, reason="needs the pool-iroh sidecar")
def test_the_sidecar_refuses_a_bad_ticket(pool_env, small_chunks):
    from tests.test_pool import free_port

    one = start_agent(pool_env, "box1", iroh={"binary": IROH, "port": free_port(), "lan": True})
    two = start_agent(pool_env, "box2", iroh={"binary": IROH, "port": free_port(), "lan": True})
    wait_for(lambda: one.sidecar and two.sidecar and one.peer_key)
    peer = {"name": "box1", "iroh": one.profile["peer"]["iroh"]}
    with pytest.raises(peers.SidecarError, match="refused"):
        two.sidecar.fetch(peer, "a" * 64, "m:0:bad", pool_env["tmp"] / "x")
