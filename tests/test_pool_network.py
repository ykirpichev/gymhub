"""Nothing reachable from outside the LAN (#128, docs/design/worker-v2.md section 4): the peer
server and island ports take only private sources, and the pool server warns on a public bind."""

import http.client
import urllib.request
from types import SimpleNamespace

import pytest

from pool import island, peers, server
from pool.agent import Agent


@pytest.mark.parametrize("address", ["10.0.0.5", "172.16.3.4", "192.168.1.20", "127.0.0.1", "100.101.102.103",
                                     "169.254.1.1", "::1", "fd12:3456::1", "fe80::1%eth0", "::ffff:192.168.1.2"])
def test_lan_addresses_are_private(address):
    assert peers.private_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "1.1.1.1", "2001:4860:4860::8888", "::ffff:8.8.8.8", "0.0.0.0",
                                     "224.0.0.1", "", None, "example.com"])
def test_public_and_bogus_addresses_are_not(address):
    assert not peers.private_address(address)


def _server(log):
    return peers.PeerServer(0, lambda: "k" * 32, lambda sha: None, host="127.0.0.1", log=log.append).start()


def test_the_peer_server_refuses_a_public_source_and_logs_it():
    log = []
    srv = _server(log)
    try:
        assert srv.httpd.verify_request(None, ("34.120.5.7", 40000)) is False
        assert srv.httpd.verify_request(None, ("192.168.1.9", 40000)) is True
        assert srv.refused == 1 and "34.120.5.7" in log[0]
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/peer/healthz", timeout=5) as reply:
            assert reply.status == 200
    finally:
        srv.stop()


def test_a_connection_from_a_faked_public_address_is_closed(monkeypatch):
    log = []
    srv = _server(log)
    monkeypatch.setattr(peers, "private_address", lambda address: False)  # loopback now looks public
    try:
        with pytest.raises((http.client.RemoteDisconnected, OSError)):
            urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/peer/healthz", timeout=5).read()
        assert srv.refused == 1 and "127.0.0.1" in log[0]
    finally:
        srv.stop()


def test_the_peer_server_binds_to_the_lan_address():
    assert peers.bind_address("192.168.1.20") == "192.168.1.20"
    assert peers.bind_address(None) == "0.0.0.0"
    assert peers.bind_address("fd00::1") == "0.0.0.0"  # the server is IPv4; the source check still applies


def test_island_lets_in_only_private_peers():
    script = island.rules_script("10.0.0.164", "172.17.0.2", 41000, 41063, ["10.0.0.165", "8.8.8.8"])
    assert "-s 10.0.0.165 " in script and "8.8.8.8" not in script
    holder = island.holder_command({"id": "j1", "tries": 0}, "m1", "10.0.0.164", 41000, 41063)
    assert "10.0.0.164:41000-41063:41000-41063/tcp" in holder


def test_no_island_on_a_public_peer_address():
    agent = SimpleNamespace(runtime=SimpleNamespace(name="docker"), config={})
    can, why = Agent.island_support(agent, "8.8.8.8")
    assert not can and "not a private" in why


def test_the_server_warns_when_it_listens_outside_the_lan():
    assert server.exposure_warning("192.168.1.10") is None
    assert server.exposure_warning("127.0.0.1") is None
    assert server.exposure_warning("0.0.0.0", default_route="192.168.1.10") is None
    assert "HTTPS" in server.exposure_warning("0.0.0.0", default_route="8.8.8.8")
    assert "8.8.8.8" in server.exposure_warning("8.8.8.8")
