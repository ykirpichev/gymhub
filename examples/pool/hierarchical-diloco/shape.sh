#!/usr/bin/env bash
# Shape the link between two loopback addresses (two islands on one machine) with tc netem:
#   ./shape.sh on 100mbit [5ms]   RATE each way, plus a one-way delay, between $A and $B
#   ./shape.sh off
#   ./shape.sh show
# Only packets from or to A=127.0.0.2 and B=127.0.0.3 are shaped; the rest of the loopback traffic
# goes through an unshaped band. Needs Docker: tc runs in a short-lived container with NET_ADMIN on
# the host's network, using the host's own tc (chroot into a read-only view of /), the way
# pool/island.py runs ip and iptables. `off` puts lo back to its default (no qdisc).
set -euo pipefail
A="${A:-127.0.0.2}"; B="${B:-127.0.0.3}"
tc() { docker run --rm --network host --cap-add NET_ADMIN -v /:/host:ro busybox:1.36 chroot /host /usr/sbin/tc "$@"; }
case "${1:-show}" in
  on)
    rate="$2"; delay="${3:-5ms}"
    tc qdisc del dev lo root 2>/dev/null || true
    tc qdisc add dev lo root handle 1: prio bands 5 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
    tc qdisc add dev lo parent 1:4 handle 40: netem rate "$rate" delay "$delay" limit 200000   # A -> B
    tc qdisc add dev lo parent 1:5 handle 50: netem rate "$rate" delay "$delay" limit 200000   # B -> A
    tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip dst "$B"/32 flowid 1:4
    tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip src "$A"/32 flowid 1:4
    tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip dst "$A"/32 flowid 1:5
    tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip src "$B"/32 flowid 1:5
    echo "lo: $A <-> $B at $rate each way, $delay one way";;
  off) tc qdisc del dev lo root 2>/dev/null || true; echo "lo: not shaped";;
  show) tc -s qdisc show dev lo;;
  *) echo "usage: $0 on RATE [DELAY] | off | show" >&2; exit 2;;
esac
