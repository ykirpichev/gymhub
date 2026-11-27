#!/bin/sh
# The pool image's entrypoint (deploy/pool/Dockerfile).
#
#   server [ARGS]       the pool server on :8765, state in $POOL_DATA (/data)
#   agent [ARGS]        the machine agent (the default): registers first when $POOL_CODE is set and
#                       this machine is not registered yet, then `pool agent run --work $POOL_STATE/work`
#   pool-agent CMD      owner commands in a running agent: docker exec pool-agent pool-agent status
#   pool CMD            the member CLI (pool login, pool machine ls, ...)
#   anything else       run as given
#
# Started as root, it drops to POOL_UID:POOL_GID (default 10001) before running anything. The agent
# also gets the group that owns /var/run/docker.sock, which is what lets it start job containers.
set -eu

POOL_UID="${POOL_UID:-10001}"
POOL_GID="${POOL_GID:-$POOL_UID}"
STATE="${POOL_STATE:-/var/lib/pool}"
CONF="${POOL_AGENT_CONFIG:-/etc/pool/agent.json}"
SOCK=/var/run/docker.sock

# Docker restarts the agent's container whatever its exit status (it has no RestartPreventExitStatus):
# before giving up on a missing registration, wait, so the restarts are minutes apart.
give_up() {
  echo "pool agent: $*" >&2
  sleep "${POOL_RETRY_SECONDS:-120}"
  exit 78
}

as_pool() {  # CMD...: run CMD as POOL_UID, with the docker socket's group when there is one
  if [ "$(id -u)" != 0 ]; then exec "$@"; fi
  groups="$POOL_GID"
  if [ -S "$SOCK" ]; then groups="$groups,$(stat -c %g "$SOCK")"; fi
  HOME="$STATE" exec setpriv --reuid="$POOL_UID" --regid="$POOL_GID" --groups="$groups" --inh-caps=-all -- "$@"
}

own() {  # DIR...: create each folder (not recursively chowned: a cache can be large) for POOL_UID
  if [ "$(id -u)" != 0 ]; then mkdir -p "$@"; return; fi
  for dir in "$@"; do
    mkdir -p "$dir"
    chown "$POOL_UID:$POOL_GID" "$dir"
  done
}

prepare_agent() {
  # The work folder and the job sockets must be at the same path on the host and here: job containers
  # are started by the host's Docker, which mounts them from the host (the compose files and
  # install.sh mount $POOL_STATE at the same path).
  own "$STATE" "$STATE/work" "$STATE/run" "$(dirname "$CONF")"
  chmod 0700 "$(dirname "$CONF")" "$STATE/run" 2> /dev/null || true
  if [ "$(id -u)" = 0 ] && [ -f "$CONF" ]; then chown "$POOL_UID:$POOL_GID" "$CONF"; fi
  export POOL_SOCKET_DIR="$STATE/run"
  if [ ! -S "$SOCK" ]; then
    give_up "$SOCK is not mounted: run this container with -v /var/run/docker.sock:/var/run/docker.sock"
  fi
}

case "$(basename "$0")" in  # the image's /usr/local/bin/pool-agent and /usr/local/bin/pool link here
  pool-agent) set -- pool-agent "$@" ;;
  pool) set -- pool "$@" ;;
esac
cmd="${1:-agent}"
[ $# -eq 0 ] || shift
case "$cmd" in
  server)
    own "${POOL_DATA:-/data}"
    as_pool python -m pool.cli server --data "${POOL_DATA:-/data}" "$@"
    ;;
  agent)
    prepare_agent
    if [ ! -f "$CONF" ]; then
      if [ -z "${POOL_CODE:-}" ] || [ -z "${POOL_SERVER:-}" ]; then
        give_up "not registered ($CONF is missing): set POOL_SERVER and POOL_CODE (from \`pool machine add\`)"
      fi
      name="${POOL_NAME:-$(hostname -s 2> /dev/null || hostname)}"
      ( as_pool python -m pool.cli agent register --server "$POOL_SERVER" --code "$POOL_CODE" --name "$name" ) \
        || give_up "registration failed (a code works once: get a new one with \`pool machine add\`)"
    fi
    as_pool python -m pool.cli agent run --work "$STATE/work" "$@"
    ;;
  register)
    prepare_agent
    as_pool python -m pool.cli agent register "$@"
    ;;
  pool-agent)
    export POOL_SOCKET_DIR="$STATE/run"
    as_pool python -m pool.cli agent "$@"
    ;;
  pool)
    as_pool python -m pool.cli "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
