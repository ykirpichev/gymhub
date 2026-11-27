#!/bin/sh
# The dispatcher image's entrypoint (deploy/pool/Dockerfile.dispatcher).
#
#   (no arguments)   python -m gymhub.runner --dispatch-only, with --api $GYMHUB_API when it is set
#   pool CMD         the pool CLI as the dispatcher's member: pool login <server> --code CODE --name dispatcher
#   anything else    run as given
set -eu
# compose passes unset settings as empty strings: drop them, so they read as unset.
for name in GYMHUB_API GYMHUB_WORKER_TOKEN GYMHUB_STORE GYMHUB_POOL POOL_SERVER POOL_TOKEN \
    GYMHUB_GRADER_IMAGE GYMHUB_GRADER_VERSION GYMHUB_POOL_IMAGE; do
  eval "[ -n \"\${$name:-}\" ] || unset $name"
done

mkdir -p /state/cache  # <repo>/.cache links here (deploy/pool/Dockerfile.dispatcher)

missing() {  # what the dispatcher still lacks, or nothing
  if [ -z "${POOL_TOKEN:-}" ] && [ ! -f "${POOL_CONFIG_DIR:-/state/pool}/member.json" ]; then
    echo "a pool member: set POOL_TOKEN, or run: docker compose -f deploy/pool/compose.controller.yml run --rm dispatcher pool login http://pool-server:8765 --code CODE --name dispatcher"
  elif [ -z "${GYMHUB_API:-}" ] && [ -z "${GYMHUB_STORE:-}" ]; then
    echo "a board: set GYMHUB_API (the board's worker URL) and GYMHUB_WORKER_TOKEN (Compute page), or GYMHUB_STORE"
  fi
}

case "${1:-}" in
  ''|--*)
    # Not configured yet: wait and say what is missing, instead of a restart loop.
    while need=$(missing) && [ -n "$need" ]; do
      echo "dispatcher: waiting for $need" >&2
      sleep 30
    done
    exec python -m gymhub.runner --dispatch-only --name "${GYMHUB_RUNNER_NAME:-dispatcher}" "$@" ;;
  pool)
    shift
    exec python -m pool.cli "$@" ;;
  *)
    exec "$@" ;;
esac
