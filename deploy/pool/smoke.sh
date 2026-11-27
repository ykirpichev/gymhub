#!/bin/sh
# Smoke test of the pool in Docker (CI: .github/workflows/pool-images.yml): a pool server container,
# an agent container started from deploy/pool/compose.agent.yml, and one CPU job that must come back
# with its output, run as the agent's uid. Needs Docker, compose and python3 (for the pool CLI).
#
#   deploy/pool/smoke.sh pool:smoke          (an image built from deploy/pool/Dockerfile)
#
# It uses /etc/pool and /var/lib/pool on this machine, as the agent's container does: run it on a CI
# runner or a scratch machine, not on a pool machine.
set -eu
IMAGE="${1:?usage: deploy/pool/smoke.sh IMAGE}"
JOB_IMAGE="${SMOKE_JOB_IMAGE:-python:3.12-slim}"
PORT="${SMOKE_PORT:-18765}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
export PYTHONPATH="$REPO" POOL_CONFIG_DIR="$WORK/member"
server=pool-smoke-server
compose() { POOL_IMAGE="$IMAGE" docker compose -f "$REPO/deploy/pool/compose.agent.yml" "$@"; }

cleanup() {
  status=$?
  if [ "$status" != 0 ]; then
    echo "--- server log"; docker logs "$server" 2>&1 | tail -n 30 || true
    echo "--- agent log"; docker logs pool-agent 2>&1 | tail -n 30 || true
  fi
  compose down > /dev/null 2>&1 || true
  docker rm -f "$server" > /dev/null 2>&1 || true
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT

echo "==> pool server ($IMAGE server)"
docker rm -f "$server" > /dev/null 2>&1 || true
docker run -d --name "$server" -p "127.0.0.1:$PORT:8765" "$IMAGE" server --public-url "http://127.0.0.1:$PORT" > /dev/null
for _ in $(seq 30); do
  code=$(docker logs "$server" 2>&1 | sed -n 's/.*--code \([^ ]*\).*/\1/p' | head -n 1)
  [ -z "$code" ] || break
  sleep 1
done
[ -n "$code" ] || { echo "the server printed no admin code" >&2; exit 1; }
python3 -m pool.cli login "http://127.0.0.1:$PORT" --code "$code" --name smoke

echo "==> agent (compose.agent.yml, $IMAGE agent)"
machine_code=$(python3 -m pool.cli machine add --name smoke-agent | sed -n 's/.*agent register .*--code \([^ ]*\).*/\1/p' | head -n 1)
[ -n "$machine_code" ] || { echo "pool machine add printed no code" >&2; exit 1; }
docker pull -q "$JOB_IMAGE" > /dev/null
POOL_SERVER="http://127.0.0.1:$PORT" POOL_CODE="$machine_code" POOL_NAME=smoke-agent POOL_UID=12345 POOL_GID=12345 \
  compose up -d --no-build
for _ in $(seq 60); do
  python3 -m pool.cli machine ls | grep -q 'smoke-agent .* online' && break
  sleep 1
done
python3 -m pool.cli machine ls

echo "==> a job"
mkdir -p "$WORK/job"
cat > "$WORK/job/main.py" << 'EOF'
import os
print("uid", os.getuid(), "socket", os.path.exists("/run/pool/pool.sock"))
open(os.path.join(os.environ["POOL_OUTPUTS"], "out.txt"), "w").write("smoke ok\n")
EOF
cd "$WORK/job"
# --mem 1: CI runners have less than the default 8 GB.
timeout 300 python3 -m pool.cli run --mem 1 --image "$JOB_IMAGE" -- python main.py | tee "$WORK/run.log"
grep -q "uid 12345 socket True" "$WORK/run.log"
grep -q "smoke ok" pool-outputs/*/out.txt
docker exec pool-agent pool-agent status | grep -q "running in a container"
echo "PASS: server, agent and a job, all in containers"
