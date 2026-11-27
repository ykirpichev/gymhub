#!/usr/bin/env sh
# Build a grader image on this machine, under the tag the pool expects. Run from the repo root:
#   sh deploy/grader/build-local.sh l4t-r36    on a Jetson (JetPack 6 / L4T r36)
#   sh deploy/grader/build-local.sh cuda       on a DGX Spark, or any machine with an NVIDIA GPU
#   sh deploy/grader/build-local.sh cpu        anywhere, for CPU gyms
# The images CI publishes to ghcr.io are private and pool machines have no ghcr login, so build the
# GPU image a machine needs here (#163). The agent finds the image locally and does not pull it; a
# machine with neither fails the job with "ImageUnavailable" (gymhub/runner/pooled.py).
set -eu
FLAVOR="${1:-cuda}"
case "$FLAVOR" in cpu|cuda) FILE="deploy/grader/Dockerfile.$FLAVOR" ;; l4t-r36) FILE=deploy/grader/Dockerfile.l4t ;; *) echo "flavor: cpu, cuda or l4t-r36" >&2; exit 2 ;; esac
IMAGE="${GYMHUB_GRADER_IMAGE:-ghcr.io/ykirpichev/gymhub/grader}"
docker build -f "$FILE" -t "$IMAGE:$FLAVOR" .
echo "built $IMAGE:$FLAVOR"
