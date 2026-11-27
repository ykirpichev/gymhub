#!/usr/bin/env bash
# Install the runner as a systemd user service, so it starts at login and restarts if it crashes.
#
# Default: worker-token mode. Set RUNNER_API in deploy/gcp.env to the board's worker URL (behind IAP,
# the gymhub-workers service that `deploy/gcp.sh deploy` prints) and put a worker token from the
# Compute page in ~/.config/gymhub/worker-token. The machine needs no Google login at all.
#
# RUNNER_DIRECT=1 keeps the old direct mode (Firestore with this machine's application-default
# login). That login can read and write the whole database, so use it only on a machine you trust
# with the project.
set -euo pipefail
cat >&2 << 'EOF'
DEPRECATED: the gymhub runner that executes jobs on this machine is being replaced by the pool agent
(docs/design/worker-v2.md). To add a GPU machine, use "Add a machine" under Pool machines on the
board's Compute page: it gives a one-line installer (curl ... /install.sh | sudo sh -s -- --code ...)
that needs no checkout of this repository. The runner itself becomes a dispatcher that executes
nothing: python -m gymhub.runner --api <board> --dispatch-only (on any machine, e.g. the pool server).
This script still installs the old runner for now.
EOF
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
source "$HERE/gcp.env"
UNIT="$HOME/.config/systemd/user/gymhub-runner.service"
mkdir -p "$(dirname "$UNIT")"
if [[ -n "${RUNNER_DIRECT:-}" ]]; then
  echo "Direct mode: the runner uses this machine's Google login for Firestore (RUNNER_DIRECT=1)." >&2
  RUNNER_ARGS=""
  ENVIRONMENT="Environment=GYMHUB_STORE=firestore:$PROJECT_ID\nEnvironment=GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
else
  if [[ -z "${RUNNER_API:-}" ]]; then
    echo "Set RUNNER_API in deploy/gcp.env to the board's worker URL (deploy/gcp.sh status prints it)." >&2
    echo "Or set RUNNER_DIRECT=1 to keep direct Firestore mode with this machine's Google login." >&2
    exit 1
  fi
  [[ -f "$HOME/.config/gymhub/worker-token" ]] || { echo "Put the worker token in ~/.config/gymhub/worker-token first (Compute page → Worker tokens)." >&2; exit 1; }
  chmod 600 "$HOME/.config/gymhub/worker-token"
  RUNNER_ARGS="--api $RUNNER_API"
  # Variables a user manager may have imported (systemctl --user import-environment) never reach the runner.
  ENVIRONMENT="# worker-token mode: no cloud credentials in this unit\nUnsetEnvironment=GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_PROJECT CLOUDSDK_CONFIG GYMHUB_STORE GYMHUB_WORKER_TOKEN"
fi
sed -e "s|@REPO@|$REPO|g" -e "s|@RUNNER_ARGS@|$RUNNER_ARGS|g" -e "s|@ENVIRONMENT@|$ENVIRONMENT|g" "$HERE/gymhub-runner.service" > "$UNIT"
systemctl --user daemon-reload
systemctl --user enable --now gymhub-runner
systemctl --user restart gymhub-runner
# Keep running after you log out.
loginctl enable-linger "$USER" 2> /dev/null || echo "Note: run 'sudo loginctl enable-linger $USER' to keep the runner alive after logout."
systemctl --user --no-pager status gymhub-runner | head -5
echo "Logs: journalctl --user -u gymhub-runner -f"
ADC="${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}/application_default_credentials.json"
if [[ -z "${RUNNER_DIRECT:-}" && -f "$ADC" ]]; then
  echo
  echo "This machine still has a Google application-default login ($ADC), which the runner no longer needs"
  echo "and which can read and write the board's whole database. Remove it with:"
  echo "  gcloud auth application-default revoke     (or, without gcloud: rm $ADC)"
fi
