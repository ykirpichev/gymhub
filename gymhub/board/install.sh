#!/bin/sh
# Human Gym worker installer. Get a code on the board's Compute page ("Add a machine"), then:
#
#   curl -fsSL __BOARD__/install.sh | sh -s -- --code ABCD-1234
#
# Installs into ~/gymhub-worker (override with --dir), registers this machine with the board, and
# starts the runner as a systemd user service when possible. Needs Python 3.10+, curl and tar.
# Nothing is installed system-wide and nothing needs root.
#
# Options: --code CODE (required)  --dir DIR  --cpu-only  --no-service  --no-deps
set -eu

BOARD="__BOARD__"
CODE=""
DIR="$HOME/gymhub-worker"
CPU_ONLY=""
SERVICE=1
DEPS=1

while [ $# -gt 0 ]; do
  case "$1" in
    --code) CODE="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --no-service) SERVICE=""; shift ;;
    --no-deps) DEPS=""; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "$CODE" ] || die "pass the code from the board: --code ABCD-1234"
command -v curl > /dev/null || die "curl is required"
command -v tar > /dev/null || die "tar is required"
PYTHON=$(command -v python3 || true)
[ -n "$PYTHON" ] || die "python3 (3.10 or newer) is required"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "python3 must be 3.10 or newer"
if [ -n "$DEPS" ] && ! "$PYTHON" -c 'import ensurepip, venv' > /dev/null 2>&1; then
  PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  die "python3 cannot create virtual environments here. On Ubuntu/Debian run:  sudo apt install python${PYVER}-venv  (then run this command again)"
fi

GPU=""
if [ -z "$CPU_ONLY" ] && command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
fi

say "Downloading the worker into $DIR"
mkdir -p "$DIR"
BUNDLE="$DIR/.bundle.tar.gz"
# No -L: a board behind a sign-in proxy (IAP) answers with a redirect to a login page, not the bundle.
STATUS=$(curl -sS -o "$BUNDLE" -w '%{http_code}' "$BOARD/v1/register/bundle.tar.gz?code=$CODE") || die "could not reach $BOARD"
case "$STATUS" in
  200) ;;
  30*) die "the board redirected to a sign-in page: it is behind IAP, so machines cannot register. The board's admin must switch it to in-app sign-in (AUTH_MODE=app)." ;;
  403) die "the code is not valid (codes work once and expire after 10 minutes). Create a new one on the board's Compute page." ;;
  *) die "the board answered HTTP $STATUS when downloading the worker" ;;
esac
tar -xzf "$BUNDLE" -C "$DIR" || die "the download is not a worker bundle"
rm -f "$BUNDLE"

cd "$DIR"
if [ -n "$DEPS" ]; then
  say "Creating the Python environment"
  "$PYTHON" -m venv .venv
  .venv/bin/pip install -q -U pip
  if [ -n "$GPU" ]; then
    say "Installing PyTorch with CUDA for $GPU (a few GB, this takes a while)"
    .venv/bin/pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
    .venv/bin/pip install -q triton ninja numpy || say "Triton could not be installed; the Triton exercises will not run here"
  else
    say "No NVIDIA GPU found: installing the CPU-only environment (algorithm exercises)"
    .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || true
  fi
  .venv/bin/pip install -q -e . rich
fi

say "Registering this machine"
REPLY=$(curl -fsS -X POST "$BOARD/v1/register" -H 'Content-Type: application/json' \
  -d "{\"code\": \"$CODE\", \"host\": \"$(hostname)\"}") || die "registration failed. Create a new code on the board."
TOKEN=$(printf '%s' "$REPLY" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["token"])')
NAME=$(printf '%s' "$REPLY" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["name"])')
mkdir -p "$HOME/.config/gymhub"
umask 077
printf '%s\n' "$TOKEN" > "$HOME/.config/gymhub/worker-token"
umask 022

# This machine runs its owner's jobs directly. Other people's code runs only on the compute pool
# (gymhub/runner/pooled.py), never here: the runner has no sandbox of its own.
START="$DIR/.venv/bin/python -m gymhub.runner --api $BOARD --name $NAME"
if [ -n "$SERVICE" ] && command -v systemctl > /dev/null && systemctl --user show-environment > /dev/null 2>&1; then
  say "Installing the systemd user service gymhub-worker"
  mkdir -p "$HOME/.config/systemd/user"
  cat > "$HOME/.config/systemd/user/gymhub-worker.service" << EOF
[Unit]
Description=Human Gym worker ($NAME)
After=network-online.target

[Service]
WorkingDirectory=$DIR
ExecStart=$START
Restart=always
RestartSec=10
# 78: the board revoked this machine's token. Restarting would only be refused again.
RestartPreventExitStatus=78
KillMode=control-group
TimeoutStopSec=30
UnsetEnvironment=GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_PROJECT CLOUDSDK_CONFIG GYMHUB_STORE GYMHUB_WORKER_TOKEN
NoNewPrivileges=yes
UMask=0077
LockPersonality=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
SystemCallArchitectures=native

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now gymhub-worker > /dev/null
  loginctl enable-linger "$(id -un)" > /dev/null 2>&1 || true
  say "Done. $NAME is starting; it appears on the board's Compute page within a minute."
  echo "    Logs: journalctl --user -u gymhub-worker -f"
else
  say "Done. Start the worker with:"
  echo "    cd $DIR && $START"
fi
