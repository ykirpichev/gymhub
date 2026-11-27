"""The one-command installer the server serves at /install.sh (docs/design/worker-v2.md section 3).

    curl -fsSL https://<server>/install.sh | sudo sh -s -- --code ABCD-1234 [--name NAME] [--yes]

It is a POSIX sh script kept as a string here, so the server can serve it from the same `pool/*.py`
files it ships as the agent package (pool/update.py only allows `pool/<name>.py` in a package).
`render` fills in the server's address and the sha256 of the agent package it serves.

With sudo it installs Docker CE and the NVIDIA container toolkit when they are missing (Ubuntu and
Debian), the server's agent package in /opt/pool and its settings in /etc/pool/agent.json (0600).
By default the agent then runs in its own container, `pool-agent` (deploy/pool/Dockerfile, restarted
by Docker), which drives the host's Docker through its socket, as the user who ran sudo; the host
needs no Python. `--host-agent` runs it as a system unit, pool-agent.service, on the host's Python
instead. Then it runs a smoke test container: no network, read-only root, the GPU works.
`--no-sudo` keeps the oldest install: the agent in ~/.local/share/pool-agent and a user unit.

Running it again repairs or upgrades the install; it keeps the registration unless --reregister.
The owner then has `pool-agent status | pause | resume | uninstall` (pool/agent.py).
"""

from __future__ import annotations

import os
import re

# Finding the older --no-sudo agent of a user: a `pool.cli agent run` process whose PYTHONPATH,
# command line or working folder is that install's folder (~/.local/share/pool-agent). Agents run from
# anywhere else (a checkout's .venv, say) are not touched. PROC is /proc; tests point it elsewhere.
OLD_AGENTS_SH = r"""
PROC="${POOL_INSTALL_PROC:-/proc}"

old_agent() {  # PID DIR: true when process PID is `pool.cli agent run` from the install in DIR
  cmdline=$(tr '\0' ' ' < "$PROC/$1/cmdline" 2> /dev/null || true)
  case "$cmdline" in
    *"pool.cli agent run"*) ;;
    *) return 1 ;;
  esac
  pythonpath=$(tr '\0' '\n' < "$PROC/$1/environ" 2> /dev/null | sed -n 's/^PYTHONPATH=//p' | head -n 1)
  case ":$pythonpath:" in
    *":$2:"*|*":$2/:"*) return 0 ;;
  esac
  case "$cmdline" in
    *"$2/"*|*"$2 "*) return 0 ;;
  esac
  [ "$(readlink "$PROC/$1/cwd" 2> /dev/null || true)" = "$2" ]
}

old_agent_pids() {  # USER DIR: the PIDs of USER's agents run from DIR, one per line
  for pid in $(pgrep -u "$1" -f 'pool.cli agent run' 2> /dev/null || true); do
    if old_agent "$pid" "$2"; then echo "$pid"; fi
  done
}

stop_old_agents() {  # USER DIR: SIGTERM (running jobs stop and requeue), then SIGKILL after 30 s
  pids=$(old_agent_pids "$1" "$2" | tr '\n' ' ')
  [ -n "${pids% }" ] || return 0
  for pid in $pids; do
    printf '==> Older agent: stopping pid %s: %s\n' "$pid" "$(tr '\0' ' ' < "$PROC/$pid/cmdline" 2> /dev/null || true)"
    kill "$pid" 2> /dev/null || true
  done
  waited=0
  while [ "$waited" -lt 30 ]; do
    alive=''
    for pid in $pids; do
      if kill -0 "$pid" 2> /dev/null; then alive="$alive $pid"; fi
    done
    [ -n "$alive" ] || break
    sleep 1
    waited=$((waited + 1))
  done
  for pid in $alive; do
    kill -9 "$pid" 2> /dev/null || true
    printf '    pid %s did not stop in 30 s: killed\n' "$pid"
  done
  printf '    stopped: %s\n' "${pids% }"
}
"""

INSTALL_SH = r"""#!/bin/sh
# shellcheck disable=SC1091,SC2016
# Adds this computer to the pool at __SERVER__.
#
#   curl -fsSL __SERVER__/install.sh | sudo sh -s -- --code CODE [--name NAME] [--yes]
#   curl -fsSL __SERVER__/install.sh | sh -s -- --no-sudo --code CODE      (a user service, no root)
#
# Options:
#   --code CODE        the machine code from `pool machine add` (once; not needed to repair or upgrade)
#   --name NAME        the machine's name in the pool (default: the host name)
#   --yes              do not ask before changing the system
#   --no-sudo          install for this user only: ~/.local/share/pool-agent and a user unit
#   --host-agent       run the agent on the host's Python as pool-agent.service, not in a container
#   --image IMAGE      the agent's container image (default __AGENT_IMAGE__)
#   --user USER        the user the agent runs as (default: the one who ran sudo, else a `pool` user)
#   --sha256 HEX       refuse unless the agent package has this sha256 (`pool machine add` prints it)
#   --warm-image IMG   pull this image (the grader image) in the background at the end
#   --reregister       register again with --code even if this machine is registered already
#
# Running it again repairs or upgrades the install. Remove it with: sudo pool-agent uninstall
set -eu

SERVER='__SERVER__'
PACKAGE_SHA256='__SHA256__'
SMOKE_IMAGE='ubuntu:24.04'
AGENT_IMAGE='__AGENT_IMAGE__'
CONTAINER=pool-agent
PREFIX=/opt/pool
ETC=/etc/pool
STATE=/var/lib/pool
UNIT=/etc/systemd/system/pool-agent.service
WRAPPER=/usr/local/bin/pool-agent

CODE=''
NAME=''
YES=''
NO_SUDO=''
RUN_AS=''
WANT_SHA=''
WARM_IMAGE=''
REREGISTER=''
HOST_AGENT=''

say() { printf '%s\n' "$*"; }
step() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
usage() { say "usage: curl -fsSL $SERVER/install.sh | sudo sh -s -- --code CODE [--name NAME] [--yes] [--no-sudo] [--host-agent] [--image IMAGE] [--user USER] [--sha256 HEX] [--warm-image IMAGE] [--reregister]"; }
have() { command -v "$1" > /dev/null 2>&1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --code) [ $# -ge 2 ] || die "--code needs a value"; CODE="$2"; shift 2 ;;
    --code=*) CODE="${1#*=}"; shift ;;
    --name) [ $# -ge 2 ] || die "--name needs a value"; NAME="$2"; shift 2 ;;
    --name=*) NAME="${1#*=}"; shift ;;
    --user) [ $# -ge 2 ] || die "--user needs a value"; RUN_AS="$2"; shift 2 ;;
    --user=*) RUN_AS="${1#*=}"; shift ;;
    --sha256) [ $# -ge 2 ] || die "--sha256 needs a value"; WANT_SHA="$2"; shift 2 ;;
    --sha256=*) WANT_SHA="${1#*=}"; shift ;;
    --warm-image) [ $# -ge 2 ] || die "--warm-image needs a value"; WARM_IMAGE="$2"; shift 2 ;;
    --warm-image=*) WARM_IMAGE="${1#*=}"; shift ;;
    --yes|-y) YES=1; shift ;;
    --no-sudo) NO_SUDO=1; shift ;;
    --host-agent) HOST_AGENT=1; shift ;;
    --image) [ $# -ge 2 ] || die "--image needs a value"; AGENT_IMAGE="$2"; shift 2 ;;
    --image=*) AGENT_IMAGE="${1#*=}"; shift ;;
    --reregister) REREGISTER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option $1 (see the top of $SERVER/install.sh)" ;;
    *)  # the older form: install.sh CODE [NAME]
      if [ -z "$CODE" ]; then CODE="$1"; elif [ -z "$NAME" ]; then NAME="$1"; else die "unexpected argument $1"; fi
      shift ;;
  esac
done

[ -n "$NAME" ] || NAME=$(hostname -s 2> /dev/null || hostname)
case "$NAME" in
  *[!A-Za-z0-9._-]*|'') die "machine name '$NAME': letters, digits, '.', '_' and '-' only (pass --name)" ;;
esac
if [ -n "$WANT_SHA" ] && [ "$WANT_SHA" != "$PACKAGE_SHA256" ]; then
  die "this server serves agent package $PACKAGE_SHA256, not the $WANT_SHA you expected"
fi

fetch() {  # URL FILE
  if have curl; then curl -fsSL "$1" -o "$2"; elif have wget; then wget -qO "$2" "$1"; else die "needs curl or wget"; fi
}

sha256_of() {
  if have sha256sum; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

python_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2> /dev/null; }

install_package() {  # DIR: download the agent, check its sha256, unpack it to DIR/pool
  tmp=$(mktemp -d)
  fetch "$SERVER/v1/agent/package.tar.gz" "$tmp/agent.tar.gz" || die "could not download the agent from $SERVER"
  got=$(sha256_of "$tmp/agent.tar.gz")
  if [ "$got" != "$PACKAGE_SHA256" ]; then
    rm -rf "$tmp"
    die "the agent package's sha256 is $got, expected $PACKAGE_SHA256: not installing it"
  fi
  mkdir -p "$tmp/x"
  tar -xzf "$tmp/agent.tar.gz" -C "$tmp/x"
  [ -f "$tmp/x/pool/agent.py" ] || die "the agent package has no pool/agent.py"
  mkdir -p "$1"
  rm -rf "$1/pool.new"
  mv "$tmp/x/pool" "$1/pool.new"
  rm -rf "$1/pool"
  mv "$1/pool.new" "$1/pool"
  chmod -R a+rX "$1"
  rm -rf "$tmp"
}

__OLD_AGENTS__

config_field() {  # CONFIG FIELD: a string field of agent.json (json.dumps with indent: one field a line), or nothing
  sed -n "s/^ *\"$2\": *\"\([^\"]*\)\".*/\1/p" "$1" 2> /dev/null | head -n 1
}
config_name() { config_field "$1" name; }

# ---- --no-sudo: this user only (the older install) ------------------------------------------------
if [ -n "$NO_SUDO" ]; then
  PY=$(command -v python3) || die "pool needs Python 3.10 or newer (python3)"
  python_ok "$PY" || die "pool needs Python 3.10 or newer; $PY is older"
  have docker || warn "Docker is not installed; jobs run in containers (GPUs also need the NVIDIA container toolkit)"
  DIR="$HOME/.local/share/pool-agent"
  CONF="${POOL_CONFIG_DIR:-$HOME/.config/pool}/agent.json"
  step "Agent: $DIR (sha256 $PACKAGE_SHA256)"
  install_package "$DIR"
  if [ -f "$CONF" ] && [ -z "$REREGISTER" ]; then
    step "Register: already registered as $(config_name "$CONF") ($CONF)"
  else
    [ -n "$CODE" ] || die "no --code: get one with \`pool machine add\` on a computer logged in to the pool"
    PYTHONPATH="$DIR" "$PY" -m pool.cli agent register --server "$SERVER" --code "$CODE" --name "$NAME"
  fi
  mkdir -p "$HOME/.local/bin"
  cat > "$HOME/.local/bin/pool-agent" << EOF
#!/bin/sh
# pool-agent status | pause | resume | uninstall   (written by $SERVER/install.sh --no-sudo)
PYTHONPATH="$DIR" exec "$PY" -m pool.cli agent "\$@"
EOF
  chmod 0755 "$HOME/.local/bin/pool-agent"
  if have systemctl && systemctl --user show-environment > /dev/null 2>&1; then
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/pool-agent.service" << EOF
[Unit]
Description=pool agent
After=network-online.target

[Service]
Environment=PYTHONPATH=$DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PY -m pool.cli agent run
Restart=always
RestartSec=5
RestartPreventExitStatus=78

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable pool-agent.service > /dev/null 2>&1
    systemctl --user restart pool-agent.service
    say "The agent runs as a user service (systemctl --user status pool-agent)."
    say "To keep it running after you log out: sudo loginctl enable-linger $(id -un)"
  else
    say "Start the agent with:  PYTHONPATH=$DIR $PY -m pool.cli agent run"
  fi
  say "Owner commands: ~/.local/bin/pool-agent status | pause | resume | uninstall"
  exit 0
fi

# ---- 1. preflight ---------------------------------------------------------------------------------
[ "$(id -u)" = 0 ] || die "run it with sudo (curl -fsSL $SERVER/install.sh | sudo sh -s -- --code CODE), or add --no-sudo for a user service"
[ "$(uname -s)" = Linux ] || die "the system install is for Linux; on $(uname -s) use --no-sudo"
if [ -n "$HOST_AGENT" ]; then
  have systemctl && [ -d /run/systemd/system ] || die "this machine does not run systemd; drop --host-agent (Docker restarts the agent's container), or use --no-sudo"
fi

OS_ID=$( (. /etc/os-release 2> /dev/null && echo "${ID:-}") || true)
OS_LIKE=$( (. /etc/os-release 2> /dev/null && echo "${ID_LIKE:-}") || true)
OS_VERSION=$( (. /etc/os-release 2> /dev/null && echo "${VERSION_ID:-}") || true)
OS_CODENAME=$( (. /etc/os-release 2> /dev/null && echo "${VERSION_CODENAME:-}") || true)
OS_PRETTY=$( (. /etc/os-release 2> /dev/null && echo "${PRETTY_NAME:-}") || echo unknown)
APT_DISTRO=''
case "$OS_ID" in
  ubuntu|debian) APT_DISTRO="$OS_ID" ;;
  *)
    case " $OS_LIKE " in
      *" ubuntu "*) APT_DISTRO=ubuntu; OS_CODENAME=$( (. /etc/os-release && echo "${UBUNTU_CODENAME:-$OS_CODENAME}") || true) ;;
      *" debian "*) APT_DISTRO=debian ;;
    esac ;;
esac
have apt-get || APT_DISTRO=''

ARCH=$(uname -m)
case "$ARCH" in
  x86_64|aarch64) ;;
  *) die "unsupported architecture $ARCH (x86_64 and aarch64 only)" ;;
esac

JETSON=''
JETSON_RELEASE=''
if [ -f /etc/nv_tegra_release ]; then
  JETSON=1
  JETSON_RELEASE=$(head -n 1 /etc/nv_tegra_release | sed 's/^# *//; s/, *GCID.*//')
fi

GPU=''
GPU_DESC='none (CPU jobs only)'
if have nvidia-smi && nvidia-smi -L > /dev/null 2>&1; then
  GPU=1
  GPU_COUNT=$(nvidia-smi -L | grep -c '^GPU' || true)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -n 1 || true)
  GPU_DESC="$GPU_COUNT GPU(s), driver ${DRIVER:-unknown}"
elif [ -n "$JETSON" ]; then
  GPU=1
  GPU_DESC="Jetson integrated GPU"
elif have lspci && lspci 2> /dev/null | grep -qi 'nvidia'; then
  die "this machine has an NVIDIA GPU but no working driver (nvidia-smi fails): install the NVIDIA driver, reboot, and run this again"
fi
[ -z "$JETSON" ] || GPU_DESC="$GPU_DESC; Jetson $JETSON_RELEASE"

DISK_GB=$(df -Pk / | awk 'NR==2 {printf "%d", $4 / 1048576}')
[ "$DISK_GB" -ge 5 ] || die "only $DISK_GB GB free on /: jobs and images need at least 5 GB (20+ recommended)"

PY=$(command -v python3 || true)
NEED_PYTHON=''
if [ -n "$HOST_AGENT" ] && { [ -z "$PY" ] || ! python_ok "$PY"; }; then
  [ -n "$APT_DISTRO" ] || die "pool needs Python 3.10 or newer (python3)"
  NEED_PYTHON=1
fi

if [ -z "$RUN_AS" ]; then
  RUN_AS="${SUDO_USER:-}"
  [ "$RUN_AS" != root ] || RUN_AS=''
  [ -n "$RUN_AS" ] || RUN_AS=pool
fi
NEW_USER=''
id "$RUN_AS" > /dev/null 2>&1 || NEW_USER=1
RUN_HOME=''
[ -n "$NEW_USER" ] || RUN_HOME=$(getent passwd "$RUN_AS" | cut -d: -f6)

# The older --no-sudo install of the same user (a user unit, or a `setsid nohup` process): this install
# takes over its registration and stops it, so two agents never share one machine token.
OLD_DIR="${RUN_HOME:-/nonexistent}/.local/share/pool-agent"
OLD_CONF="${RUN_HOME:-/nonexistent}/.config/pool/agent.json"
OLD_UNIT="${RUN_HOME:-/nonexistent}/.config/systemd/user/pool-agent.service"
OLD_PIDS=$(old_agent_pids "$RUN_AS" "$OLD_DIR" | tr '\n' ' ')

NEED_DOCKER=''
have docker || NEED_DOCKER=1
if [ -n "$NEED_DOCKER" ] && [ -z "$APT_DISTRO" ]; then
  die "Docker is not installed and this installer only installs it on Ubuntu or Debian ($OS_PRETTY): install Docker (https://docs.docker.com/engine/install/), then run this again"
fi
has_toolkit() { have nvidia-ctk || have nvidia-container-runtime || have nvidia-container-toolkit; }
NEED_TOOLKIT=''
if [ -n "$GPU" ] && ! has_toolkit; then
  [ -n "$APT_DISTRO" ] || die "the NVIDIA container toolkit is not installed and this installer only installs it on Ubuntu or Debian: install it (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), then run this again"
  NEED_TOOLKIT=1
fi

case "$OS_ID:$OS_VERSION" in
  ubuntu:22.04|ubuntu:24.04|debian:12) ;;
  *) warn "$OS_PRETTY is not tested yet (Ubuntu 22.04/24.04 and Debian 12 are); going on" ;;
esac

CONF="$ETC/agent.json"
REGISTERED=''
MIGRATE=''
[ -f "$CONF" ] && [ -z "$REREGISTER" ] && REGISTERED=1
if [ -z "$REGISTERED" ] && [ -z "$REREGISTER" ] && [ -f "$OLD_CONF" ]; then
  MIGRATE=1  # keep the older install's machine (its name, history and cache) instead of a new one
  REGISTERED=1
fi
if [ -z "$REGISTERED" ] && [ -z "$CODE" ]; then
  die "no --code: get one with \`pool machine add\` on a computer logged in to the pool"
fi

step "Preflight: $OS_PRETTY, $ARCH, GPU: $GPU_DESC, $DISK_GB GB free"

# ---- the plan, and a confirmation ---------------------------------------------------------------
say "This will:"
[ -z "$NEED_PYTHON" ] || say "  - install python3 (apt)"
[ -z "$NEED_DOCKER" ] || say "  - install Docker CE from download.docker.com"
[ -z "$NEED_TOOLKIT" ] || say "  - install the NVIDIA container toolkit and configure Docker for it"
say "  - install the agent from $SERVER in $PREFIX (sha256 $PACKAGE_SHA256)"
[ -n "$HOST_AGENT" ] || say "  - pull the agent's image $AGENT_IMAGE"
if [ -n "$MIGRATE" ]; then
  say "  - move the older install's registration ($OLD_CONF) to $CONF${CODE:+ (the code stays unused)}"
elif [ -n "$REGISTERED" ]; then
  say "  - keep this machine's registration ($CONF)"
else
  say "  - register this machine as $NAME, settings in $CONF"
fi
[ -z "$NEW_USER" ] || say "  - create a system user $RUN_AS"
[ ! -f "$OLD_UNIT" ] || say "  - stop and disable $RUN_AS's older user unit pool-agent.service"
[ -z "$OLD_PIDS" ] || say "  - stop the older agent process(es) of $RUN_AS run from $OLD_DIR: $OLD_PIDS"
if [ -n "$HOST_AGENT" ]; then
  say "  - run the agent as pool-agent.service, user $RUN_AS (added to the docker group: it can run containers),"
  say "    replacing the agent's container $CONTAINER if there is one"
else
  say "  - run the agent in the container $CONTAINER (restarted by Docker) as $RUN_AS's uid, with access to Docker"
  [ ! -f "$UNIT" ] || say "  - stop and disable the host agent pool-agent.service (the container replaces it)"
fi
say "  - run a smoke test container (no network, read-only, GPU)"
if [ -z "$YES" ] && ( exec < /dev/tty ) 2> /dev/null; then
  printf 'Continue? [y/N] ' > /dev/tty
  read -r answer < /dev/tty || answer=''
  case "$answer" in
    y|Y|yes|YES) ;;
    *) die "stopped; nothing was changed" ;;
  esac
fi

apt_ready=''
apt_install() {
  if [ -z "$apt_ready" ]; then
    apt-get update -q > /dev/null
    apt_ready=1
  fi
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$@" > /dev/null
}

if [ -n "$NEED_PYTHON" ]; then
  apt_install python3
  PY=$(command -v python3)
  python_ok "$PY" || die "pool needs Python 3.10 or newer; this system's python3 is older"
fi

# ---- 2. Docker and the NVIDIA container toolkit --------------------------------------------------
if [ -n "$NEED_DOCKER" ]; then
  step "Docker: installing Docker CE"
  apt_install ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  fetch "https://download.docker.com/linux/$APT_DISTRO/gpg" /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$APT_DISTRO $OS_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt_ready=''
  apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi
systemctl enable --now docker > /dev/null 2>&1 || warn "could not start docker.service"
docker info > /dev/null 2>&1 || die "Docker is installed but not running (systemctl status docker)"
[ -n "$NEED_DOCKER" ] || step "Docker: $(docker --version 2> /dev/null || echo present)"

docker_has_nvidia() { docker info --format '{{json .Runtimes}}' 2> /dev/null | grep -q nvidia; }
if [ -n "$GPU" ]; then
  if [ -n "$NEED_TOOLKIT" ]; then
    step "NVIDIA container toolkit: installing"
    apt_install ca-certificates curl gnupg
    if [ -n "$JETSON" ] && apt_install nvidia-container-toolkit 2> /dev/null; then
      :  # JetPack's own apt repository has it: keep to it on a Jetson
    else
      fetch https://nvidia.github.io/libnvidia-container/gpgkey /tmp/nvidia-container-toolkit.gpg
      gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg < /tmp/nvidia-container-toolkit.gpg
      rm -f /tmp/nvidia-container-toolkit.gpg
      fetch https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list /tmp/nvidia-container-toolkit.list
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        /tmp/nvidia-container-toolkit.list > /etc/apt/sources.list.d/nvidia-container-toolkit.list
      rm -f /tmp/nvidia-container-toolkit.list
      apt_ready=''
      apt_install nvidia-container-toolkit
    fi
  fi
  if docker_has_nvidia; then
    step "NVIDIA container toolkit: Docker has the nvidia runtime"
  elif have nvidia-ctk; then
    step "NVIDIA container toolkit: configuring Docker (restarts docker)"
    nvidia-ctk runtime configure --runtime=docker > /dev/null
    systemctl restart docker
    docker_has_nvidia || warn "Docker still lists no nvidia runtime (docker info): GPU jobs may fail"
  else
    warn "the NVIDIA container toolkit has no nvidia-ctk; configure Docker's nvidia runtime by hand"
  fi
fi

# ---- the user the agent runs as -----------------------------------------------------------------
if [ -n "$NEW_USER" ]; then
  useradd --system --home-dir "$STATE" --shell /usr/sbin/nologin "$RUN_AS"
fi
RUN_GROUP=$(id -gn "$RUN_AS")
RUN_UID=$(id -u "$RUN_AS")
RUN_GID=$(id -g "$RUN_AS")
if [ -n "$HOST_AGENT" ]; then  # the agent's container gets the socket's group by itself
  getent group docker > /dev/null 2>&1 || groupadd --system docker
  usermod -aG docker "$RUN_AS"
fi

# ---- 3. the agent ----------------------------------------------------------------------------------
step "Agent: $PREFIX (sha256 $PACKAGE_SHA256 checked)"
install -d -m 0755 "$PREFIX"
install_package "$PREFIX"
install -d -m 0700 -o "$RUN_AS" -g "$RUN_GROUP" "$ETC"
install -d -m 0750 -o "$RUN_AS" -g "$RUN_GROUP" "$STATE"
if [ -n "$HOST_AGENT" ]; then
  cat > "$WRAPPER" << EOF
#!/bin/sh
# pool-agent status | pause | resume | uninstall   (written by $SERVER/install.sh --host-agent)
PYTHONPATH=$PREFIX POOL_AGENT_CONFIG=$CONF exec $PY -m pool.cli agent "\$@"
EOF
else
  cat > "$WRAPPER" << EOF
#!/bin/sh
# pool-agent status | pause | resume | logs | uninstall   (written by $SERVER/install.sh)
# The agent runs in the container $CONTAINER: these run inside it, so they need Docker access (sudo).
set -eu
case "\${1:-}" in
  logs) shift; exec docker logs "\$@" $CONTAINER ;;
  uninstall)
    [ "\$(id -u)" = 0 ] || { echo "the agent is installed for the whole system: sudo pool-agent uninstall" >&2; exit 1; }
    case " \$* " in
      *" --yes "*|*" -y "*) ;;
      *) printf 'Remove this machine from the pool at %s and delete the agent? [y/N] ' '$SERVER'
         read -r answer || answer=''
         case "\$answer" in y|Y|yes|YES) ;; *) echo "nothing changed"; exit 1 ;; esac ;;
    esac
    if [ "\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2> /dev/null)" = true ]; then
      docker exec $CONTAINER pool-agent uninstall --yes || true
    else  # unregister from a one-off container of the same image
      docker run --rm --network host -v /var/run/docker.sock:/var/run/docker.sock -v $ETC:$ETC -v $STATE:$STATE \\
        -v $PREFIX:/opt/pool:ro -e POOL_UID=$RUN_UID -e POOL_GID=$RUN_GID $AGENT_IMAGE pool-agent uninstall --yes || true
    fi
    docker rm -f $CONTAINER > /dev/null 2>&1 || true
    rm -rf $ETC $STATE $PREFIX $WRAPPER
    echo "removed the container $CONTAINER, $ETC, $STATE, $PREFIX and $WRAPPER; Docker and the image $AGENT_IMAGE stay" ;;
  *) exec docker exec $CONTAINER pool-agent "\$@" ;;
esac
EOF
fi
chmod 0755 "$WRAPPER"

# ---- the older --no-sudo agent of the same user: stop it before this one starts -------------------
if [ -f "$OLD_UNIT" ]; then
  uid=$(id -u "$RUN_AS")
  if systemctl --user -M "$RUN_AS@" disable --now pool-agent.service > /dev/null 2>&1 \
    || runuser -u "$RUN_AS" -- env XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user disable --now pool-agent.service > /dev/null 2>&1; then
    step "Older agent: stopped and disabled $RUN_AS's user unit pool-agent.service"
  else
    step "Older agent: $RUN_AS's user manager is not running, so its user unit is not either"
  fi
  # Renamed, so it can never start again at the next login, whatever the user manager did.
  mv "$OLD_UNIT" "$OLD_UNIT.replaced-by-system-install"
  say "  renamed $OLD_UNIT to $OLD_UNIT.replaced-by-system-install"
fi
stop_old_agents "$RUN_AS" "$OLD_DIR"

# ---- the agent's container (the default) or the host agent, never both -----------------------------
GPU_FLAGS=''
if [ -n "$GPU" ] && [ -n "$JETSON" ]; then
  # CSV mode: the NVIDIA runtime mounts the GPU's device nodes; nv_tegra_release labels it gpu=tegra.
  GPU_FLAGS="--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -v /etc/nv_tegra_release:/etc/nv_tegra_release:ro"
elif [ -n "$GPU" ]; then
  GPU_FLAGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=utility"  # nvidia-smi, so the agent can report the GPUs
fi
agent_container() {  # DOCKER-RUN-OPTIONS... IMAGE ARGS...: with the agent's mounts ($STATE at its host path)
  # shellcheck disable=SC2086
  docker run --network host --pid host $GPU_FLAGS \
    -v /var/run/docker.sock:/var/run/docker.sock -v "$ETC:$ETC" -v "$STATE:$STATE" -v "$PREFIX:/opt/pool:ro" \
    -e "POOL_UID=$RUN_UID" -e "POOL_GID=$RUN_GID" -e "POOL_AGENT_CONFIG=$CONF" -e "POOL_STATE=$STATE" "$@"
}
stop_container() {  # running jobs get their grace period, then requeue
  if docker inspect "$CONTAINER" > /dev/null 2>&1; then
    docker stop -t 90 "$CONTAINER" > /dev/null 2>&1 || true
    docker rm -f "$CONTAINER" > /dev/null 2>&1 || true
  fi
}
if [ -n "$HOST_AGENT" ]; then
  if docker inspect "$CONTAINER" > /dev/null 2>&1; then
    step "Agent container: stopping $CONTAINER (the host agent replaces it)"
    stop_container
  fi
else
  step "Agent image: $AGENT_IMAGE"
  if ! docker pull -q "$AGENT_IMAGE" > /dev/null 2>&1 && ! docker image inspect "$AGENT_IMAGE" > /dev/null 2>&1; then
    die "could not pull $AGENT_IMAGE: build it from a checkout (docker build -f deploy/pool/Dockerfile -t pool .) and run this again with --image pool, or use --host-agent"
  fi
  if [ -f "$UNIT" ]; then
    step "Host agent: stopping and disabling pool-agent.service (the container replaces it)"
    systemctl disable --now pool-agent.service > /dev/null 2>&1 || true
    mv "$UNIT" "$UNIT.replaced-by-container"
    systemctl daemon-reload > /dev/null 2>&1 || true
  fi
fi

# ---- 4. register --------------------------------------------------------------------------------
if [ -n "$MIGRATE" ]; then
  cp "$OLD_CONF" "$CONF"
  mv "$OLD_CONF" "$OLD_CONF.moved-to-etc-pool"
  step "Register: kept the older install's registration as $(config_name "$CONF") (moved $OLD_CONF to $CONF)"
  old_server=$(config_field "$CONF" server)
  [ "$old_server" = "$SERVER" ] || say "  it keeps talking to $old_server, the server it registered with"
elif [ -n "$REGISTERED" ]; then
  step "Register: already registered as $(config_name "$CONF")"
else
  [ -z "$REREGISTER" ] || [ ! -f "$CONF" ] || mv "$CONF" "$CONF.old"
  if [ -n "$HOST_AGENT" ]; then
    PYTHONPATH="$PREFIX" POOL_AGENT_CONFIG="$CONF" "$PY" -m pool.cli agent register --server "$SERVER" --code "$CODE" --name "$NAME" \
      || die "registration failed (a code works once, for 7 days: get a new one with \`pool machine add\`)"
  else
    agent_container --rm "$AGENT_IMAGE" register --server "$SERVER" --code "$CODE" --name "$NAME" \
      || die "registration failed (a code works once, for 7 days: get a new one with \`pool machine add\`)"
  fi
fi
chown "$RUN_AS:$RUN_GROUP" "$CONF"
chmod 0600 "$CONF"
NAME=$(config_name "$CONF")

if [ -n "$HOST_AGENT" ] && have pgrep && pgrep -u "$RUN_AS" -f 'pool.cli agent run' > /dev/null 2>&1 && ! systemctl is-active --quiet pool-agent; then
  say "  note: other agents run as $RUN_AS and were left alone (pgrep -af 'pool.cli agent run'), for example ones started from a checkout"
fi

# ---- 5. the service -----------------------------------------------------------------------------
if [ -z "$HOST_AGENT" ]; then
  stop_container
  agent_container -d --name "$CONTAINER" --restart unless-stopped --stop-timeout 90 \
    -e "POOL_IMAGE=$AGENT_IMAGE" "$AGENT_IMAGE" agent > /dev/null \
    || die "could not start the agent's container (docker logs $CONTAINER)"
  step "Service: container $CONTAINER started, restarted by Docker (uid $RUN_UID, $RUN_AS)"
else
cat > "$UNIT" << EOF
[Unit]
Description=pool agent: runs the pool's jobs in containers
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
User=$RUN_AS
Group=$RUN_GROUP
Environment=PYTHONPATH=$PREFIX
Environment=POOL_AGENT_CONFIG=$CONF
Environment=PYTHONUNBUFFERED=1
ExecStart=$PY -m pool.cli agent run --work $STATE/work
Restart=always
RestartSec=5
RestartPreventExitStatus=78
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable pool-agent.service > /dev/null 2>&1
systemctl restart pool-agent.service
step "Service: pool-agent.service enabled and started (user $RUN_AS)"
fi

# ---- 6. smoke test (warnings only) ---------------------------------------------------------------
as_agent() {  # the host agent runs containers as $RUN_AS; the agent's container has the socket's group
  if [ -n "$HOST_AGENT" ] && have runuser; then runuser -u "$RUN_AS" -- "$@"; else "$@"; fi
}
pass() { say "  PASS  $*"; }
fail() { say "  FAIL  $*"; FAILED=1; }
FAILED=''
step "Smoke test"
if ! as_agent docker image inspect "$SMOKE_IMAGE" > /dev/null 2>&1 && ! as_agent docker pull -q "$SMOKE_IMAGE" > /dev/null 2>&1; then
  fail "docker pull $SMOKE_IMAGE${HOST_AGENT:+ as $RUN_AS} (no registry access, or no docker access)"
else
  pass "docker works${HOST_AGENT:+ for $RUN_AS}"
  out=$(as_agent docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges "$SMOKE_IMAGE" \
    sh -c 'echo $(ls /sys/class/net); if touch /probe 2> /dev/null; then echo rw; else echo ro; fi' 2>&1 || true)
  if [ "$(echo "$out" | sed -n 1p)" = lo ]; then pass "network unreachable (loopback only)"; else fail "network: the container sees $(echo "$out" | sed -n 1p)"; fi
  if [ "$(echo "$out" | sed -n 2p)" = ro ]; then pass "root filesystem read-only"; else fail "root filesystem: $(echo "$out" | sed -n 2p)"; fi
  if [ -n "$GPU" ]; then
    # Jetson: the GPU comes in as device nodes (CSV mode); nvidia-smi is not mounted into containers.
    if [ -n "$JETSON" ]; then gpu_flags='--gpus all'; else gpu_flags='--gpus device=0'; fi
    if have nvidia-smi && [ -z "$JETSON" ]; then
      # shellcheck disable=SC2086
      gpu_out=$(as_agent docker run --rm --network none --read-only $gpu_flags "$SMOKE_IMAGE" nvidia-smi -L 2>&1 || true)
    else  # a Jetson without nvidia-smi: the GPU's device nodes are there
      # shellcheck disable=SC2086
      gpu_out=$(as_agent docker run --rm --network none --read-only $gpu_flags "$SMOKE_IMAGE" sh -c 'ls -d /dev/nvhost-* /dev/nvgpu /dev/nvmap 2> /dev/null | head -n 1' 2>&1 || true)
    fi
    case "$gpu_out" in
      GPU*|/dev/*) pass "GPU in a container: $(echo "$gpu_out" | head -n 1)" ;;
      *) fail "GPU in a container ($gpu_flags): $(echo "$gpu_out" | tail -n 1)" ;;
    esac
  fi
fi
sleep 3
if [ -n "$HOST_AGENT" ]; then
  if systemctl is-active --quiet pool-agent.service; then pass "pool-agent.service is running"; else fail "pool-agent.service is not running (journalctl -u pool-agent -n 50)"; fi
elif [ "$(docker inspect -f '{{.State.Running}} {{.RestartCount}}' "$CONTAINER" 2> /dev/null)" = "true 0" ]; then
  pass "the agent's container $CONTAINER is running"
else
  fail "the agent's container $CONTAINER is not running (docker logs $CONTAINER)"
fi
[ -z "$FAILED" ] || warn "some checks failed: the agent is installed, but jobs may fail until they pass"

# ---- 7. warm up ---------------------------------------------------------------------------------
if [ -n "$WARM_IMAGE" ]; then
  step "Warm up: pulling $WARM_IMAGE in the background"
  (as_agent docker pull -q "$WARM_IMAGE" > /dev/null 2>&1 &) || true
fi

# ---- 8. done ------------------------------------------------------------------------------------
say ""
say "This machine is $NAME in the pool at $SERVER."
say "  see it:   pool machine ls            (from a computer logged in to the pool)"
if [ -n "$HOST_AGENT" ]; then
  say "  status:   pool-agent status          (logs: journalctl -u pool-agent -f)"
else
  say "  status:   sudo pool-agent status     (logs: sudo pool-agent logs -f)"
  say "  update:   run this installer again   (the server's agent, the latest image)"
fi
say "  pause:    pool-agent pause           (running jobs finish; then: pool-agent resume)"
say "  remove:   sudo pool-agent uninstall  (unregisters it; leaves Docker installed)"
"""


# The agent's container image (deploy/pool/Dockerfile, published by .github/workflows/pool-images.yml).
# A server names another with POOL_AGENT_IMAGE; a machine with install.sh --image.
AGENT_IMAGE = "ghcr.io/ykirpichev/human-gym/pool:latest"


def render(server: str, package_sha256: str, agent_image: str | None = None) -> str:
    image = agent_image or os.environ.get("POOL_AGENT_IMAGE") or AGENT_IMAGE
    if not re.fullmatch(r"[A-Za-z0-9./:@_-]+", image):
        raise ValueError(f"agent image {image!r}: letters, digits and ./:@_- only")
    return (INSTALL_SH.replace("__OLD_AGENTS__", OLD_AGENTS_SH.strip()).replace("__AGENT_IMAGE__", image)
            .replace("__SERVER__", server).replace("__SHA256__", package_sha256))
