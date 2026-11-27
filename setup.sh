#!/usr/bin/env bash
# Create the local interpreter (.venv), install PyTorch for this machine, the gyms, and clangd.
#
# Works on Linux (x86_64, aarch64) and macOS (Apple silicon). It picks the PyTorch build itself:
#   Linux + NVIDIA driver  -> the CUDA build that matches the driver (Triton included)
#   Linux, no NVIDIA GPU   -> the CPU build (algorithm and LLM exercises; GPU exercises need CUDA)
#   macOS                  -> the default build (CPU / Apple MPS; no CUDA, no Triton)
#
# Overrides (environment variables):
#   PYTHON=python3.12      interpreter that creates .venv (3.10 or newer)
#   TORCH_INDEX=<url>      PyTorch wheel index, e.g. https://download.pytorch.org/whl/cu128
#   TORCH_INDEX=cpu        force the CPU build
#   TORCH_INDEX=none       do not install PyTorch (you install it yourself)
#   SKIP_CLANGD=1          do not set up clangd (editor autocomplete only)
#
#   ./setup.sh --dry-run   print what would be installed and exit
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h | --help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "setup.sh: unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

say() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() {
  printf 'setup.sh: %s\n' "$*" >&2
  exit 1
}

TORCH_BASE="https://download.pytorch.org/whl"
# CUDA builds PyTorch publishes, newest first. A driver runs any build up to the CUDA version it reports.
CUDA_BUILDS="13.2 13.0 12.9 12.8 12.6"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$ARCH" in
  arm64) ARCH=aarch64 ;;
  amd64) ARCH=x86_64 ;;
esac
case "$OS/$ARCH" in
  Linux/x86_64 | Linux/aarch64 | Darwin/aarch64) ;;
  Darwin/x86_64)
    die "Intel Macs are not supported: PyTorch stopped publishing builds for them after 2.2. Use an Apple-silicon Mac or Linux." ;;
  *)
    die "unsupported platform $OS/$ARCH. Supported: Linux x86_64, Linux aarch64, macOS on Apple silicon." ;;
esac

# --- Python -------------------------------------------------------------------------------------
pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    command -v "$PYTHON" > /dev/null || die "PYTHON=$PYTHON was not found"
    echo "$PYTHON"
    return
  fi
  local candidate
  for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" > /dev/null \
      && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2> /dev/null; then
      echo "$candidate"
      return
    fi
  done
  die "Python 3.10 or newer is required and none was found. Install it (e.g. 'sudo apt install python3 python3-venv' or 'brew install python'), or set PYTHON=/path/to/python3."
}
PY="$(pick_python)"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || die "$PY is Python $("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])'); 3.10 or newer is required."

# --- PyTorch build ------------------------------------------------------------------------------
# The highest CUDA version the installed NVIDIA driver supports ("13.0"), or empty with no driver.
driver_cuda_version() {
  command -v nvidia-smi > /dev/null || return 0
  # nvidia-smi fails when the driver is broken or mismatched: treat that as no driver.
  { nvidia-smi 2> /dev/null || true; } | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}

# 1 when version $1 <= version $2 (both "major.minor").
version_le() {
  local a_major="${1%%.*}" a_minor="${1#*.}" b_major="${2%%.*}" b_minor="${2#*.}"
  ((a_major < b_major || (a_major == b_major && a_minor <= b_minor)))
}

TORCH_NOTE=""
choose_torch() {
  case "${TORCH_INDEX:-}" in
    none) TORCH_KIND=none; TORCH_NOTE="TORCH_INDEX=none: not installing PyTorch"; return ;;
    cpu) TORCH_KIND=cpu; TORCH_URL="$TORCH_BASE/cpu"; TORCH_NOTE="TORCH_INDEX=cpu"; return ;;
    "") ;;
    *) TORCH_KIND=custom; TORCH_URL="$TORCH_INDEX"; TORCH_NOTE="TORCH_INDEX override"; return ;;
  esac
  if [[ "$OS" == Darwin ]]; then
    TORCH_KIND=default
    TORCH_URL=""
    TORCH_NOTE="macOS: CPU and Apple MPS build (no CUDA, so the CUDA and Triton exercises will not run)"
    return
  fi
  if [[ -f /etc/nv_tegra_release ]]; then
    die "this is an NVIDIA Jetson. The PyTorch builds on download.pytorch.org do not support it: install NVIDIA's Jetson build of PyTorch into .venv yourself and rerun with TORCH_INDEX=none (or TORCH_INDEX=cpu for a CPU-only setup)."
  fi
  local driver build
  driver="$(driver_cuda_version)"
  if [[ -z "$driver" ]]; then
    TORCH_KIND=cpu
    TORCH_URL="$TORCH_BASE/cpu"
    TORCH_NOTE="no NVIDIA driver found (nvidia-smi): CPU build. The algorithm and LLM exercises run; the GPU exercises need an NVIDIA GPU"
    return
  fi
  for build in $CUDA_BUILDS; do
    if version_le "$build" "$driver"; then
      TORCH_KIND=cuda
      TORCH_URL="$TORCH_BASE/cu${build/./}"
      TORCH_NOTE="NVIDIA driver supports CUDA $driver: CUDA $build build"
      return
    fi
  done
  TORCH_KIND=cpu
  TORCH_URL="$TORCH_BASE/cpu"
  TORCH_NOTE="the NVIDIA driver supports only CUDA $driver, older than any current PyTorch build (${CUDA_BUILDS##* } or newer): CPU build. Update the driver for the GPU exercises"
}
choose_torch

# --- clangd (editor autocomplete for CUDA and C++) -----------------------------------------------
CLANGD_VERSION=18.1.3
CLANGD_DEST=".tools/clangd"

# A clangd that is already installed: on PATH, Xcode's on macOS, or Homebrew's LLVM.
find_clangd() {
  local candidate
  for candidate in clangd clangd-20 clangd-19 clangd-18 clangd-17; do
    if command -v "$candidate" > /dev/null; then
      command -v "$candidate"
      return
    fi
  done
  if [[ "$OS" == Darwin ]]; then
    for candidate in "$(xcrun --find clangd 2> /dev/null || true)" /opt/homebrew/opt/llvm/bin/clangd; do
      if [[ -n "$candidate" && -x "$candidate" ]]; then
        echo "$candidate"
        return
      fi
    done
  fi
}

clangd_plan() {
  if [[ -n "${SKIP_CLANGD:-}" ]]; then
    echo "skip (SKIP_CLANGD set)"
  elif [[ -x "$CLANGD_DEST/bin/clangd" ]]; then
    echo "already set up in $CLANGD_DEST"
  elif [[ -n "$(find_clangd)" ]]; then
    echo "use $(find_clangd)"
  elif [[ "$OS/$ARCH" == Linux/x86_64 ]]; then
    echo "download clangd $CLANGD_VERSION from github.com/clangd/clangd"
  elif [[ "$OS" == Linux ]] && command -v apt-get > /dev/null; then
    echo "extract clangd-18 from apt packages into $CLANGD_DEST (no sudo)"
  else
    echo "not available (install clangd for editor autocomplete)"
  fi
}

write_clangd_wrapper() { # $1: the script body after the shebang
  mkdir -p "$CLANGD_DEST/bin"
  printf '#!/usr/bin/env bash\n%s\n' "$1" > "$CLANGD_DEST/bin/clangd"
  chmod +x "$CLANGD_DEST/bin/clangd"
}

clangd_from_github() {
  local stage
  stage="$(mktemp -d)"
  if curl -fsSL -o "$stage/clangd.zip" \
    "https://github.com/clangd/clangd/releases/download/$CLANGD_VERSION/clangd-linux-$CLANGD_VERSION.zip" \
    && "$PY" -m zipfile -e "$stage/clangd.zip" "$stage/out"; then
    rm -rf "$CLANGD_DEST/root"
    mkdir -p "$CLANGD_DEST"
    mv "$stage/out/clangd_$CLANGD_VERSION" "$CLANGD_DEST/root"
    chmod +x "$CLANGD_DEST/root/bin/clangd"
    write_clangd_wrapper 'exec "$(cd "$(dirname "$0")/.." && pwd)/root/bin/clangd" "$@"'
    rm -rf "$stage"
    return 0
  fi
  rm -rf "$stage"
  return 1
}

# Debian / Ubuntu: unpack the packages into .tools without installing them system-wide.
clangd_from_apt() {
  local stage root_abs multiarch
  multiarch="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2> /dev/null || echo "$ARCH-linux-gnu")"
  stage="$(mktemp -d)"
  root_abs="$(pwd)/$CLANGD_DEST/root"
  mkdir -p "$root_abs"
  # Package names as of Ubuntu 24.04; other releases name the libraries differently.
  if (cd "$stage" && apt-get download clangd-18 libgrpc++1.51t64 libgrpc29t64 libclang-common-18-dev libcares2 libre2-10) \
    > /dev/null 2>&1; then
    local deb
    for deb in "$stage"/*.deb; do
      dpkg-deb -x "$deb" "$root_abs"
    done
    rm -rf "$stage"
    write_clangd_wrapper "HERE=\"\$(cd \"\$(dirname \"\$0\")/..\" && pwd)\"
export LD_LIBRARY_PATH=\"\$HERE/root/usr/lib/$multiarch\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"
exec \"\$HERE/root/usr/lib/llvm-18/bin/clangd\" \"\$@\""
    return 0
  fi
  rm -rf "$stage" "$root_abs"
  return 1
}

# clangd is an editor convenience: failing to set it up warns and never stops the setup.
install_clangd() {
  [[ -z "${SKIP_CLANGD:-}" ]] || return 0
  [[ ! -x "$CLANGD_DEST/bin/clangd" ]] || return 0
  local existing
  existing="$(find_clangd)"
  if [[ -n "$existing" ]]; then
    say "Using clangd at $existing"
    write_clangd_wrapper "exec \"$existing\" \"\$@\""
    return 0
  fi
  if [[ "$OS/$ARCH" == Linux/x86_64 ]]; then
    say "Downloading clangd $CLANGD_VERSION"
    clangd_from_github && return 0
  elif [[ "$OS" == Linux ]] && command -v apt-get > /dev/null; then
    say "Extracting clangd-18 from apt packages"
    clangd_from_apt && return 0
  fi
  if [[ "$OS" == Darwin ]]; then
    warn "clangd not found; editor autocomplete is off. Install it with 'xcode-select --install' or 'brew install llvm', then rerun ./setup.sh."
  else
    warn "could not set up clangd; editor autocomplete is off. Install clangd (e.g. 'sudo apt install clangd'), then rerun ./setup.sh."
  fi
}

# --- Plan ---------------------------------------------------------------------------------------
case "$TORCH_KIND" in
  none) TORCH_LINE="skip" ;;
  default) TORCH_LINE="pip install torch (PyPI)" ;;
  *) TORCH_LINE="pip install torch --index-url $TORCH_URL" ;;
esac
say "Platform: $OS $ARCH, $("$PY" --version 2>&1) ($PY)"
say "PyTorch:  $TORCH_LINE"
say "          $TORCH_NOTE"
say "clangd:   $(clangd_plan)"
if [[ "$TORCH_KIND" == cuda ]] && ! command -v nvcc > /dev/null && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  warn "the CUDA toolkit (nvcc) was not found at /usr/local/cuda: the PyTorch and Triton exercises will run, the CUDA ones need the toolkit (https://developer.nvidia.com/cuda-downloads)."
fi
if ((DRY_RUN)); then
  exit 0
fi

# --- Install ------------------------------------------------------------------------------------
if [[ ! -x .venv/bin/python ]]; then
  say "Creating .venv"
  "$PY" -m venv .venv \
    || die "could not create .venv with $PY. On Debian/Ubuntu install the venv module: sudo apt install python3-venv"
fi
.venv/bin/pip install -q -U pip setuptools

case "$TORCH_KIND" in
  none) ;;
  default)
    say "Installing PyTorch"
    .venv/bin/pip install torch || die "installing PyTorch failed (see the pip output above)."
    ;;
  *)
    say "Installing PyTorch from $TORCH_URL"
    .venv/bin/pip install torch --index-url "$TORCH_URL" \
      || die "installing PyTorch from $TORCH_URL failed (see the pip output above). Pick another build with TORCH_INDEX=<url> (https://pytorch.org/get-started/locally/), or TORCH_INDEX=cpu."
    ;;
esac

say "Installing the gyms"
.venv/bin/pip install -q ninja numpy rich
.venv/bin/pip install -q -e ".[hub]"
.venv/bin/python -m gymhub.gymsfetch || warn "could not fetch the gyms; run .venv/bin/python -m gymhub.gymsfetch later."

install_clangd
if [[ "$TORCH_KIND" != none ]]; then
  .venv/bin/gpu-gym compile-db || warn "gpu-gym compile-db failed; CUDA autocomplete may be incomplete."
fi

echo
echo "Ready. Open this folder in VS Code or Cursor, then:"
echo "  source .venv/bin/activate"
if [[ "$TORCH_KIND" == cuda || "$TORCH_KIND" == custom ]]; then
  echo "  gpu-gym test playground --lang all     # checks the CUDA toolchain"
else
  echo "  algo-gym next                          # no CUDA here: start with the algo gym"
fi
echo "  gymhub-board                           # the board in your browser at http://localhost:8080, graded on this machine"
