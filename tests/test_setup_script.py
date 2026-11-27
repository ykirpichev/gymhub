"""setup.sh picks the PyTorch build and clangd source for the machine it runs on (checked with --dry-run).

uname and nvidia-smi are replaced by stubs on PATH, so every platform is checked on any Linux machine.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _dry_run(tmp_path: Path, os_name: str = "Linux", arch: str = "x86_64", cuda: str | None = None, **env: str):
    """Run setup.sh --dry-run from a copy in tmp_path, as if on os_name/arch with a driver for `cuda`."""
    work = tmp_path / "repo"
    work.mkdir()
    shutil.copy(ROOT / "setup.sh", work / "setup.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "uname", f'case "$1" in -s) echo {os_name} ;; -m) echo {arch} ;; *) echo {os_name} ;; esac')
    if cuda:
        _stub(bin_dir, "nvidia-smi", f'echo "| NVIDIA-SMI 580.00   Driver Version: 580.00   CUDA Version: {cuda}     |"')
    else:
        # Hide a real nvidia-smi on the test machine.
        _stub(bin_dir, "nvidia-smi", "exit 127")
    environ = {k: v for k, v in os.environ.items() if k not in ("TORCH_INDEX", "PYTHON", "SKIP_CLANGD")}
    environ["PATH"] = f"{bin_dir}{os.pathsep}{environ.get('PATH', '')}"
    environ.update(env)
    return subprocess.run(
        ["bash", str(work / "setup.sh"), "--dry-run"], capture_output=True, text=True, env=environ, timeout=60
    )


def test_cpu_build_without_nvidia_driver(tmp_path):
    result = _dry_run(tmp_path, cuda=None)
    assert result.returncode == 0, result.stderr
    assert "--index-url https://download.pytorch.org/whl/cpu" in result.stdout
    assert "no NVIDIA driver" in result.stdout


@pytest.mark.parametrize(
    "driver, build",
    [("13.2", "cu132"), ("13.1", "cu130"), ("13.0", "cu130"), ("12.9", "cu129"), ("12.8", "cu128"), ("12.7", "cu126")],
)
def test_cuda_build_matches_driver(tmp_path, driver, build):
    result = _dry_run(tmp_path, arch="aarch64", cuda=driver)
    assert result.returncode == 0, result.stderr
    assert f"--index-url https://download.pytorch.org/whl/{build}\n" in result.stdout


def test_old_driver_falls_back_to_cpu(tmp_path):
    result = _dry_run(tmp_path, cuda="12.2")
    assert result.returncode == 0, result.stderr
    assert "whl/cpu" in result.stdout
    assert "Update the driver" in result.stdout


def test_macos_apple_silicon_uses_pypi_build(tmp_path):
    result = _dry_run(tmp_path, os_name="Darwin", arch="arm64")
    assert result.returncode == 0, result.stderr
    assert "pip install torch (PyPI)" in result.stdout
    assert "Darwin aarch64" in result.stdout


def test_intel_mac_fails_with_a_reason(tmp_path):
    result = _dry_run(tmp_path, os_name="Darwin", arch="x86_64")
    assert result.returncode == 1
    assert "Intel Macs are not supported" in result.stderr


def test_unknown_platform_fails_with_a_reason(tmp_path):
    result = _dry_run(tmp_path, os_name="FreeBSD", arch="amd64")
    assert result.returncode == 1
    assert "unsupported platform FreeBSD/x86_64" in result.stderr


@pytest.mark.parametrize(
    "value, expected",
    [
        ("none", "PyTorch:  skip"),
        ("cpu", "--index-url https://download.pytorch.org/whl/cpu"),
        ("https://example.org/whl/cu128", "--index-url https://example.org/whl/cu128"),
    ],
)
def test_torch_index_override(tmp_path, value, expected):
    result = _dry_run(tmp_path, cuda="13.2", TORCH_INDEX=value)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_skip_clangd(tmp_path):
    result = _dry_run(tmp_path, SKIP_CLANGD="1")
    assert result.returncode == 0, result.stderr
    assert "clangd:   skip" in result.stdout


def test_missing_python_is_explained(tmp_path):
    result = _dry_run(tmp_path, PYTHON="python-does-not-exist")
    assert result.returncode == 1
    assert "PYTHON=python-does-not-exist was not found" in result.stderr
