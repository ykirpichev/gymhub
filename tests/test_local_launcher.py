"""gymhub-board (gymhub/local.py): one command for a board and a runner on this machine."""

from gymhub import local

GPU = [{"name": "NVIDIA GeForce RTX 4090", "compute_capability": "8.9", "memory_gb": 24}]
CUDA_TORCH = {"torch": "2.9.0+cu130", "torch_cuda": "13.0", "sees_gpu": True, "device": "NVIDIA GeForce RTX 4090",
              "triton": "3.5.0"}


def test_a_ready_gpu_machine():
    lines, ready = local.readiness({"gpus": GPU, "driver": "580.95", "cuda": "13.0"}, CUDA_TORCH)
    assert ready
    assert all(line.strip().startswith("ok") for line in lines)
    assert "RTX 4090" in lines[0] and "nvcc 13.0" in "\n".join(lines)


def test_gpu_without_nvcc_or_triton_names_each_fix():
    torch = {**CUDA_TORCH, "triton": None}
    lines, ready = local.readiness({"gpus": GPU, "driver": "580.95", "cuda": None}, torch)
    text = "\n".join(lines)
    assert not ready
    assert "no nvcc" in text and "cuda-downloads" in text
    assert "Triton       not installed" in text


def test_cpu_pytorch_on_a_gpu_machine_says_to_rerun_setup():
    torch = {"torch": "2.9.0+cpu", "torch_cuda": None, "sees_gpu": False, "triton": None}
    lines, ready = local.readiness({"gpus": GPU, "driver": "580.95", "cuda": "13.0"}, torch)
    text = "\n".join(lines)
    assert not ready and "CPU-only build" in text and "./setup.sh" in text


def test_no_gpu_is_not_an_error_but_is_said():
    lines, ready = local.readiness({"gpus": [], "driver": None, "cuda": None},
                                   {"torch": None, "triton": None})
    text = "\n".join(lines)
    assert not ready and "none found" in text and "PyTorch      not installed" in text
    assert "nvcc" not in text and "Triton" not in text  # nothing GPU-specific to fix without a GPU


def test_environment_defaults_and_overrides():
    env = local.environment({})
    assert env["GYMHUB_STORE"].startswith("sqlite:") and env["GYMHUB_STORE"].endswith("dev.db")
    assert env["GYMHUB_AUTH"] == "dev" and env["GYMHUB_TRUST_JOBS"] == "1"
    mine = local.environment({"GYMHUB_STORE": "sqlite:/x.db", "GYMHUB_AUTH": "app"})
    assert mine["GYMHUB_STORE"] == "sqlite:/x.db" and "GYMHUB_TRUST_JOBS" not in mine


def test_no_sign_in_refuses_a_public_address(monkeypatch, capsys):
    monkeypatch.delenv("GYMHUB_AUTH", raising=False)
    assert local.main(["--host", "0.0.0.0"]) == 2
    assert "anyone who reaches" in capsys.readouterr().err


def test_check_only_prints_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(local, "readiness", lambda: (["  ok    GPU          test"], True))
    assert local.main(["--check"]) == 0
    assert "What this machine can grade" in capsys.readouterr().out
