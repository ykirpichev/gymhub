"""GPU gym tensors of other element types than float32: FP16 and BF16 inputs for Tensor Core kernels,
int8 weights for quantized ones. No GPU is needed: the cases are built on the CPU (gpu_gym.inputs.DEVICE)."""

import pytest

torch = pytest.importorskip("torch")

from gpu_gym import inputs, runner  # noqa: E402
from gpu_gym.catalog import load_problem  # noqa: E402
from gpu_gym.cuda_backend import _wrapper_source  # noqa: E402
from gpu_gym.paths import PROBLEMS  # noqa: E402
from gpu_gym.spec import Param  # noqa: E402


@pytest.fixture
def on_cpu(monkeypatch):
    monkeypatch.setattr(inputs, "DEVICE", "cpu")


def test_a_tensor_param_names_its_dtype_and_other_params_cannot():
    assert Param("A", "tensor").dtype == "float32"
    assert Param("A", "tensor", dtype="float16").torch_dtype == torch.float16
    with pytest.raises(ValueError, match="dtype must be one of"):
        Param("A", "tensor", dtype="float8")
    with pytest.raises(ValueError, match="only a tensor"):
        Param("n", "int", dtype="int8")


def test_cuda_signature_uses_each_tensors_element_type():
    source = _wrapper_source([
        Param("A", "tensor", dtype="float16"), Param("B", "tensor", dtype="bfloat16"), Param("W", "tensor", dtype="int8"),
        Param("C", "tensor", output=True), Param("N", "int"), Param("eps", "float"),
    ])
    assert "#include <cuda_fp16.h>" in source and "#include <cuda_bf16.h>" in source and "#include <cstdint>" in source
    assert 'extern "C" void solve(const __half* A, const __nv_bfloat16* B, const int8_t* W, float* C, int N, float eps);' in source


def test_outputs_are_poisoned_so_unwritten_elements_are_caught():
    half = torch.zeros(3, dtype=torch.float16)
    runner._poison(half)
    assert torch.isnan(half).all()
    ints = torch.zeros(3, dtype=torch.int8)
    runner._poison(ints)
    assert ints.tolist() == [127, 127, 127]


def test_floats_compare_with_tolerance_integers_exactly_and_dtypes_must_match():
    want = torch.tensor([1.0, 2.0], dtype=torch.float16)
    assert runner._matches(torch.tensor([1.0005, 2.0], dtype=torch.float16), want, 1e-3, 1e-3)
    assert not runner._matches(torch.tensor([1.0, 2.0]), want, 1e-3, 1e-3)  # float32 where float16 is expected
    assert runner._matches(torch.tensor([3, -4], dtype=torch.int32), torch.tensor([3, -4], dtype=torch.int32), 1.0, 1.0)
    assert not runner._matches(torch.tensor([3, -3], dtype=torch.int32), torch.tensor([3, -4], dtype=torch.int32), 1.0, 1.0)


def test_case_summary_names_non_default_dtypes(on_cpu):
    problem = load_problem(PROBLEMS / "gemm_fp16")
    summary = runner._case_summary(problem, problem.functional_cases()[0])
    assert summary.startswith("A[16, 16] float16, B[16, 16] float16, C[16, 16], M=16")


def _check_cases(problem, solve):
    for case in problem.functional_cases():
        for param in problem.params:
            if param.kind == "tensor":
                assert case[param.name].dtype == param.torch_dtype, (case["_name"], param.name)
        user, reference = runner._split(problem, case)
        problem.reference(*(reference[param.name] for param in problem.params))
        solve(*(user[param.name] for param in problem.params))
        for param in problem.params:
            if param.output:
                yield case["_name"], runner._matches(user[param.name], reference[param.name], problem.atol, problem.rtol)


def test_gemm_fp16_accepts_fp32_accumulation(on_cpu):
    problem = load_problem(PROBLEMS / "gemm_fp16")

    def solve(A, B, C, M, N, K):
        C.copy_(A.float() @ B.float())

    assert all(ok for _, ok in _check_cases(problem, solve))


def test_gemm_fp16_rejects_an_fp16_accumulator(on_cpu):
    problem = load_problem(PROBLEMS / "gemm_fp16")
    case = next(case for case in problem.functional_cases() if case["K"] == 1024)
    user, reference = runner._split(problem, case)
    problem.reference(*(reference[param.name] for param in problem.params))
    acc = torch.zeros(case["M"], case["N"], dtype=torch.float16)
    for k in range(case["K"]):  # what a kernel with a __half accumulator computes
        acc = (acc.float() + case["A"][:, k:k + 1].float() * case["B"][k:k + 1, :].float()).half()
    assert not runner._matches(acc.float(), reference["C"], problem.atol, problem.rtol)


def test_int8_matvec_cases_and_example(on_cpu):
    problem = load_problem(PROBLEMS / "int8_matvec")

    def solve(W, scale, x, y, N, K):
        y.copy_((W.float() @ x.float()) * scale)

    assert all(ok for _, ok in _check_cases(problem, solve))
    example = problem.functional_cases()[0]
    user, _ = runner._split(problem, example)
    solve(*(user[param.name] for param in problem.params))
    assert user["y"].tolist() == pytest.approx([-1.5, 2.54], abs=1e-6)


def test_custom_cases_are_cast_to_each_params_dtype(on_cpu, tmp_path, monkeypatch):
    problem = load_problem(PROBLEMS / "int8_matvec")
    cases_file = tmp_path / "cases.json"
    cases_file.write_text('[{"W": {"tensor": [[1, 2]]}, "scale": {"tensor": [1.0]}, "x": {"tensor": [1.0, 1.0]},'
                          ' "y": {"empty": [1]}, "N": 1, "K": 2}]')
    monkeypatch.setattr(torch.Tensor, "to", _cpu_only(torch.Tensor.to))
    case = runner.custom_cases(problem, cases_file)[0]
    assert case["W"].dtype == torch.int8 and case["x"].dtype == torch.float16 and case["scale"].dtype == torch.float32


def _cpu_only(to):
    """Tensor.to with device="cuda" redirected to the CPU, for checking the cast without a GPU."""
    def patched(self, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return to(self, *args, **kwargs)
    return patched
