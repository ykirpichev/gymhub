from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, Y, N):
    Y.copy_(X * 2)


def _case(n, seed, name):
    return {"_name": name, "X": cuda_rand(n, seed=seed), "Y": cuda_empty(n), "N": n}


def functional_cases():
    known = {"_name": "example", "X": cuda_tensor([1.0, -2.0, 0.5]), "Y": cuda_empty(3), "N": 3}
    return [known] + [_case(n, index + 1, f"n={n}") for index, n in enumerate([1, 17, 257, 10_000])]


def performance_case(seed: int = 123):
    return _case(1 << 22, seed, "perf")


def traffic(case):
    return 2 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="scale",
        title="Scale a vector",
        difficulty="easy",
        summary="Y = 2·X for a float32 vector.",
        atol=1e-6,
        rtol=1e-6,
        params=[Param("X", "tensor"), Param("Y", "tensor", output=True), Param("N", "int")],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
