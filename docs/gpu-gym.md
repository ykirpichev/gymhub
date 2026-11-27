# GPU gym: kernel practice on your own GPU

Practice GPU kernels on your own GPU, with no website and no submission limit. Each exercise is a `solve` function you write in CUDA, PyTorch, or Triton. A PyTorch reference checks the result, a benchmark tells you how close you are to the hardware's limit, and your progress is saved.

The exercises have the same shape as [LeetGPU](https://leetgpu.com) (fill in `solve`, write outputs in place), but they are an original set. Nothing here contacts LeetGPU.

## Setup (once)

```bash
./setup.sh                    # venv, PyTorch for this machine, rich, clangd
source .venv/bin/activate
gpu-gym test playground --lang all   # a solved example, checks the toolchain
```

`setup.sh` runs on Linux (x86_64 and aarch64) and on Apple-silicon Macs. With an NVIDIA driver it installs the PyTorch CUDA build that matches the driver (Triton included); without one it installs the CPU build, so the algorithm and LLM gyms work and the GPU exercises need a machine with an NVIDIA GPU and the CUDA toolkit. `./setup.sh --dry-run` shows what it would install, and `./setup.sh --help` lists the overrides (`PYTHON`, `TORCH_INDEX`, `SKIP_CLANGD`).

Open **this folder** in VS Code or Cursor, accept the clangd extension, and CUDA and Python autocomplete will work.

## The practice loop

```bash
gpu-gym next                  # print the first exercise you haven't solved
gpu-gym watch vector_add      # re-runs the tests every time you save
# edit gyms/gpu/solutions/vector_add/solution.cu in your editor ...
gpu-gym hint vector_add       # stuck? reveals one hint at a time
gpu-gym bench vector_add      # once it passes: how fast is it?
gpu-gym progress              # what you've solved, best times
```

All commands take `--lang cuda|pytorch|triton|all` (default `cuda`).

In the editor, open a file in `gyms/gpu/solutions/<exercise>/` and press **Ctrl+Shift+B** to test that exercise. Use **Terminal → Run Task** to watch, bench, or get a hint.

## Reading an exercise

`gpu-gym show <exercise>` prints `gyms/gpu/problems/<exercise>/PROBLEM.md`. You can also open that file in the editor's Markdown preview. Every statement has the same sections:

- **Task**: what goes in, what comes out, in a few sentences.
- **Arguments**: name, type, shape, and role of each `solve` argument.
- **Formula** and a small worked **Example**.
- **How to think about it**: where the parallelism is, and why the naive version is slow.
- **Tests**: generated from `problem.py`. It lists the exact test shapes, the tolerance, the benchmark input, and the fastest time this GPU's memory bandwidth allows.

## Reading a benchmark

```
vector_add  [cuda]  pass  6/6
  time    0.8537 ms median   min 0.8505  max 0.8555  15 runs on A[16777216], ...
  memory  235.8 GB/s = 98% of this GPU's copy peak (241 GB/s)   192.0 MiB moved
  torch   0.8839 ms   you are 1.04x faster than PyTorch
  best    new best! 0.9120 -> 0.8537 ms
```

- **memory** is for memory-bound kernels (most of the easy and medium ones). The ceiling is a measured device-to-device copy, so above about 80% there is little left to gain.
- **compute** (TFLOP/s) is shown for matmul-like problems, where arithmetic is the limit.
- **torch** times the PyTorch reference on the same inputs. Beating it on simple kernels is realistic. On matmul it is a hard target.

Each timed run uses fresh inputs and flushes L2 first. Compilation is not timed.

## Progress

Each `test` and `bench` run is recorded in `gyms/gpu/progress.json` (plain JSON, gitignored like `solutions/`, so your work stays local). It holds pass/fail per language, attempts, first-solved date, best time, a short benchmark history, and how many hints you've seen.

```bash
gpu-gym list                          # every exercise with ✓ / ✗ / · per language
gpu-gym progress                      # totals, best times, recent runs
gpu-gym next --lang triton            # next exercise not yet solved in Triton
gpu-gym progress --reset matmul --force
```

To redo an exercise from scratch: `gpu-gym reset <exercise> --force` (restores the blank starter).

## Exercises

`gpu-gym list` shows them in the suggested order:

| Track | Exercises |
| --- | --- |
| Warm-up: indexing and launch config | `vector_add`, `relu`, `saxpy`, `gelu`, `reverse` |
| Memory patterns: coalescing, shared-memory tiles | `transpose`, `conv1d`, `conv2d` |
| Cooperation in a block: reductions, atomics, scans | `reduction`, `histogram`, `matvec`, `prefix_sum` |
| Row-wise model kernels | `row_softmax`, `layer_norm` |
| LLM building blocks | `embedding`, `swiglu`, `rms_norm`, `rope`, `cross_entropy` |
| Compute-bound: tiling for reuse | `matmul`, `attention`, `causal_attention`, `moe_matmul` |
| Data-dependent work | `spmv_csr`, `top_k`, `sort` |
| Low precision | `gemm_fp16`, `int8_matvec` |

`playground` is a solved vector add for checking the setup and is not counted.

For further reading, the gym page lists its sources (`sources` in `gyms/gpu/gym.json`): NVIDIA's CUDA Samples and CUTLASS, GPU Puzzles, Triton Puzzles, the Triton tutorials, and the GPU MODE lectures and reference kernels. The exercises here are original.

Tensors are contiguous and on the GPU, float32 unless the exercise says otherwise: the low-precision track passes FP16 (`__half`) and int8 (`int8_t`) tensors, and its statements give each argument's type. `solve` writes its output arguments and returns nothing. Before your code runs, outputs are filled with NaN (integer outputs with the type's largest value), so an element you forget to write shows up as a failure. In CUDA the tensor arguments are device pointers and `solve` must be `extern "C"`.

In your own exercise, give a tensor another type with `Param("A", "tensor", dtype="float16")` (`float32`, `float16`, `bfloat16`, `int8`, `uint8` or `int32`), and build its cases with the `dtype=` argument of the helpers in `gpu_gym/inputs.py`.

## Add your own exercise

```bash
gpu-gym new my_kernel --title "My kernel"
```

This creates `gyms/gpu/problems/my_kernel/` with `PROBLEM.md`, `HINTS.md`, `problem.py` (reference, test cases, benchmark case, bytes moved, optional flops) and `starter/`. It starts as a working vector add, so you can change one piece at a time. If you change a CUDA signature, run `gpu-gym compile-db` so autocomplete picks it up.

## Layout

```
gyms/gpu/problems/<name>/PROBLEM.md      statement
gyms/gpu/problems/<name>/HINTS.md        progressive hints
gyms/gpu/problems/<name>/problem.py      reference + cases (the checker)
gyms/gpu/problems/<name>/starter/        blank solutions
gyms/gpu/solutions/<name>/               your code
gyms/gpu/progress.json                   your history
gpu_gym/                                 the runner
```
