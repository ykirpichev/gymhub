"""ModelRefs (#45 section 4): a named model artifact plus what placement needs to know about it.

    pool model add qwen3.6-35b-a3b@fp8 ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/<rev>

`describe(folder)` reads config.json and the safetensors headers (no weights are loaded, standard
library only) for parameters, bytes per dtype, layers, hidden size, experts and top-k. MoE
"active" parameters are the non-expert ones plus top_k/experts of the expert ones.

`need(meta, mode, kv_gb, gpus)` is what fits() checks: GPU memory per GPU and system RAM.
"""

import json
import struct
from pathlib import Path

OVERHEAD_GB = 1.5  # CUDA context, activations, allocator slack per GPU
SLACK = 1.05


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > 100 * 1024 * 1024:
            raise ValueError(f"{path.name}: implausible safetensors header")
        return json.loads(handle.read(length))


def _is_expert(name: str) -> bool:
    return ".experts." in name or ".expert." in name or "block_sparse_moe.experts" in name


def describe(folder: Path) -> dict:
    folder = Path(folder)
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    text = config.get("text_config") or config  # multimodal configs nest the language model
    params = expert_params = 0
    weight_bytes = 0
    by_dtype: dict[str, int] = {}
    for path in sorted(folder.rglob("*.safetensors")):
        header = _safetensors_header(path)
        weight_bytes += path.stat().st_size
        for name, info in header.items():
            if name == "__metadata__" or not isinstance(info, dict):
                continue
            count = 1
            for dim in info.get("shape", []):
                count *= dim
            params += count
            begin, end = info.get("data_offsets", [0, 0])
            by_dtype[info.get("dtype", "?")] = by_dtype.get(info.get("dtype", "?"), 0) + (end - begin)
            if _is_expert(name):
                expert_params += count
    if not weight_bytes:  # other formats: count the bytes of everything that looks like weights
        weight_bytes = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and p.suffix in (".bin", ".pt", ".gguf", ".pth"))
    experts = text.get("num_experts") or text.get("num_local_experts") or text.get("n_routed_experts") or 0
    top_k = text.get("num_experts_per_tok") or text.get("moe_top_k") or 0
    active = params
    if experts and top_k and expert_params:
        active = params - expert_params + expert_params * top_k / experts
    return {
        "architecture": (config.get("architectures") or [None])[0],
        "params": params, "active_params": int(active), "weight_bytes": weight_bytes,
        "bytes_by_dtype": by_dtype, "layers": text.get("num_hidden_layers"), "hidden": text.get("hidden_size"),
        "experts": experts or None, "top_k": top_k or None,
        "quantization": (config.get("quantization_config") or {}).get("quant_method"),
    }


def need(meta: dict, mode: str, kv_gb: float, gpus: int) -> dict:
    """GB of GPU memory each GPU needs, and GB of system RAM, to run `meta` in `mode` on `gpus` GPUs."""
    weights_gb = meta.get("weight_bytes", 0) / 1024**3
    share = max(gpus, 1)
    if mode == "full":
        gpu_gb = (weights_gb + kv_gb) * SLACK / share + OVERHEAD_GB
        ram_gb = 0.0
    elif mode == "offload":  # active parameters on the GPU, all of them in RAM
        fraction = meta.get("active_params", 0) / meta["params"] if meta.get("params") else 1.0
        gpu_gb = (weights_gb * fraction + kv_gb) * SLACK / share + OVERHEAD_GB
        ram_gb = weights_gb * SLACK
    elif mode.startswith("stage:"):
        k, n = (int(x) for x in mode[6:].split("/"))
        gpu_gb = (weights_gb * k / n + kv_gb) * SLACK / share + OVERHEAD_GB
        ram_gb = 0.0
    else:
        raise ValueError(f"unknown fit mode {mode}")
    return {"gpu_gb": round(gpu_gb, 2), "ram_gb": round(ram_gb, 2)}
