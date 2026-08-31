"""Qwen3-ASR graph assembly shared by the Phonon CPU runtime.

Mounts the official Qwen3-ASR Transformers graph and loads every non-packed
tensor of the published artifact into it byte-identically (audio tower,
embedding, norms, tied vocabulary head). The 196 packed text-decoder
projections are swapped in by `model.py`; this module only provides the
graph, the backend-module resolution, and the dense state loader.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from artifact import Fold4Matrix

DECODER_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _qwen_backend_modules():
    """Load only Qwen's Transformers backend, bypassing demo/server extras.

    The upstream ``qwen_asr`` package initializer imports its forced aligner,
    Japanese/Korean tokenizers, audio URL helpers, and optional serving code.
    None is needed to construct this English ASR model.  This image installs
    the official wheel without dependencies, then this private package alias
    loads its Apache-2.0 Transformers backend directly.
    """
    package = "_phonon_qwen_transformers_backend"
    if package not in sys.modules:
        spec = importlib.util.find_spec("qwen_asr")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("qwen-asr==0.0.6 is not installed")
        root = Path(next(iter(spec.submodule_search_locations)))
        backend = root / "core" / "transformers_backend"
        if not (backend / "modeling_qwen3_asr.py").is_file():
            raise RuntimeError(f"official Qwen Transformers backend missing: {backend}")
        module = types.ModuleType(package)
        module.__path__ = [str(backend)]
        module.__package__ = package
        sys.modules[package] = module
    configuration = importlib.import_module(package + ".configuration_qwen3_asr")
    modeling = importlib.import_module(package + ".modeling_qwen3_asr")
    processing = importlib.import_module(package + ".processing_qwen3_asr")
    return configuration, modeling, processing


def _unpack_fold4(weights: Fold4Matrix) -> torch.Tensor:
    """Reconstruct the exact load-time fold4 matrix as CPU BF16.

    The BF16 round-trip is part of the published reconstruction algorithm,
    not an optimization; the CPU graph upcasts the result to FP32 exactly.
    """
    packed = torch.from_numpy(weights.codes)
    codes = torch.empty((weights.rows, weights.cols), dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    scale = torch.from_numpy(weights.scales).to(torch.float32)[:, None]
    center = torch.from_numpy(weights.centers).to(torch.float32)[:, None]
    return ((codes.to(torch.float32) - center) * scale).to(torch.bfloat16)


def _decoder_module_names(model_dir: Path) -> list[str]:
    manifest = json.loads((Path(model_dir) / "packed_manifest.json").read_text())
    modules = [str(row["name"]) for row in manifest.get("modules", [])]
    if len(modules) != 196:
        raise RuntimeError(f"expected 196 packed decoder modules, found {len(modules)}")
    bad = [name for name in modules if not name.endswith(DECODER_SUFFIXES)]
    if bad:
        raise RuntimeError(f"unexpected packed module names: {bad[:3]}")
    return modules


def _set_child(root: nn.Module, dotted_name: str, child: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


def _artifact_state(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all non-packed tensors and map them into the official wrapper."""
    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if name.endswith((".quint5_q", ".base_alpha", ".residual_scale")):
            continue
        by_shard.setdefault(shard, []).append(name)

    state: dict[str, torch.Tensor] = {}
    for shard, names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                # The artifact stores audio-tower convolutions in MLX layout
                # [O, H, W, C]. Torch expects [O, C, H, W]; invert that one
                # transform here.
                if "conv2d" in name and name.endswith(".weight") and tensor.ndim == 4:
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
                state["thinker." + name] = tensor
    return state
