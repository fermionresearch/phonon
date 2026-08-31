"""Complete Torch/Qwen runtime adapter for the Phonon fold4 artifact.

This module deliberately keeps the published ten-base-5 artifact as the source
of truth.  It mounts the official Qwen3-ASR Transformers graph, replaces the
196 text-decoder projections with Phonon's load-time fold4 representation, and
loads the remaining audio tower, embedding, norms, and tied vocabulary head
from the published safetensors shards.

There are two execution paths inside each projection:

* multi-row prefill uses ``torch.nn.functional.linear`` over the reconstructed
  BF16 fold4 matrix; and
* one-row autoregressive decode may use ``libphonon_fold4_cuda.so`` through its
  versioned C ABI.

The packed CUDA path is opt-in until decoded-text and full WER gates pass.  A
successful kernel microbenchmark alone is not permission to change transcripts.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Iterable

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from phonon_cuda_artifact import Fold4Matrix, load_fold4_matrix
from phonon_cuda_hybrid import artifact_profile, hybrid_artifact_state
from phonon_cuda_runtime import CudaKernelLibrary, Fold4CudaMatrix


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
    None is needed to construct this English ASR model.  The CUDA image installs
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
    """Reconstruct the exact load-time fold4 matrix as CPU BF16."""
    packed = torch.from_numpy(weights.codes)
    codes = torch.empty((weights.rows, weights.cols), dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    scale = torch.from_numpy(weights.scales).to(torch.float32)[:, None]
    center = torch.from_numpy(weights.centers).to(torch.float32)[:, None]
    return ((codes.to(torch.float32) - center) * scale).to(torch.bfloat16)


class PhononFold4Linear(nn.Module):
    """A fold4 decoder projection with safe prefill and optional CUDA decode."""

    def __init__(self, weights: Fold4Matrix):
        super().__init__()
        self.in_features = weights.cols
        self.out_features = weights.rows
        self.name = weights.name
        self._packed = weights
        # Runtime-derived data is intentionally absent from model state_dict.
        self.register_buffer("weight", _unpack_fold4(weights), persistent=False)
        self._cuda_matrix: Fold4CudaMatrix | None = None

    @property
    def packed_cuda_enabled(self) -> bool:
        return self._cuda_matrix is not None

    def enable_packed_cuda(self, library: CudaKernelLibrary) -> None:
        if self.weight.device.type != "cuda":
            raise RuntimeError("move the model to CUDA before enabling packed decode")
        if self._cuda_matrix is None:
            self._cuda_matrix = library.matrix(self._packed)

    def disable_packed_cuda(self) -> None:
        if self._cuda_matrix is not None:
            self._cuda_matrix.close()
            self._cuda_matrix = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._cuda_matrix is not None
            and x.is_cuda
            and x.numel() == self.in_features
        ):
            # The C ABI consumes FP32 activations and emits FP32. Preserve the
            # caller's rank/dtype exactly at the module boundary.
            flat = x.reshape(self.in_features).to(torch.float32).contiguous()
            out = torch.empty(self.out_features, device=x.device, dtype=torch.float32)
            stream = torch.cuda.current_stream(x.device)
            self._cuda_matrix.run_device(
                flat.data_ptr(), out.data_ptr(), stream.cuda_stream
            )
            return out.to(x.dtype).reshape(*x.shape[:-1], self.out_features)
        return F.linear(x, self.weight)


def _decoder_module_names(model_dir: Path) -> list[str]:
    import json

    manifest = json.loads((model_dir / "packed_manifest.json").read_text())
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
    import json

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
                # [O, H, W, C] (mlx_audio's sanitize ran at export). Torch
                # expects [O, C, H, W]; invert that one transform here.
                if "conv2d" in name and name.endswith(".weight") and tensor.ndim == 4:
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
                state["thinker." + name] = tensor
    return state


@dataclass
class PhononCudaBundle:
    model: nn.Module
    processor: object
    projections: list[PhononFold4Linear]
    kernel_library: CudaKernelLibrary | None = None
    profile: dict | None = None  # artifact storage profile (parity/audio6/micro)

    def enable_packed_decode(self, library_path: str | Path | None = None) -> None:
        if self.kernel_library is not None:
            return
        library = CudaKernelLibrary(library_path)
        enabled: list[PhononFold4Linear] = []
        try:
            for projection in self.projections:
                projection.enable_packed_cuda(library)
                enabled.append(projection)
        except Exception:
            for projection in enabled:
                projection.disable_packed_cuda()
            raise
        self.kernel_library = library

    def disable_packed_decode(self) -> None:
        for projection in self.projections:
            projection.disable_packed_cuda()
        self.kernel_library = None


def load_phonon_cuda(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    packed_decode: bool = False,
    library_path: str | Path | None = None,
) -> PhononCudaBundle:
    """Load the complete Phonon model on the official Qwen3-ASR graph.

    Requires the dependency set pinned by ``qwen-asr==0.0.6`` (notably
    Transformers 4.57.6).  It is intended for the isolated CUDA container,
    not the MLX/macOS Python environment.
    """
    configuration, modeling, processing = _qwen_backend_modules()
    Qwen3ASRConfig = configuration.Qwen3ASRConfig
    Qwen3ASRForConditionalGeneration = modeling.Qwen3ASRForConditionalGeneration
    Qwen3ASRProcessor = processing.Qwen3ASRProcessor

    model_dir = Path(model_dir).resolve()
    config = Qwen3ASRConfig.from_pretrained(model_dir)

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen3ASRForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(old_dtype)

    projections: list[PhononFold4Linear] = []
    for name in _decoder_module_names(model_dir):
        projection = PhononFold4Linear(load_fold4_matrix(model_dir, name))
        _set_child(model.thinker, name, projection)
        projections.append(projection)

    # The three published models share the decoder format above; they differ
    # in how the audio tower and the tied embedding are STORED. The
    # dense-tower model keeps the original BF16 state loader, unchanged. The
    # quantized-tower models store MLX affine tensors that dequantize to
    # dense fp32 values (scales * code + biases from BF16 metadata is not
    # generally BF16-representable), so for them the graph is made fp32
    # FIRST — otherwise ``load_state_dict`` would round the dequantized
    # values through the graph's default BF16 parameters — and stays fp32
    # end to end (the configuration the CPU runtime's transcript gates
    # passed; the fold4 decoder buffers upcast exactly).
    profile = artifact_profile(model_dir)
    if profile["hybrid"]:
        model.float()
        state = hybrid_artifact_state(model_dir)
    else:
        state = _artifact_state(model_dir)
    incompatible = model.load_state_dict(state, strict=False)
    del state
    unexpected = list(incompatible.unexpected_keys)
    missing = [name for name in incompatible.missing_keys if name != "thinker.lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"artifact/model key mismatch; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    model.tie_weights()
    model.eval().to(device)
    processor = Qwen3ASRProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    bundle = PhononCudaBundle(model=model, processor=processor,
                              projections=projections, profile=profile)
    if packed_decode:
        bundle.enable_packed_decode(library_path)
    return bundle


def iter_projection_names(bundle: PhononCudaBundle) -> Iterable[str]:
    return (projection.name for projection in bundle.projections)
