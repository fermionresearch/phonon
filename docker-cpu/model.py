"""Torch-CPU runtime adapter for the Phonon fold4 artifact.

Reuses the shared graph assembly (`graph.py`: official Qwen3-ASR
Transformers graph, 196 decoder projections swapped for load-time fold4,
artifact tensors byte-identical through the proven loader, conv2d
MLX->Torch permute fix) with the CPU execution profile:

* the whole graph runs FP32 (exact upcast of the artifact's BF16 tensors);
* multi-row prefill uses ``F.linear`` over the FP32 dense-from-fold4 matrix
  (BF16 round-trip preserved, then exact FP32 upcast);
* one-row decode calls the packed kernel library through ctypes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from artifact import Fold4Matrix, load_fold4_matrix
from graph import (
    _artifact_state,
    _decoder_module_names,
    _qwen_backend_modules,
    _set_child,
    _unpack_fold4,
)
from hybrid import artifact_profile, hybrid_artifact_state
from runtime import VARIANTS, CpuKernelLibrary


@dataclass
class ComponentStats:
    """Wall-time accumulators (single stream, sequential execution)."""
    phase: str = "prefill"
    proj_decode_s: float = 0.0
    proj_decode_calls: int = 0
    proj_prefill_s: float = 0.0
    attn_s: dict = field(default_factory=lambda: {"prefill": 0.0, "decode": 0.0})
    mlp_s: dict = field(default_factory=lambda: {"prefill": 0.0, "decode": 0.0})
    head_s: dict = field(default_factory=lambda: {"prefill": 0.0, "decode": 0.0})
    encoder_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    decode_steps: int = 0

    def reset(self):
        self.__init__()

    def snapshot(self) -> dict:
        return {
            "encoder_s": self.encoder_s,
            "prefill_s": self.prefill_s,
            "decode_s": self.decode_s,
            "decode_steps": self.decode_steps,
            "proj_decode_s": self.proj_decode_s,
            "proj_decode_calls": self.proj_decode_calls,
            "proj_prefill_s": self.proj_prefill_s,
            "attn_prefill_s": self.attn_s["prefill"],
            "attn_decode_s": self.attn_s["decode"],
            "mlp_prefill_s": self.mlp_s["prefill"],
            "mlp_decode_s": self.mlp_s["decode"],
            "head_prefill_s": self.head_s["prefill"],
            "head_decode_s": self.head_s["decode"],
        }


STATS = ComponentStats()


class PhononCpuFold4Linear(nn.Module):
    """fold4 decoder projection: FP32 dense prefill + packed CPU decode."""

    def __init__(self, weights: Fold4Matrix, library: CpuKernelLibrary | None,
                 variant: int):
        super().__init__()
        self.in_features = weights.cols
        self.out_features = weights.rows
        self.name = weights.name
        self.register_buffer(
            "weight", _unpack_fold4(weights).to(torch.float32), persistent=False)
        self._library = library
        self._variant = variant
        self._handle = (library.matrix(weights.rows, weights.cols, weights.codes,
                                       weights.scales, weights.centers)
                        if library is not None else -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        if self._handle >= 0 and x.numel() == self.in_features:
            flat = x.reshape(self.in_features)
            if flat.dtype != torch.float32 or not flat.is_contiguous():
                flat = flat.to(torch.float32).contiguous()
            out = torch.empty(self.out_features, dtype=torch.float32)
            self._library.matvec_ptr(self._handle, self._variant,
                                     flat.data_ptr(), out.data_ptr())
            STATS.proj_decode_s += time.perf_counter() - t0
            STATS.proj_decode_calls += 1
            return out.to(x.dtype).reshape(*x.shape[:-1], self.out_features)
        out = F.linear(x, self.weight)
        STATS.proj_prefill_s += time.perf_counter() - t0
        return out


def _timing_hooks(model: nn.Module) -> None:
    """Attach component wall-time hooks (sequential CPU execution)."""
    def wrap(module: nn.Module, attr: str, subtract_proj: bool):
        state = {}

        def pre(mod, args, kwargs=None):
            state["t0"] = time.perf_counter()
            state["p0"] = STATS.proj_decode_s + STATS.proj_prefill_s

        def post(mod, args, out):
            dt = time.perf_counter() - state["t0"]
            if subtract_proj:
                dt -= (STATS.proj_decode_s + STATS.proj_prefill_s) - state["p0"]
            # resolve through STATS at call time: reset() replaces the dicts
            getattr(STATS, attr)[STATS.phase] += dt

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    thinker = model.thinker
    for layer in thinker.model.layers:
        wrap(layer.self_attn, "attn_s", subtract_proj=True)
        wrap(layer.mlp, "mlp_s", subtract_proj=True)
    wrap(thinker.lm_head, "head_s", subtract_proj=False)

    # Greedy decode only consumes logits[:, -1]; the thinker computes the
    # full-sequence lm_head at prefill. Slicing to the last position is
    # OUTPUT-INVARIANT for this decode configuration (values never read) and
    # saves a [seq_len x 151936] FP32 GEMM per utterance.
    head = thinker.lm_head
    inner_head_forward = head.forward

    def head_forward(x):
        if x.dim() == 3 and x.shape[1] > 1:
            return inner_head_forward(x[:, -1:, :])
        return inner_head_forward(x)

    head.forward = head_forward

    tower_state = {}

    def tower_pre(mod, args, kwargs=None):
        tower_state["t0"] = time.perf_counter()

    def tower_post(mod, args, out):
        STATS.encoder_s += time.perf_counter() - tower_state["t0"]

    thinker.audio_tower.register_forward_pre_hook(tower_pre)
    thinker.audio_tower.register_forward_hook(tower_post)


def _phase_wrapper(module: nn.Module) -> None:
    """Split forward walls into prefill (first, multi-row) vs decode steps."""
    inner = module.forward

    def forward(*args, **kwargs):
        ids = kwargs.get("input_ids")
        embeds = kwargs.get("inputs_embeds")
        n_pos = None
        for t in (ids, embeds):
            if t is not None:
                n_pos = t.shape[1]
                break
        prefill = n_pos is None or n_pos > 1
        STATS.phase = "prefill" if prefill else "decode"
        t0 = time.perf_counter()
        out = inner(*args, **kwargs)
        dt = time.perf_counter() - t0
        if prefill:
            STATS.prefill_s += dt
        else:
            STATS.decode_s += dt
            STATS.decode_steps += 1
        return out

    module.forward = forward


@dataclass
class PhononCpuBundle:
    model: nn.Module
    processor: object
    projections: list
    library: CpuKernelLibrary | None
    stats: ComponentStats
    profile: dict | None = None  # artifact storage profile (parity/audio6/micro)


def load_phonon_cpu(
    model_dir: str | Path,
    *,
    variant: str = "lut",
    nthreads: int = 6,
    packed_decode: bool = True,
    library_path: str | Path | None = None,
    spin_iters: int = 60000,
) -> PhononCpuBundle:
    configuration, modeling, processing = _qwen_backend_modules()
    Qwen3ASRConfig = configuration.Qwen3ASRConfig
    Qwen3ASRForConditionalGeneration = modeling.Qwen3ASRForConditionalGeneration
    Qwen3ASRProcessor = processing.Qwen3ASRProcessor

    model_dir = Path(model_dir).resolve()
    config = Qwen3ASRConfig.from_pretrained(model_dir)
    model = Qwen3ASRForConditionalGeneration(config)

    library = (CpuKernelLibrary(library_path, nthreads=nthreads,
                                spin_iters=spin_iters)
               if packed_decode else None)
    vcode = VARIANTS[variant]

    projections = []
    for name in _decoder_module_names(model_dir):
        projection = PhononCpuFold4Linear(
            load_fold4_matrix(model_dir, name), library, vcode)
        _set_child(model.thinker, name, projection)
        projections.append(projection)

    # The three published models share the decoder format above; they differ
    # in tower/embedding STORAGE. The dense-tower model keeps the plain state
    # loader (BF16 tensors, exact upcast below). The quantized-tower models
    # store affine tensors that are dequantized to dense fp32 — so the graph
    # is made fp32 FIRST, or load_state_dict would round the dequantized
    # values through the graph's default BF16 parameters.
    profile = artifact_profile(model_dir)
    if profile["hybrid"]:
        model.float()
        state = hybrid_artifact_state(model_dir)
    else:
        state = _artifact_state(model_dir)
    incompatible = model.load_state_dict(state, strict=False)
    del state
    unexpected = list(incompatible.unexpected_keys)
    missing = [n for n in incompatible.missing_keys if n != "thinker.lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"artifact/model key mismatch; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}")
    model.tie_weights()
    # Whole graph FP32 on CPU: exact upcast of the artifact's BF16 values.
    model.float()
    model.eval()
    assert model.thinker.lm_head.weight.dtype == torch.float32
    processor = Qwen3ASRProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    _timing_hooks(model)
    _phase_wrapper(model.thinker)
    return PhononCpuBundle(model=model, processor=processor,
                           projections=projections, library=library,
                           stats=STATS, profile=profile)
