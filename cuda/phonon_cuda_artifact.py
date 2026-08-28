"""Portable Phonon fold4 artifact conversion for CUDA and CPU runtimes.

The published artifact remains ten-base-5 symbols per 24 bits. This module
reproduces ``quint_fold_codec.fold_two_planes(..., bits=4)`` without importing
MLX, producing the packed codes/scales/centers the runtime reconstructs
dense weights from.

It intentionally accepts only the published broadcast-scales format. Generic
group-wise affine matrices have different metadata and must not pass this
loader silently.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


@dataclass(frozen=True)
class Fold4Matrix:
    name: str
    rows: int
    cols: int
    codes: np.ndarray
    scales: np.ndarray
    centers: np.ndarray

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (self.codes, self.scales, self.centers):
            digest.update(memoryview(np.ascontiguousarray(value)))
        return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing artifact file: {path}") from exc


def _load_tensor(model_dir: Path, index: dict, name: str) -> torch.Tensor:
    try:
        shard = index["weight_map"][name]
    except KeyError as exc:
        raise ValueError(f"artifact has no tensor {name!r}") from exc
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _unpack_symbols(quint5: torch.Tensor, cols: int) -> torch.Tensor:
    """Decode ten base-5 symbols from each little-endian 24-bit payload."""
    if quint5.ndim != 2 or quint5.dtype != torch.uint8:
        raise ValueError("quint5_q must be a two-dimensional uint8 tensor")
    expected = ((cols + 9) // 10) * 3
    if quint5.shape[1] != expected:
        raise ValueError(
            f"quint5_q width {quint5.shape[1]} != expected {expected}")
    source = quint5.to(torch.int32).reshape(quint5.shape[0], -1, 3)
    payload = source[:, :, 0] | (source[:, :, 1] << 8) | (source[:, :, 2] << 16)
    powers = torch.tensor(
        [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125],
        dtype=torch.int32,
    )
    symbols = torch.remainder(payload[:, :, None] // powers, 5)
    return symbols.reshape(quint5.shape[0], -1)[:, :cols]


def _fold4(alpha_bf16: torch.Tensor, residual_bf16: torch.Tensor,
           symbols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact Torch spelling of the MLX fold4 row-grid algorithm."""
    alpha = alpha_bf16.reshape(-1).to(torch.float32)
    residual = residual_bf16.reshape(-1).to(torch.float32)
    if residual.numel() == 1:
        residual = residual.expand_as(alpha)
    if residual.shape != alpha.shape:
        raise ValueError("residual_scale must be scalar or one value per row")

    big = torch.abs(alpha + residual)
    small = torch.abs(alpha - residual)
    big = torch.where(big > 0, big, torch.ones_like(big))
    ns = torch.arange(1, 8, dtype=torch.float32)
    # The BF16 round-trip is part of the production algorithm, not optional.
    steps = (big[:, None] / ns[None, :]).to(torch.bfloat16).to(torch.float32)
    small_codes = torch.minimum(
        torch.clamp(torch.round(small[:, None] / steps), min=0),
        ns[None, :],
    )
    errors = torch.abs(small[:, None] - small_codes * steps)
    best = torch.argmin(errors, dim=1)
    rows = torch.arange(alpha.numel())
    scale = steps[rows, best]
    center = (best + 1).to(torch.uint8)

    levels = torch.stack(
        (-alpha - residual, -alpha + residual, torch.zeros_like(alpha),
         alpha - residual, alpha + residual),
        dim=1,
    )
    weights = torch.gather(levels, 1, symbols.to(torch.int64))
    codes = torch.clamp(
        torch.round(weights / scale[:, None]) + center.to(torch.float32)[:, None],
        0,
        15,
    ).to(torch.uint8)
    return codes, scale, center


def load_fold4_matrix(model_dir: str | Path, module_name: str) -> Fold4Matrix:
    """Load one decoder module and derive its CUDA fold4 runtime matrix."""
    model_dir = Path(model_dir)
    manifest = _read_json(model_dir / "packed_manifest.json")
    decoder_codes = manifest.get("decoder_codes") or {}
    metadata = manifest.get("decoder_metadata") or {}
    if decoder_codes.get("format") != "ten-base5-per-24bit-v1":
        raise ValueError("CUDA fold4 loader requires ten-base5-per-24bit-v1")
    if metadata.get("format") != "broadcast-scales-v1":
        raise ValueError("CUDA fold4 loader requires broadcast-scales-v1")
    if manifest.get("group_size") != 128:
        raise ValueError("CUDA fold4 loader requires group_size=128")
    module = next(
        (row for row in manifest.get("modules", []) if row.get("name") == module_name),
        None,
    )
    if module is None:
        raise ValueError(f"packed manifest has no decoder module {module_name!r}")
    rows = int(module["out_features"])
    cols = int(module["in_features"])
    index = _read_json(model_dir / "model.safetensors.index.json")
    quint5 = _load_tensor(model_dir, index, module_name + ".quint5_q")
    alpha = _load_tensor(model_dir, index, module_name + ".base_alpha")
    residual = _load_tensor(model_dir, index, module_name + ".residual_scale")
    if tuple(quint5.shape[:1]) != (rows,):
        raise ValueError("quint5_q row count disagrees with packed manifest")

    symbols = _unpack_symbols(quint5, cols)
    codes, scales, centers = _fold4(alpha, residual, symbols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return Fold4Matrix(
        name=module_name,
        rows=rows,
        cols=cols,
        codes=np.ascontiguousarray(packed.numpy(), dtype=np.uint8),
        scales=np.ascontiguousarray(scales.numpy(), dtype=np.float32),
        centers=np.ascontiguousarray(centers.numpy(), dtype=np.uint8),
    )
