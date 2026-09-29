"""Loading support for the Phonon-1 artifacts with a quantized audio tower.

Phonon-1 and Phonon-1 Micro store the audio tower and the tied
embedding/vocabulary head quantized; Phonon-1 Big stores them dense. For the
quantized artifacts this module reads every module the artifact manifest
declares as quantized and returns a dense fp32 state for ``load_state_dict``.
``artifact_profile`` reports ``hybrid=False`` for Phonon-1 Big, which uses
the plain state loader in `phonon_cuda_model.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

AFFINE_FORMAT = "mlx-native-affine-v1"
PACKED_DECODER_SUFFIXES = (".quint5_q", ".base_alpha", ".residual_scale")


def _read_manifest(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "packed_manifest.json").read_text())


def artifact_profile(model_dir: str | Path) -> dict:
    """Describe how this artifact stores its tower / embedding.

    Returns ``{"key", "hybrid", "format", "tower_bits", "embed_bits",
    "tower_modules"}``.  ``key`` is the catalog profile name the storage
    shape corresponds to (parity / audio6 / micro), or "hybrid" for a
    layout this runtime does not accept.
    """
    manifest = _read_manifest(Path(model_dir))
    fmt = str(manifest.get("format", ""))
    hq = manifest.get("hybrid_quantization")
    if not hq:
        return {"key": "parity", "hybrid": False, "format": fmt,
                "tower_bits": None, "embed_bits": None, "tower_modules": 0}
    if hq.get("format") != AFFINE_FORMAT:
        raise ValueError(
            f"unsupported hybrid_quantization format {hq.get('format')!r} "
            f"(this loader implements {AFFINE_FORMAT})")
    linears = hq.get("audio_linears") or []
    tower_bits = sorted({int(row["bits"]) for row in linears})
    embed = hq.get("embedding")
    embed_bits = int(embed["bits"]) if embed else None
    if "head8audio6" in fmt and tower_bits == [6] and embed_bits == 8:
        key = "audio6"
    elif "hybrid4" in fmt and tower_bits == [4] and embed_bits == 4:
        key = "micro"
    else:
        key = "hybrid"
    return {"key": key, "hybrid": True, "format": fmt,
            "tower_bits": tower_bits[0] if len(tower_bits) == 1 else tower_bits,
            "embed_bits": embed_bits, "tower_modules": len(linears)}


def unpack_affine_codes(packed: torch.Tensor, bits: int,
                        in_features: int) -> torch.Tensor:
    """Unpack one quantized weight tensor into uint8 codes [out, in_features]."""
    if packed.ndim != 2:
        raise ValueError("packed affine weight must be two-dimensional")
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported affine bit width {bits}")
    rows = packed.shape[0]
    stream = packed.contiguous().view(torch.uint8).reshape(rows, -1)
    nbytes = stream.shape[1]
    if nbytes * 8 != in_features * bits:
        raise ValueError(
            f"packed width {packed.shape[1]} words does not hold "
            f"{in_features} codes of {bits} bits")
    if bits == 8:
        return stream.clone()
    if bits == 4:
        codes = torch.empty((rows, in_features), dtype=torch.uint8)
        codes[:, 0::2] = stream & 0x0F
        codes[:, 1::2] = stream >> 4
        return codes
    offsets = torch.arange(in_features, dtype=torch.int64) * bits
    byte_idx = offsets // 8
    shift = (offsets % 8).to(torch.int32)
    hi_idx = torch.clamp(byte_idx + 1, max=nbytes - 1)
    mask = (1 << bits) - 1
    lo = stream[:, byte_idx].to(torch.int32)
    hi = stream[:, hi_idx].to(torch.int32)
    codes = ((lo | (hi << 8)) >> shift[None, :]) & mask
    return codes.to(torch.uint8)


def dequantize_affine(packed: torch.Tensor, scales: torch.Tensor,
                      biases: torch.Tensor, *, bits: int, group_size: int,
                      in_features: int, row_block: int = 16384) -> torch.Tensor:
    """Dense fp32 weights for one quantized tensor."""
    rows = packed.shape[0]
    groups = in_features // group_size
    if in_features % group_size:
        raise ValueError("in_features must be a multiple of group_size")
    if tuple(scales.shape) != (rows, groups) or tuple(biases.shape) != (rows, groups):
        raise ValueError(
            f"affine metadata shape {tuple(scales.shape)} != {(rows, groups)}")
    out = torch.empty((rows, in_features), dtype=torch.float32)
    for r0 in range(0, rows, row_block):
        r1 = min(rows, r0 + row_block)
        codes = unpack_affine_codes(packed[r0:r1], bits, in_features)
        block = codes.to(torch.float32).reshape(r1 - r0, groups, group_size)
        s = scales[r0:r1].to(torch.float32)[:, :, None]
        b = biases[r0:r1].to(torch.float32)[:, :, None]
        out[r0:r1] = (block * s + b).reshape(r1 - r0, in_features)
    return out


def quantized_modules(manifest: dict) -> dict[str, dict]:
    """name -> {bits, group_size, in_features, out_features} from the manifest."""
    hq = manifest.get("hybrid_quantization") or {}
    if hq.get("format") != AFFINE_FORMAT:
        raise ValueError(f"not a {AFFINE_FORMAT} artifact")
    modules: dict[str, dict] = {}
    for row in hq.get("audio_linears") or []:
        if row.get("mode", "affine") != "affine":
            raise ValueError(f"{row['name']}: unsupported mode {row.get('mode')!r}")
        modules[str(row["name"])] = {
            "bits": int(row["bits"]), "group_size": int(row["group_size"]),
            "in_features": int(row["in_features"]),
            "out_features": int(row["out_features"])}
    embed = hq.get("embedding")
    if embed:
        if embed.get("mode", "affine") != "affine":
            raise ValueError("embedding: unsupported quantization mode")
        modules[str(embed["name"])] = {
            "bits": int(embed["bits"]), "group_size": int(embed["group_size"]),
            "in_features": int(embed["dims"]),
            "out_features": int(embed["num_embeddings"])}
    return modules


def hybrid_artifact_state(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Dense state for the official wrapper from a quantized-tower artifact.

    Same key mapping as `phonon_cuda_model._artifact_state`, with every module
    the manifest declares as quantized loaded as a dense fp32 weight.
    """
    model_dir = Path(model_dir)
    manifest = _read_manifest(model_dir)
    quant = quantized_modules(manifest)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]

    companions: set[str] = set()
    for base in quant:
        for suffix in (".weight", ".scales", ".biases"):
            key = base + suffix
            if key not in weight_map:
                raise ValueError(f"hybrid artifact is missing {key!r}")
            companions.add(key)

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.endswith(PACKED_DECODER_SUFFIXES) or name in companions:
            continue
        by_shard.setdefault(shard, []).append(name)

    state: dict[str, torch.Tensor] = {}
    handles: dict[str, object] = {}
    try:
        for shard, names in by_shard.items():
            handle = safe_open(model_dir / shard, framework="pt", device="cpu")
            handles[shard] = handle
            for name in names:
                tensor = handle.get_tensor(name)
                if "conv2d" in name and name.endswith(".weight") and tensor.ndim == 4:
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
                state["thinker." + name] = tensor
        for base, spec in quant.items():
            def get(suffix):
                shard = weight_map[base + suffix]
                if shard not in handles:
                    handles[shard] = safe_open(model_dir / shard, framework="pt",
                                               device="cpu")
                return handles[shard].get_tensor(base + suffix)
            packed = get(".weight")
            if packed.shape[0] != spec["out_features"]:
                raise ValueError(f"{base}: packed rows {packed.shape[0]} != "
                                 f"manifest out_features {spec['out_features']}")
            state["thinker." + base + ".weight"] = dequantize_affine(
                packed, get(".scales"), get(".biases"),
                bits=spec["bits"], group_size=spec["group_size"],
                in_features=spec["in_features"])
    finally:
        for handle in handles.values():
            close = getattr(handle, "__exit__", None)
            if close is not None:
                close(None, None, None)
    return state
