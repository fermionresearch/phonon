"""MLX runtime for packed Phonon decoder artifacts.

A Phonon artifact contains a two-plane five-state decoder.  Derived
artifacts may additionally store the tied token embedding/output head and
selected audio-tower linears in MLX's native affine format.  Their exact
layouts are declared in ``packed_manifest.json`` so loading never depends on
filename conventions or an implicit global quantization policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_audio.stt.models.qwen3_asr import Model, ModelConfig
from mlx_audio.utils import load_weights

from quint5_codec import packed_bytes_per_row, unpack_ten_base5_per_24bit

# NOTE (public release build): the internal tree also carries
# ``fused_trit_kernel`` (an experimental fused two-plane QMV Metal kernel, never
# a default and superseded by MLX's own native path) and ``trit3_codec`` (a
# superseded 3-bit storage format).  Neither is required by any shipping
# profile, so both are imported lazily here instead of at module scope.  Every
# released profile decodes through stock ``mx.quantized_matmul`` /
# ``nn.QuantizedLinear`` / ``nn.QuantizedEmbedding``.


class PackedTritLinear(nn.Module):
    """Learned five-value linear using two native 2-bit Metal QMMs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        *,
        use_fused_qmv: bool = False,
        slim_metadata: bool = False,
        trit3_codes: bool = False,
        quint5_codes: bool = False,
    ):
        super().__init__()
        if in_features % group_size:
            raise ValueError(f"in_features={in_features} is not divisible by {group_size}")
        packed_width = in_features // 16
        groups = in_features // group_size
        self.group_size = group_size
        self.use_fused_qmv = use_fused_qmv
        self.slim_metadata = slim_metadata
        self.trit3_codes = trit3_codes
        self.quint5_codes = quint5_codes
        self._in_features = in_features
        if trit3_codes and quint5_codes:
            raise ValueError("decoder code formats are mutually exclusive")
        if quint5_codes:
            self.quint5_q = mx.zeros(
                (out_features, packed_bytes_per_row(in_features)), dtype=mx.uint8
            )
        elif trit3_codes:
            self.trit3_q = mx.zeros((out_features, in_features * 3 // 8), dtype=mx.uint8)
        else:
            self.base_q = mx.zeros((out_features, packed_width), dtype=mx.uint32)
            self.residual_q = mx.zeros((out_features, packed_width), dtype=mx.uint32)
        if slim_metadata:
            self.base_alpha = mx.zeros((out_features,), dtype=mx.bfloat16)
            self.residual_scale = mx.zeros((1,), dtype=mx.bfloat16)
        else:
            self.base_scales = mx.zeros((out_features, groups), dtype=mx.bfloat16)
            self.base_biases = mx.zeros((out_features, groups), dtype=mx.bfloat16)
            self.residual_scales = mx.zeros((out_features, groups), dtype=mx.bfloat16)
            self.residual_biases = mx.zeros((out_features, groups), dtype=mx.bfloat16)

    def materialize_packed_codes(self) -> None:
        if self.quint5_codes:
            self.base_q, self.residual_q = unpack_ten_base5_per_24bit(
                self.quint5_q, self._in_features
            )
            mx.eval(self.base_q, self.residual_q)
            del self.quint5_q
            return
        if not self.trit3_codes:
            return
        in_features = self.trit3_q.shape[1] * 8 // 3
        from trit3_codec import unpack_five_value_3bit  # not in the public build

        self.base_q, self.residual_q = unpack_five_value_3bit(self.trit3_q, in_features)
        mx.eval(self.base_q, self.residual_q)
        del self.trit3_q

    def _metadata(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if not self.slim_metadata:
            return (
                self.base_scales,
                self.base_biases,
                self.residual_scales,
                self.residual_biases,
            )
        return (
            self._runtime_base_scales,
            self._runtime_base_biases,
            self._runtime_residual_scales,
            self._runtime_residual_biases,
        )

    def materialize_runtime_metadata(self) -> None:
        """Reconstruct contiguous affine metadata after strict weight loading.

        MLX QMM currently requires physically contiguous metadata; a stride-zero
        broadcast view is not numerically valid even though it has the expected
        logical values and shape.  Keeping these arrays runtime-only still
        removes the exact repetition from the distributed artifact.
        """

        if not self.slim_metadata:
            return
        out_features = self.base_q.shape[0]
        groups = self.base_q.shape[1] * 16 // self.group_size
        shape = (out_features, groups)
        self._runtime_base_scales = mx.contiguous(
            mx.broadcast_to(self.base_alpha[:, None], shape)
        )
        self._runtime_base_biases = mx.contiguous(-self._runtime_base_scales)
        self._runtime_residual_scales = mx.contiguous(
            mx.broadcast_to(self.residual_scale.reshape((1, 1)), shape)
        )
        self._runtime_residual_biases = mx.contiguous(-self._runtime_residual_scales)

    def materialize_folded_plane(self, bits: int) -> dict:
        """Replace the two 2-bit planes with one folded affine plane.

        Opt-in only.  This is a lossy re-encoding (see ``quint_fold_codec``) and
        therefore cannot be part of the byte-identical parity contract; it is
        gated on WER exactly like the compact/nano/micro fronts.  The two-plane
        codes are dropped afterwards so runtime memory stays comparable.
        """

        from quint_fold_codec import fold_two_planes

        base_scales, base_biases, residual_scales, residual_biases = self._metadata()
        packed, scales, biases, stats = fold_two_planes(
            self.base_q,
            base_scales,
            base_biases,
            self.residual_q,
            residual_scales,
            residual_biases,
            bits=bits,
            group_size=self.group_size,
        )
        self._fold = (packed, scales, biases, bits)
        del self.base_q
        del self.residual_q
        return stats

    def __call__(self, x: mx.array) -> mx.array:
        fold = getattr(self, "_fold", None)
        if fold is not None:
            packed, scales, biases, bits = fold
            return mx.quantized_matmul(
                x,
                packed,
                scales,
                biases,
                transpose=True,
                group_size=self.group_size,
                bits=bits,
                mode="affine",
            )
        base_scales, base_biases, residual_scales, residual_biases = self._metadata()
        if self.use_fused_qmv and x.size // x.shape[-1] == 1:
            from fused_trit_kernel import fused_two_plane_qmv  # not in the public build

            return fused_two_plane_qmv(
                x,
                self.base_q,
                base_scales,
                base_biases,
                self.residual_q,
                residual_scales,
                residual_biases,
            )
        base = mx.quantized_matmul(
            x,
            self.base_q,
            base_scales,
            base_biases,
            transpose=True,
            group_size=self.group_size,
            bits=2,
            mode="affine",
        )
        residual = mx.quantized_matmul(
            x,
            self.residual_q,
            residual_scales,
            residual_biases,
            transpose=True,
            group_size=self.group_size,
            bits=2,
            mode="affine",
        )
        return base + residual


def _parent_and_leaf(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def load_v18(
    model_path: str | Path,
    *,
    lazy: bool = False,
    use_fused_qmv: bool = False,
    fold_bits: int | None = None,
) -> Model:
    model_path = Path(model_path).expanduser().resolve()
    config_dict = json.loads((model_path / "config.json").read_text())
    config_dict.pop("quantization", None)
    config_dict.pop("quantization_config", None)
    model = Model(ModelConfig.from_dict(config_dict))

    manifest = json.loads((model_path / "packed_manifest.json").read_text())
    if manifest.get("status") != "PASS" or len(manifest.get("modules", [])) != 196:
        raise RuntimeError("invalid packed model manifest")
    if manifest.get("group_size") != 128 or manifest.get("bits") != 2:
        raise RuntimeError("unsupported packed model layout")
    decoder_metadata = manifest.get("decoder_metadata")
    slim_metadata = decoder_metadata is not None
    if slim_metadata and decoder_metadata.get("format") != "broadcast-scales-v1":
        raise RuntimeError("unsupported packed decoder metadata format")
    decoder_codes = manifest.get("decoder_codes")
    quint5_codes = bool(
        decoder_codes is not None
        and decoder_codes.get("format") == "ten-base5-per-24bit-v1"
    )
    trit3_codes = bool(
        decoder_codes is not None
        and decoder_codes.get("format") == "five-value-3bit-v1"
    )
    if decoder_codes is not None and not (trit3_codes or quint5_codes):
        raise RuntimeError("unsupported packed decoder code format")

    core = model._model
    packed_modules: list[PackedTritLinear] = []
    for row in manifest["modules"]:
        parent, leaf = _parent_and_leaf(core, row["name"])
        packed = PackedTritLinear(
            row["in_features"],
            row["out_features"],
            group_size=128,
            use_fused_qmv=use_fused_qmv,
            slim_metadata=slim_metadata,
            trit3_codes=trit3_codes,
            quint5_codes=quint5_codes,
        )
        setattr(
            parent,
            leaf,
            packed,
        )
        packed_modules.append(packed)

    hybrid = manifest.get("hybrid_quantization")
    if hybrid is not None:
        if hybrid.get("format") != "mlx-native-affine-v1":
            raise RuntimeError("unsupported hybrid quantization format")
        embedding = hybrid.get("embedding")
        if embedding:
            parent, leaf = _parent_and_leaf(core, embedding["name"])
            setattr(
                parent,
                leaf,
                nn.QuantizedEmbedding(
                    embedding["num_embeddings"],
                    embedding["dims"],
                    group_size=embedding["group_size"],
                    bits=embedding["bits"],
                    mode=embedding["mode"],
                ),
            )
        for row in hybrid.get("audio_linears", []):
            parent, leaf = _parent_and_leaf(core, row["name"])
            setattr(
                parent,
                leaf,
                nn.QuantizedLinear(
                    row["in_features"],
                    row["out_features"],
                    bias=row["bias"],
                    group_size=row["group_size"],
                    bits=row["bits"],
                    mode=row["mode"],
                ),
            )

    weights = load_weights(model_path)
    model.load_weights(list(weights.items()), strict=True)
    for packed in packed_modules:
        packed.materialize_packed_codes()
        packed.materialize_runtime_metadata()
    if fold_bits is not None:
        for packed in packed_modules:
            packed.materialize_folded_plane(fold_bits)
        mx.clear_cache()
    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    return Model.post_load_hook(model, model_path)
