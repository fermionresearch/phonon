"""Exact Metal decoder for the V18 ten-base-5-symbols-per-24-bit format."""

from __future__ import annotations

import mlx.core as mx


_SOURCE = r"""
    uint output_index = thread_position_in_grid.x;
    uint packed_width = in_features / 16;
    uint output_words = quint5_q_shape[0] * packed_width;
    if (output_index >= output_words) return;
    uint row = output_index / packed_width;
    uint word = output_index - row * packed_width;
    uint bytes_per_row = quint5_q_shape[1];
    uint base_word = 0;
    uint residual_word = 0;
    #pragma unroll
    for (uint j = 0; j < 16; ++j) {
        uint logical_index = word * 16 + j;
        uint group = logical_index / 10;
        uint offset = logical_index - group * 10;
        const device uchar* source = quint5_q + row * bytes_per_row + group * 3;
        uint payload = uint(source[0]) | (uint(source[1]) << 8) |
                       (uint(source[2]) << 16);
        uint divisor;
        switch (offset) {
            case 0: divisor = 1; break;
            case 1: divisor = 5; break;
            case 2: divisor = 25; break;
            case 3: divisor = 125; break;
            case 4: divisor = 625; break;
            case 5: divisor = 3125; break;
            case 6: divisor = 15625; break;
            case 7: divisor = 78125; break;
            case 8: divisor = 390625; break;
            default: divisor = 1953125; break;
        }
        uint symbol = (payload / divisor) % 5;
        uint base_code;
        uint residual_code;
        switch (symbol) {
            case 0: base_code = 0; residual_code = 0; break;
            case 1: base_code = 0; residual_code = 2; break;
            case 2: base_code = 1; residual_code = 1; break;
            case 3: base_code = 2; residual_code = 0; break;
            default: base_code = 2; residual_code = 2; break;
        }
        base_word |= base_code << (2 * j);
        residual_word |= residual_code << (2 * j);
    }
    base_q[output_index] = base_word;
    residual_q[output_index] = residual_word;
"""

_UNPACK = mx.fast.metal_kernel(
    name="phonon_unpack_ten_base5_per_24bit_v1",
    input_names=("quint5_q", "in_features"),
    output_names=("base_q", "residual_q"),
    source=_SOURCE,
)


def packed_bytes_per_row(in_features: int) -> int:
    return ((in_features + 9) // 10) * 3


def unpack_ten_base5_per_24bit(
    quint5_q: mx.array, in_features: int
) -> tuple[mx.array, mx.array]:
    """Reconstruct the exact pair of native 2-bit packed planes."""

    if quint5_q.dtype != mx.uint8 or quint5_q.ndim != 2:
        raise ValueError("quint5_q must be a two-dimensional uint8 array")
    if in_features % 16 or quint5_q.shape[1] != packed_bytes_per_row(in_features):
        raise ValueError("invalid ten-base-5-symbols-per-24-bit layout")
    out_features = quint5_q.shape[0]
    shape = (out_features, in_features // 16)
    count = out_features * shape[1]
    return tuple(
        _UNPACK(
            inputs=(quint5_q, mx.array(in_features, dtype=mx.uint32)),
            grid=(count, 1, 1),
            threadgroup=(min(256, count), 1, 1),
            output_shapes=(shape, shape),
            output_dtypes=(mx.uint32, mx.uint32),
        )
    )
