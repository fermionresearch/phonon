"""Fold the two 2-bit V18 decoder planes into one affine plane.

Background (measured): MLX's
``quantized_matmul`` cost at prefill shapes is essentially independent of bit
width -- 2, 3, 4, 5, 6 and 8 bits all land within noise of each other, because
the kernel is dequantize/compute bound rather than bandwidth bound.  The V18
decoder therefore pays an exact 2x for representing each learned weight as two
2-bit affine planes: the second plane is free in bytes but doubles the kernel
count and the arithmetic.

Every learned decoder weight takes one of five values per output row,
``{0, +/-d, +/-D}`` where ``d = |alpha - r|`` and ``D = alpha + r``.  A uniform
affine grid contains all five exactly only when ``D/d`` is an integer, which the
trained weights do not satisfy (measured ratios 1.42-5.51).  So this fold is a
lossy re-encoding and can never be part of the byte-identical ``parity``
contract; it is WER-gated like the ``compact``/``nano``/``micro`` fronts.

The grid is *not* anchored at ``D/kmax``.  For a plane with ``bits`` bits any
denominator ``n <= (2**bits - 1)//2`` may be used with step ``D/n``; choosing
``n`` per output row is a best-rational-approximation of ``d/D`` and is far more
accurate than a naive min/max quantizer.  Code ``n`` is the exact zero state,
which matters because the learned decoder is genuinely sparse there.

The distributed artifact is unchanged: the fold is derived at load time from the
exact base-5 codes.  Every step runs as MLX graph work so a cold start pays GPU
time rather than a 3.5 GB float64 NumPy round trip.
"""

from __future__ import annotations

import mlx.core as mx

SUPPORTED_BITS = (4, 8)  # bit widths whose MLX packed layout is 32-bit aligned


def _row_grid(small: mx.array, big: mx.array, bits: int):
    """Best (step, n) per row: minimises |d - round(d/step)*step| over n.

    ``small``/``big`` are the row's smallest non-zero and largest weight
    magnitudes.  Both are tiny (one value per output row), so the exhaustive
    search over denominators is negligible.
    """
    kmax = (2 ** bits - 1) // 2
    ns = mx.arange(1, kmax + 1, dtype=mx.float32)
    step = (big[:, None] / ns[None, :]).astype(mx.bfloat16).astype(mx.float32)
    code = mx.clip(mx.round(small[:, None] / mx.maximum(step, 1e-30)), 0, ns[None, :])
    err = mx.abs(small[:, None] - code * step)
    best = mx.argmin(err, axis=1)
    rows = mx.arange(big.shape[0])
    return step[rows, best], (best + 1).astype(mx.float32)


def _pack(codes: mx.array, bits: int) -> mx.array:
    """Pack integer codes into MLX's little-endian uint32 quantized layout."""
    per = 32 // bits
    out_f, in_f = codes.shape
    grouped = codes.reshape(out_f, in_f // per, per)
    packed = mx.zeros((out_f, in_f // per), dtype=mx.uint32)
    for i in range(per):
        packed = packed | (grouped[:, :, i] << mx.array(bits * i, dtype=mx.uint32))
    return packed


def fold_two_planes(
    base_q: mx.array,
    base_scales: mx.array,
    base_biases: mx.array,
    residual_q: mx.array,
    residual_scales: mx.array,
    residual_biases: mx.array,
    *,
    bits: int = 4,
    group_size: int = 128,
    stats: bool = False,
):
    """Return ``(packed, scales, biases, stats_or_None)`` for the folded plane."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported fold width {bits}; use one of {SUPPORTED_BITS}")
    w = mx.dequantize(
        base_q, base_scales, base_biases, group_size=group_size, bits=2, mode="affine"
    ).astype(mx.float32) + mx.dequantize(
        residual_q, residual_scales, residual_biases,
        group_size=group_size, bits=2, mode="affine",
    ).astype(mx.float32)

    magnitude = mx.abs(w)
    big = mx.max(magnitude, axis=1)
    positive = mx.where(magnitude > 0, magnitude, mx.full(magnitude.shape, mx.inf))
    small = mx.min(positive, axis=1)
    del positive
    small = mx.where(mx.isinf(small), big, small)
    big = mx.where(big > 0, big, mx.ones_like(big))
    mx.eval(big, small)

    step, n = _row_grid(small, big, bits)
    mx.eval(step, n)

    codes = mx.clip(
        mx.round(w / step[:, None]) + n[:, None], 0, 2 ** bits - 1
    ).astype(mx.uint32)
    mx.eval(codes)

    scale_row = step.astype(mx.bfloat16)
    bias_row = (-n * step).astype(mx.bfloat16)
    groups = codes.shape[1] // group_size
    scales = mx.contiguous(mx.broadcast_to(scale_row[:, None], (codes.shape[0], groups)))
    biases = mx.contiguous(mx.broadcast_to(bias_row[:, None], (codes.shape[0], groups)))

    report = None
    if stats:
        recon = (
            codes.astype(mx.float32) * scale_row.astype(mx.float32)[:, None]
            + bias_row.astype(mx.float32)[:, None]
        )
        err = mx.abs(recon - w)
        scale_ref = big[:, None]
        report = {
            "max_abs_err_over_row_max": float(mx.max(err / scale_ref).item()),
            "mean_abs_err_over_row_max": float(mx.mean(err / scale_ref).item()),
            "frac_exact": float(mx.mean((err == 0).astype(mx.float32)).item()),
            "median_denominator": float(mx.median(n).item())
            if hasattr(mx, "median") else None,
        }
        del recon, err
    del w, magnitude

    packed = _pack(codes, bits)
    mx.eval(packed, scales, biases)
    del codes
    return packed, scales, biases, report
