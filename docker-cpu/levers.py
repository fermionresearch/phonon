"""Encoder/prefill configuration installers for the CPU runtime.

Three installers applied at load time as part of the shipped x86
configuration (each one is transcript-gated as installed here):

1. ``install_conv_fix`` — sets the audio tower's ``conv_chunksize`` (the
   upstream default of 500 never splits; splitting at 14 keeps torch-CPU
   conv2d in its fast regime on long clips). Uses upstream's own
   ``padded_feature.split(self.conv_chunksize, dim=0)`` loop, so clips
   under ``chunksize`` chunks are byte-untouched.

2. ``install_prefill_fusion`` — at prefill widths, computes q/k/v (and
   gate/up) as ONE dense GEMM over the row-concatenated fp32 weights
   instead of 3 (2). Each module's ``weight`` buffer is re-pointed at a
   slice view of the fused matrix, so resident bytes do not grow.
   Single-row (decode) calls fall through to the packed path.

3. ``install_c_prefill_attention`` — registers a custom attention interface
   for the thinker TEXT config only (the audio tower keeps sdpa) that
   routes multi-row fp32 batch-1 attention through the packed library's
   batched causal kernel (causal GQA, fp32 accumulation). q/k norms and
   RoPE stay in Torch — only the SDPA core is replaced. Because the custom
   interface name is not in transformers' mask registry,
   ``create_causal_mask`` returns None and the C kernel owns causality;
   any call the C path cannot serve exactly (mask present, batch != 1,
   non-fp32, over-long) falls back to the stock sdpa function and is
   counted.
"""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F


def install_conv_fix(bundle, chunksize: int = 14) -> None:
    """Set the encoder conv stem's chunk split (upstream mechanism)."""
    assert 1 <= chunksize <= 500, chunksize
    bundle.model.thinker.audio_tower.conv_chunksize = int(chunksize)


# ---------------------------------------------------------------------
# Prefill qkv / gate-up fusion

def _fuse_group(mods, stats) -> None:
    """Fuse a call group (lead called first on x, trailing on the same x)."""
    rows = [m.out_features for m in mods]
    offs = [0]
    for r in rows:
        offs.append(offs[-1] + r)
    fused = torch.cat([m.weight.data for m in mods], dim=0).contiguous()
    assert fused.dtype == torch.float32
    for m, r0, r1 in zip(mods, offs[:-1], offs[1:]):
        # slice views of the fused matrix: identical bytes, frees the
        # per-module copies (resident weight bytes unchanged)
        m._buffers["weight"] = fused[r0:r1]

    stash: dict = {}
    lead = mods[0]
    lead_inner = lead.forward
    lead_rows = rows[0]

    def lead_forward(x):
        if x.numel() == lead.in_features:  # decode single row: packed path
            return lead_inner(x)
        t0 = time.perf_counter()
        y = F.linear(x, fused)
        stash.clear()  # drop any stale entries from an aborted group call
        key = (x.data_ptr(), tuple(x.shape))
        for m, r0, r1 in zip(mods[1:], offs[1:-1], offs[2:]):
            stash[(id(m), key)] = y[..., r0:r1]
        out = y[..., :lead_rows]
        stats.proj_prefill_s += time.perf_counter() - t0
        return out

    lead.forward = lead_forward

    for m in mods[1:]:
        m_inner = m.forward

        def trailing_forward(x, _m=m, _inner=m_inner):
            if x.numel() == _m.in_features:
                return _inner(x)
            got = stash.pop((id(_m), (x.data_ptr(), tuple(x.shape))), None)
            if got is None:  # unexpected input: exact per-module fallback
                t0 = time.perf_counter()
                out = F.linear(x, _m.weight)
                stats.proj_prefill_s += time.perf_counter() - t0
                return out
            return got

        m.forward = trailing_forward


def install_prefill_fusion(bundle) -> None:
    """Fuse q/k/v and gate/up dense prefill GEMMs for every decoder layer."""
    from model import STATS
    for layer in bundle.model.thinker.model.layers:
        attn = layer.self_attn
        _fuse_group([attn.q_proj, attn.k_proj, attn.v_proj], STATS)
        mlp = layer.mlp
        _fuse_group([mlp.gate_proj, mlp.up_proj], STATS)


# ---------------------------------------------------------------------
# C batched causal prefill attention

PATTN_IMPL = "phonon_pattn_cpu"


def install_c_prefill_attention(bundle, lib, max_seq: int = 2048) -> dict:
    """Route the thinker text layers' multi-row attention through the packed
    library's batched causal kernel. Returns a counters dict {"c_calls",
    "fallbacks"}. Must be installed AFTER DriverDecoder construction (the
    decoder resolves its torch-attention glue fn from the config at init)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    thinker = bundle.model.thinker
    attn0 = thinker.model.layers[0].self_attn
    hd = attn0.head_dim
    nq = attn0.q_proj.out_features // hd
    nkv = attn0.k_proj.out_features // hd
    lib.pattn_setup(nq, nkv, hd, float(attn0.scaling), max_seq)
    sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
    counters = {"c_calls": 0, "fallbacks": 0}

    def pattn_forward(module, query, key, value, attention_mask,
                      dropout=0.0, scaling=None, **kwargs):
        if (attention_mask is not None or query.shape[0] != 1
                or query.shape[2] <= 1 or query.dtype != torch.float32
                or key.shape[2] > max_seq or module.training):
            counters["fallbacks"] += 1
            return sdpa_fn(module, query, key, value, attention_mask,
                           dropout=dropout, scaling=scaling, **kwargs)
        counters["c_calls"] += 1
        n_rows, n_keys = query.shape[2], key.shape[2]
        q = query[0].contiguous()
        k = key[0].contiguous()
        v = value[0].contiguous()
        out = torch.empty((n_rows, nq, hd), dtype=torch.float32)
        lib.pattn_run(n_rows, n_keys, q.data_ptr(), k.data_ptr(),
                      v.data_ptr(), out.data_ptr())
        return out.unsqueeze(0), None

    ALL_ATTENTION_FUNCTIONS[PATTN_IMPL] = pattn_forward
    # text config only; the audio tower keeps its own (sdpa) implementation.
    # PATTN_IMPL is absent from the mask registry, so create_causal_mask
    # early-returns None for it (transformers masking_utils contract).
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    assert PATTN_IMPL not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    attn0.config._attn_implementation = PATTN_IMPL
    assert thinker.model.config._attn_implementation == PATTN_IMPL, \
        "text layers and text model must share one config object"
    return counters
