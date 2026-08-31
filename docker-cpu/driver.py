"""C decode-step driver + quantized vocabulary head (Python side).

* ``CpuDriverLibrary`` — loads the packed CPU kernel library and initialises
  its thread pool. Exposes the per-module matrix table, the fused table, the
  one-C-call-per-token decode-step driver, the int8-tiered vocabulary head,
  and the fused C attention paths (decode and batched-causal prefill).
* ``DriverDecoder`` — the flat greedy decode loop. One ctypes call per decode
  step; C owns all fused projection dispatches, Python glue callbacks own the
  Torch ops (norms / RoPE / attention glue / activations / residuals), so the
  graph numerics match the reference module-dispatch path exactly.
* ``install_quant_head`` — swaps the FP32 vocabulary head for the
  int8-tiered C head (approximate int8 pass over all rows, exact f64 rescore
  of the top candidates). Greedy decode reads only the argmax, so the
  wrapper returns one-hot logits.

Everything here executes the transcript-gated decode configuration; there
are no sampling paths.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from artifact import load_fold4_matrix
from runtime import VARIANTS, default_library

GLUE_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)
EOS_IDS = (151643, 151645)


class CpuDriverLibrary:
    """ctypes ownership wrapper for the packed kernel library's pool."""

    def __init__(self, path: str | Path | None = None, *,
                 nthreads: int = 6, max_cols: int = 3072,
                 spin_iters: int = 60000):
        self.path = Path(path) if path is not None else default_library()
        if not self.path.is_file():
            raise RuntimeError(f"Phonon CPU kernel library missing: {self.path}")
        lib = self.lib = ctypes.CDLL(str(self.path))
        u8p = ctypes.POINTER(ctypes.c_uint8)
        f32p = ctypes.POINTER(ctypes.c_float)
        lib.phonon_cpu_init_g2.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_long)
        lib.phonon_cpu_init_g2.restype = ctypes.c_int
        lib.phonon_cpu_matrix_create.argtypes = (ctypes.c_int, ctypes.c_int, u8p, f32p, u8p)
        lib.phonon_cpu_matrix_create.restype = ctypes.c_long
        lib.phonon_cpu_matrix_create_g2.argtypes = (ctypes.c_int, ctypes.c_int, u8p, f32p, u8p)
        lib.phonon_cpu_matrix_create_g2.restype = ctypes.c_long
        self._matvec_raw = ctypes.CFUNCTYPE(
            None, ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(
            ("phonon_cpu_matvec", lib))
        self._matvec_g2_raw = ctypes.CFUNCTYPE(
            None, ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(
            ("phonon_cpu_matvec_g2", lib))
        lib.phonon_cpu_head_create.argtypes = (ctypes.c_long, ctypes.c_long, ctypes.c_void_p)
        lib.phonon_cpu_head_create.restype = ctypes.c_long
        self._head_argmax = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(
            ("phonon_cpu_head_argmax", lib))
        lib.phonon_cpu_head_flips.restype = ctypes.c_long
        lib.phonon_cpu_driver_setup.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_int, GLUE_CB,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        lib.phonon_cpu_driver_setup.restype = ctypes.c_int
        lib.phonon_cpu_decode_step.restype = None
        lib.phonon_cpu_driver_proj_seconds.restype = ctypes.c_double
        # Fused C decode attention (guarded: absent from reduced builds).
        try:
            lib.phonon_cpu_attn_setup.argtypes = (
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int)
            lib.phonon_cpu_attn_setup.restype = ctypes.c_long
            lib.phonon_cpu_attn_start.argtypes = (
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
            lib.phonon_cpu_attn_start.restype = None
            self._attn_layer_raw = ctypes.CFUNCTYPE(
                None, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(
                ("phonon_cpu_attn_layer", lib))
            lib.phonon_cpu_attn_advance.restype = None
            lib.phonon_cpu_attn_enable.argtypes = (ctypes.c_int,)
            lib.phonon_cpu_attn_enable.restype = None
            lib.phonon_cpu_attn_set_naive.argtypes = (ctypes.c_int,)
            lib.phonon_cpu_attn_set_naive.restype = None
            lib.phonon_cpu_attn_seconds.restype = ctypes.c_double
            lib.phonon_cpu_attn_seq.restype = ctypes.c_int
            self.has_attn = True
        except AttributeError:
            self.has_attn = False
        # Batched causal prefill attention (guarded likewise).
        try:
            lib.phonon_cpu_pattn_setup.argtypes = (
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                ctypes.c_int)
            lib.phonon_cpu_pattn_setup.restype = ctypes.c_long
            self._pattn_run_raw = ctypes.CFUNCTYPE(
                None, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
                ("phonon_cpu_pattn_run", lib))
            lib.phonon_cpu_pattn_seconds.restype = ctypes.c_double
            lib.phonon_cpu_pattn_calls.restype = ctypes.c_long
            self.has_pattn = True
        except AttributeError:
            self.has_pattn = False
        if lib.phonon_cpu_init_g2(nthreads, max_cols, spin_iters) != 0:
            raise RuntimeError("phonon_cpu_init_g2 failed")
        self.nthreads = nthreads
        self._head_weight_ref = None  # keeps the FP32 weights alive for rescore

    # -- per-module matrix table ---------------------------------------
    def matrix(self, rows: int, cols: int, codes: np.ndarray,
               scales: np.ndarray, centers: np.ndarray) -> int:
        return self._create(self.lib.phonon_cpu_matrix_create, rows, cols,
                            codes, scales, centers)

    def matvec_ptr(self, handle: int, variant: int, x_ptr: int, y_ptr: int) -> None:
        self._matvec_raw(handle, variant, x_ptr, y_ptr)

    # -- fused matrix table --------------------------------------------
    def matrix_g2(self, rows: int, cols: int, codes: np.ndarray,
                  scales: np.ndarray, centers: np.ndarray) -> int:
        return self._create(self.lib.phonon_cpu_matrix_create_g2, rows, cols,
                            codes, scales, centers)

    def matvec_g2_ptr(self, handle: int, variant: int, x_ptr: int, y_ptr: int) -> None:
        self._matvec_g2_raw(handle, variant, x_ptr, y_ptr)

    def _create(self, fn, rows, cols, codes, scales, centers) -> int:
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        scales = np.ascontiguousarray(scales, dtype=np.float32)
        centers = np.ascontiguousarray(centers, dtype=np.uint8)
        assert codes.shape == (rows, cols // 2)
        handle = fn(rows, cols,
                    codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    centers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        if handle < 0:
            raise RuntimeError("matrix create failed")
        return int(handle)

    # -- quantized head --------------------------------------------------
    def head_create(self, weight: torch.Tensor) -> None:
        assert weight.dtype == torch.float32 and weight.is_contiguous()
        rows, cols = weight.shape
        self._head_weight_ref = weight  # rescore reads this storage in-place
        if self.lib.phonon_cpu_head_create(rows, cols, weight.data_ptr()) != 0:
            raise RuntimeError("phonon_cpu_head_create failed")

    def head_argmax(self, x_ptr: int) -> int:
        idx = self._head_argmax(x_ptr)
        if idx < 0:
            raise RuntimeError("head not created")
        return int(idx)

    def head_flips(self) -> int:
        return int(self.lib.phonon_cpu_head_flips())

    # -- decode-step driver ----------------------------------------------
    def driver_setup(self, handles: list[int], variant: int, cb,
                     ptrs: list[int]) -> None:
        nlayers = len(handles) // 4
        arr = (ctypes.c_long * len(handles))(*handles)
        if self.lib.phonon_cpu_driver_setup(nlayers, arr, variant, cb, *ptrs) != 0:
            raise RuntimeError("phonon_cpu_driver_setup failed")

    def decode_step(self) -> None:
        self.lib.phonon_cpu_decode_step()

    def driver_proj_seconds(self) -> float:
        return float(self.lib.phonon_cpu_driver_proj_seconds())

    # -- fused C decode attention ------------------------------------------
    def attn_setup(self, nlayers: int, nq: int, nkv: int, hd: int, eps: float,
                   scaling: float, max_seq: int, kring_ptr: int, vring_ptr: int,
                   qw_ptrs, kw_ptrs, naive: bool) -> None:
        if not self.has_attn:
            raise RuntimeError(f"{self.path.name} lacks the fused attention API")
        if self.lib.phonon_cpu_attn_setup(
                nlayers, nq, nkv, hd, eps, scaling, max_seq,
                kring_ptr, vring_ptr, qw_ptrs, kw_ptrs, int(naive)) != 0:
            raise RuntimeError("phonon_cpu_attn_setup failed")

    def attn_start(self, seq: int, cos_ptr: int, sin_ptr: int) -> None:
        self.lib.phonon_cpu_attn_start(seq, cos_ptr, sin_ptr)

    def attn_layer(self, layer: int, qkv_ptr: int, out_ptr: int) -> None:
        self._attn_layer_raw(layer, qkv_ptr, out_ptr)

    def attn_advance(self) -> None:
        self.lib.phonon_cpu_attn_advance()

    def attn_enable(self, on: bool) -> None:
        self.lib.phonon_cpu_attn_enable(int(on))

    def attn_set_naive(self, naive: bool) -> None:
        self.lib.phonon_cpu_attn_set_naive(int(naive))

    def attn_seconds(self) -> float:
        return float(self.lib.phonon_cpu_attn_seconds())

    # -- batched causal prefill attention -----------------------------------
    def pattn_setup(self, nq: int, nkv: int, hd: int, scaling: float,
                    max_seq: int) -> None:
        if not self.has_pattn:
            raise RuntimeError(f"{self.path.name} lacks the prefill "
                               "attention API")
        if self.lib.phonon_cpu_pattn_setup(nq, nkv, hd, scaling, max_seq) != 0:
            raise RuntimeError("phonon_cpu_pattn_setup failed")

    def pattn_run(self, nrows: int, nkeys: int, q_ptr: int, k_ptr: int,
                  v_ptr: int, out_ptr: int) -> None:
        self._pattn_run_raw(nrows, nkeys, q_ptr, k_ptr, v_ptr, out_ptr)

    def pattn_seconds(self) -> float:
        return float(self.lib.phonon_cpu_pattn_seconds())

    def pattn_calls(self) -> int:
        return int(self.lib.phonon_cpu_pattn_calls())


# ---------------------------------------------------------------------
# fold4 load-once cache (shared by module init and the fused-handle
# builder), so each packed module is decoded from the artifact exactly once
# per process.

def enable_fold4_cache():
    import model as _model

    cache: dict[str, object] = {}
    original = load_fold4_matrix

    def cached(model_dir, name):
        if name not in cache:
            cache[name] = original(model_dir, name)
        return cache[name]

    _model.load_fold4_matrix = cached
    return cached, cache


def attach_module_handles(bundle, lib: CpuDriverLibrary, model_dir,
                          loader=load_fold4_matrix) -> None:
    """Give the projection modules per-module handles on the pool."""
    for proj in bundle.projections:
        w = loader(model_dir, proj.name)
        proj._library = lib
        proj._handle = lib.matrix(w.rows, w.cols, w.codes, w.scales, w.centers)


def build_fused_handles(lib: CpuDriverLibrary, model_dir, nlayers: int = 28,
                        loader=load_fold4_matrix) -> list[int]:
    """Row-concatenate q/k/v and gate/up fold4 matrices per layer.

    Each output row is computed whole by one thread from unchanged
    codes/scales/centers and rows stay 16-aligned, so fused outputs are
    bit-identical to the per-module matvecs."""
    handles: list[int] = []
    for l in range(nlayers):
        base = f"model.layers.{l}."
        q = loader(model_dir, base + "self_attn.q_proj")
        k = loader(model_dir, base + "self_attn.k_proj")
        v = loader(model_dir, base + "self_attn.v_proj")
        assert q.cols == k.cols == v.cols
        handles.append(lib.matrix_g2(
            q.rows + k.rows + v.rows, q.cols,
            np.concatenate([q.codes, k.codes, v.codes], axis=0),
            np.concatenate([q.scales, k.scales, v.scales], axis=0),
            np.concatenate([q.centers, k.centers, v.centers], axis=0)))
        o = loader(model_dir, base + "self_attn.o_proj")
        handles.append(lib.matrix_g2(o.rows, o.cols, o.codes, o.scales, o.centers))
        g = loader(model_dir, base + "mlp.gate_proj")
        u = loader(model_dir, base + "mlp.up_proj")
        assert g.cols == u.cols and g.rows == u.rows
        handles.append(lib.matrix_g2(
            g.rows + u.rows, g.cols,
            np.concatenate([g.codes, u.codes], axis=0),
            np.concatenate([g.scales, u.scales], axis=0),
            np.concatenate([g.centers, u.centers], axis=0)))
        d = loader(model_dir, base + "mlp.down_proj")
        handles.append(lib.matrix_g2(d.rows, d.cols, d.codes, d.scales, d.centers))
    return handles


# ---------------------------------------------------------------------
# Quantized vocabulary head installers

def install_quant_head(bundle, lib: CpuDriverLibrary, mode: str) -> None:
    """mode: 'int8t' (C tiered head, one-hot logits) or 'bf16' (gemv)."""
    head = bundle.model.thinker.lm_head
    vocab = head.out_features

    if mode == "int8t":
        lib.head_create(head.weight.data)
        fab = torch.zeros((1, 1, vocab), dtype=torch.float32)
        state = {"last": 0}

        def head_forward(x):
            if x.dim() == 3 and x.shape[1] > 1:
                x = x[:, -1:, :]
            flat = x.reshape(-1)
            if flat.dtype != torch.float32 or not flat.is_contiguous():
                flat = flat.to(torch.float32).contiguous()
            idx = lib.head_argmax(flat.data_ptr())
            fab[0, 0, state["last"]] = 0.0
            fab[0, 0, idx] = 1.0
            state["last"] = idx
            return fab

        head.forward = head_forward
    elif mode == "bf16":
        w16 = head.weight.data.to(torch.bfloat16)

        def head_forward(x):
            if x.dim() == 3 and x.shape[1] > 1:
                x = x[:, -1:, :]
            return F.linear(x.to(torch.bfloat16), w16).to(torch.float32)

        head.forward = head_forward
    elif mode != "fp32":
        raise ValueError(mode)


# ---------------------------------------------------------------------
# Flat greedy decode loop over the C driver

class DriverDecoder:
    def __init__(self, bundle, lib: CpuDriverLibrary, handles: list[int],
                 variant: str, stats, *, attn: str = "torch",
                 attn_max_seq: int = 2048):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from graph import _qwen_backend_modules

        _, modeling, _ = _qwen_backend_modules()
        self.rope_fn = modeling.apply_rotary_pos_emb
        thinker = bundle.model.thinker
        self.thinker = thinker
        self.lib = lib
        self.stats = stats
        self.layers = list(thinker.model.layers)
        attn0 = self.layers[0].self_attn
        self.q_rows = attn0.q_proj.out_features
        self.kv_rows = attn0.k_proj.out_features
        self.head_dim = attn0.head_dim
        hidden = attn0.q_proj.in_features
        inter = self.layers[0].mlp.gate_proj.out_features
        impl = attn0.config._attn_implementation
        self.attn_fn = ALL_ATTENTION_FUNCTIONS[impl]
        self.attn_impl = impl

        def buf(n):
            a = np.zeros(n, dtype=np.float32)
            return a, torch.from_numpy(a)

        self._x, self.x_t = buf(hidden)
        self._qkv, self.qkv_t = buf(self.q_rows + 2 * self.kv_rows)
        self._attn, self.attn_t = buf(self.q_rows)
        self._o, self.o_t = buf(hidden)
        self._gu, self.gu_t = buf(2 * inter)
        self._act, self.act_t = buf(inter)
        self._d, self.d_t = buf(hidden)
        self.o_view3 = self.o_t.view(1, 1, hidden)
        self.d_view3 = self.d_t.view(1, 1, hidden)
        self.inter = inter

        self._cb = GLUE_CB(self._glue)  # keep alive
        ptrs = [a.ctypes.data for a in
                (self._x, self._qkv, self._attn, self._o, self._gu,
                 self._act, self._d)]
        lib.driver_setup(handles, VARIANTS[variant], self._cb, ptrs)

        self._tok = torch.zeros((1, 1), dtype=torch.long)
        self._rope_probe = torch.empty(1, dtype=torch.float32)
        self.exc = None
        self.h = None
        self.K: list[torch.Tensor] = []
        self.V: list[torch.Tensor] = []
        self.cos_tab = self.sin_tab = None
        self.cos_step = self.sin_step = None

        # -- fused C decode attention (the shipped configuration) ---------
        assert attn in ("torch", "c", "c-naive"), attn
        self.attn_mode = attn
        self.attn_max_seq = attn_max_seq
        if attn != "torch":
            nlayers = len(self.layers)
            nqh = self.q_rows // self.head_dim
            nkvh = self.kv_rows // self.head_dim
            self._kring = np.zeros((nlayers, nkvh, attn_max_seq, self.head_dim),
                                   dtype=np.float32)
            self._vring = np.zeros_like(self._kring)
            self._kring_t = torch.from_numpy(self._kring)
            self._vring_t = torch.from_numpy(self._vring)
            self._norm_w = []  # keep the fp32 norm-weight tensors alive
            qw = (ctypes.c_void_p * nlayers)()
            kw = (ctypes.c_void_p * nlayers)()
            for i, layer in enumerate(self.layers):
                qn = layer.self_attn.q_norm.weight.data
                kn = layer.self_attn.k_norm.weight.data
                assert qn.dtype == torch.float32 and qn.is_contiguous()
                assert kn.dtype == torch.float32 and kn.is_contiguous()
                self._norm_w += [qn, kn]
                qw[i] = qn.data_ptr()
                kw[i] = kn.data_ptr()
            eps = float(self.layers[0].self_attn.q_norm.variance_epsilon)
            lib.attn_setup(nlayers, nqh, nkvh, self.head_dim, eps,
                           float(attn0.scaling), attn_max_seq,
                           self._kring.ctypes.data, self._vring.ctypes.data,
                           qw, kw, naive=(attn == "c-naive"))
            self._cos_c = self._sin_c = None

    # -- per-utterance ----------------------------------------------------
    def start_utterance(self, past_key_values, prompt_len: int,
                        max_new_tokens: int = 512) -> None:
        cache = past_key_values
        if hasattr(cache, "layers"):
            self.K = [layer.keys for layer in cache.layers]
            self.V = [layer.values for layer in cache.layers]
        else:  # older DynamicCache API
            self.K = list(cache.key_cache)
            self.V = list(cache.value_cache)
        rope_delta = int(self.thinker.rope_deltas.reshape(-1)[0].item())
        pos0 = prompt_len + rope_delta
        positions = torch.arange(pos0, pos0 + max_new_tokens, dtype=torch.long)
        pos_ids = positions.view(1, 1, -1).expand(3, 1, -1)
        # elementwise per position (matmul is K=1), bit-identical to the
        # per-step [3,1,1] computation the reference generate performs.
        self.cos_tab, self.sin_tab = self.thinker.model.rotary_emb(
            self._rope_probe, pos_ids)
        self.exc = None
        if self.attn_mode != "torch":
            t0 = time.perf_counter()
            seq = int(self.K[0].shape[2])  # actual prefill cache length
            assert seq + max_new_tokens <= self.attn_max_seq, (
                seq, max_new_tokens, self.attn_max_seq)
            for l in range(len(self.layers)):
                self._kring_t[l][:, :seq].copy_(self.K[l][0])
                self._vring_t[l][:, :seq].copy_(self.V[l][0])
            # effective per-step cos/sin rows [max_new, head_dim], fp32
            self._cos_c = self.cos_tab[0].to(torch.float32).contiguous()
            self._sin_c = self.sin_tab[0].to(torch.float32).contiguous()
            self.lib.attn_start(seq, self._cos_c.data_ptr(),
                                self._sin_c.data_ptr())
            self.stats.attn_s["prefill"] += time.perf_counter() - t0

    # -- torch glue callback (invoked from C) ------------------------------
    def _glue(self, l: int, stage: int) -> None:
        try:
            if stage == 0:
                if l > 0:
                    self.h.add_(self.d_view3)
                t = self.layers[l].input_layernorm(self.h)
                self.x_t.copy_(t.reshape(-1))
            elif stage == 1:
                t0 = time.perf_counter()
                attn = self.layers[l].self_attn
                nq, nkv, hd = self.q_rows, self.kv_rows, self.head_dim
                qkv = self.qkv_t
                q = attn.q_norm(qkv[:nq].view(1, 1, -1, hd)).transpose(1, 2)
                k = attn.k_norm(qkv[nq:nq + nkv].view(1, 1, -1, hd)).transpose(1, 2)
                v = qkv[nq + nkv:].view(1, 1, -1, hd).transpose(1, 2)
                q, k = self.rope_fn(q, k, self.cos_step, self.sin_step)
                K = torch.cat([self.K[l], k], dim=-2)
                V = torch.cat([self.V[l], v], dim=-2)
                self.K[l] = K
                self.V[l] = V
                out, _ = self.attn_fn(attn, q, K, V, None,
                                      dropout=0.0, scaling=attn.scaling)
                out = out.reshape(1, 1, -1).contiguous()
                self.attn_t.copy_(out.reshape(-1))
                self.stats.attn_s["decode"] += time.perf_counter() - t0
            elif stage == 2:
                self.h.add_(self.o_view3)
                t = self.layers[l].post_attention_layernorm(self.h)
                self.x_t.copy_(t.reshape(-1))
            elif stage == 3:
                t0 = time.perf_counter()
                gu = self.gu_t
                self.act_t.copy_(F.silu(gu[:self.inter]) * gu[self.inter:])
                self.stats.mlp_s["decode"] += time.perf_counter() - t0
            elif stage == 4:
                self.h.add_(self.d_view3)
        except BaseException as exc:  # noqa: BLE001 — must not cross into C
            if self.exc is None:
                import traceback
                traceback.print_exc()
                self.exc = exc

    # -- one decode step ----------------------------------------------------
    def step(self, tok: int, idx: int) -> int:
        t0 = time.perf_counter()
        self._tok[0, 0] = tok
        self.h = self.thinker.model.embed_tokens(self._tok)
        self.cos_step = self.cos_tab[:, idx:idx + 1]
        self.sin_step = self.sin_tab[:, idx:idx + 1]
        p0 = self.lib.driver_proj_seconds()
        a0 = self.lib.attn_seconds() if self.attn_mode != "torch" else 0.0
        self.lib.decode_step()
        if self.exc is not None:
            raise self.exc
        if self.attn_mode != "torch":
            self.stats.attn_s["decode"] += self.lib.attn_seconds() - a0
        self.stats.proj_decode_s += self.lib.driver_proj_seconds() - p0
        self.stats.proj_decode_calls += 112
        hn = self.thinker.model.norm(self.h)
        logits = self.thinker.lm_head(hn)
        nxt = int(torch.argmax(logits[0, -1]).item())
        self.stats.decode_s += time.perf_counter() - t0
        self.stats.decode_steps += 1
        return nxt

    def generate_greedy(self, first_token: int, prompt_len: int,
                        max_new_tokens: int = 512) -> list[int]:
        tokens = [first_token]
        tok = first_token
        step_idx = 0
        while len(tokens) < max_new_tokens and tok not in EOS_IDS:
            tok = self.step(tok, step_idx)
            tokens.append(tok)
            step_idx += 1
        return tokens
