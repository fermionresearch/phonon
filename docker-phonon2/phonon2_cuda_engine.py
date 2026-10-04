"""Phonon-2 CUDA engine for the `phonon-cuda` container.

`Transcriber` loads Phonon-2 once on an NVIDIA GPU and exposes transcribe / segment_decode / session /
transcribe_long / describe. Input is 16 kHz mono float32 audio; output is English text, with optional segment and
word timestamps. Decoding is greedy and deterministic.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MAX_SECONDS = 30.0
N_FFT, HOP, WIN, N_MELS, PREEMPH = 512, 160, 400, 128, 0.97
LOG_GUARD = 2.0 ** -24
HF_CONFIG = {
    "architectures": ["ParakeetForTDT"], "model_type": "parakeet_tdt", "blank_token_id": 8192, "vocab_size": 8193,
    "decoder_hidden_size": 640, "num_decoder_layers": 2, "hidden_act": "relu", "max_symbols_per_step": 10,
    "durations": [0, 1, 2, 3, 4], "pad_token_id": 2, "is_encoder_decoder": True, "initializer_range": 0.02,
    "encoder_config": {"model_type": "parakeet_encoder", "hidden_size": 1024, "intermediate_size": 4096,
                       "num_hidden_layers": 24, "num_attention_heads": 8, "num_key_value_heads": 8,
                       "num_mel_bins": 128, "conv_kernel_size": 9, "subsampling_conv_channels": 256,
                       "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2, "subsampling_factor": 8,
                       "max_position_embeddings": 5000, "scale_input": False, "attention_bias": False,
                       "convolution_bias": False, "hidden_act": "silu", "dropout": 0.0, "attention_dropout": 0.0,
                       "activation_dropout": 0.0, "dropout_positions": 0.0, "layerdrop": 0.0, "initializer_range": 0.02},
}


def log(message: str) -> None:
    print(f"[phonon-cuda] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- slaney mel (librosa's matrix, numpy)
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64); f_sp = 200.0 / 3; min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-30) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64); f_sp = 200.0 / 3; min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filters(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f); ramps = np.subtract.outer(mel_f, fftfreqs)
    w = np.zeros((n_mels, 1 + n_fft // 2))
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


def _is_punctuation(piece: str) -> bool:
    return bool(piece) and not any(ch.isalnum() for ch in piece)


def words_from_tokens(tokens, offset: float = 0.0, limit=None) -> list:
    """Merge `(piece, start, duration)` decoder tokens into `{text, start, end}` words: a piece with a leading space starts a
    word, a punctuation-only piece attaches to the word before it, times are shifted by `offset` and rounded to the
    millisecond; `limit` (the segment's length in seconds) caps an end at the audio, since the last token's predicted
    duration can run one 80 ms frame past it.  The same rule as the pip package (fermion/_speech/segment.py)."""
    words: list = []; cur: list = []
    def flush():
        if cur:
            text = "".join(t[0] for t in cur).strip()
            if text:
                end = cur[-1][1] + cur[-1][2]
                if limit is not None:
                    end = min(end, limit)
                words.append({"text": text, "start": round(cur[0][1] + offset, 3), "end": round(max(end, cur[0][1]) + offset, 3)})
            cur.clear()
    for tok in tokens:
        piece, start, duration = str(tok[0]), float(tok[1]), float(tok[2])
        if piece.startswith(" ") and cur and (not piece.strip() or not _is_punctuation(piece.strip())):   # a bare marker starts the next word
            flush()
        cur.append((piece, start, duration))
    flush()
    return words


def _special(piece: str) -> bool:
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in ("<unk>", "<pad>")


# ---------------------------------------------------------------- the engine
class Transcriber:
    """Loads Phonon-2 once on the GPU; greedy TDT decode, CUDA graphs, GPU mel."""

    def __init__(self, model_key: str, model_dir: Path):
        try:
            import torch
        except ImportError:
            raise SystemExit("[phonon-cuda] PyTorch is not installed inside this image (broken build?)")
        if not torch.cuda.is_available():
            raise SystemExit("[phonon-cuda] no CUDA device available. This image requires an NVIDIA GPU "
                             "(docker run --gpus all ...); on Apple silicon use the fermion-research pip package instead.")
        from transformers import ParakeetForTDT, ParakeetTDTConfig
        from fermion_container import read_container
        self.torch = torch; self.model_key = model_key; self.model_dir = Path(model_dir)
        self.entry = {"name": "Phonon-2", "repo": "FermionResearch/Phonon-2", "profile": "five-value",
                      "aliases": ("phonon-2", "phonon2", "phonon")}
        self.accepted_names = {"", "phonon", "phonon-cuda", "phonon-2", "phonon2", "fermionresearch/phonon-2"}
        self.pad_s = float(os.environ.get("PHONON2_CUDA_PAD_S", "5.0"))
        self.K = int(os.environ.get("PHONON2_CUDA_K", "16"))
        self.dtype = getattr(torch, os.environ.get("PHONON2_CUDA_DTYPE", "bfloat16"))
        started = time.perf_counter()
        log(f"loading Phonon-2 from {self.model_dir} ...")
        cfg = ParakeetTDTConfig(**HF_CONFIG)
        model = ParakeetForTDT(cfg)
        tensors, index, raw = read_container(str(self.model_dir / "model.fermion"), with_raw=True)
        sd = {}
        for k, v in tensors.items():
            if k.endswith("num_batches_tracked"):
                v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
                sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
            else:
                arr = np.ascontiguousarray(v.astype(np.float32))
                if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
                    arr = arr[:, :, None]
                sd[k] = torch.from_numpy(arr)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise SystemExit(f"[phonon-cuda] load: missing {missing[:5]} unexpected {list(unexpected)[:5]}")
        del tensors, sd
        model = model.to(self.dtype).cuda().eval()
        del raw
        self.model = model; self.cfg = cfg
        self.blank = cfg.blank_token_id; self.V = cfg.vocab_size
        self.durations = torch.tensor(cfg.durations, device="cuda", dtype=torch.long)
        self.max_sym = getattr(cfg, "max_symbols_per_step", 10) or 10
        lstm = model.decoder.lstm; self.L = lstm.num_layers; self.H = lstm.hidden_size
        self.Wih = [getattr(lstm, f"weight_ih_l{l}") for l in range(self.L)]
        self.Whh = [getattr(lstm, f"weight_hh_l{l}") for l in range(self.L)]
        self.bih = [getattr(lstm, f"bias_ih_l{l}", None) for l in range(self.L)]
        self.bhh = [getattr(lstm, f"bias_hh_l{l}", None) for l in range(self.L)]
        vocab = json.loads((self.model_dir / "config.json").read_text())["joint"]["vocabulary"]
        self.vocab = list(vocab)
        self.window = torch.hann_window(WIN, periodic=False, device="cuda")
        self.melf = torch.from_numpy(mel_filters()).cuda()
        self.graphs = {}
        # hotwords (hotwords.py, shared with the pip engines): a separate captured graph per (Tcap, state bucket, piece
        # bucket) runs the biased step; with no hotwords the graphs above run, unchanged.
        from hotwords import HotwordState
        self.hw = HotwordState(); self.graphs_b = {}; self._hw_cache = {}
        self.device_name = torch.cuda.get_device_name(0)
        self.load_s = time.perf_counter() - started
        # warm-up (PHONON2_CUDA_WARMUP=all, default): run one silent clip per padded bucket up to the 30 s segment length, so
        # one-time per-shape setup happens at load rather than on a user's first request of that length. =first runs a
        # single 2 s clip instead.  Transcripts are the same either way.
        mode = os.environ.get("PHONON2_CUDA_WARMUP", "all").lower()
        t_w = time.perf_counter()
        if mode == "all" and self.pad_s > 0:
            nb = int(math.ceil(30.0 / self.pad_s))
            for k in range(1, nb + 1):
                self.transcribe(np.zeros(int(k * self.pad_s * SAMPLE_RATE) - 1, dtype=np.float32))
            self.warm_s = time.perf_counter() - t_w; self.warm_buckets = nb
        else:
            self.transcribe(np.zeros(int(2 * SAMPLE_RATE), dtype=np.float32)); self.warm_s = time.perf_counter() - t_w; self.warm_buckets = 1
        log(f"loaded in {self.load_s:.1f}s on {self.device_name} (path=dense, "
            f"{str(self.dtype).split('.')[-1]}, CUDA graphs K={self.K}, pad bucket {self.pad_s:.0f} s; warmed {self.warm_buckets} bucket(s) in {self.warm_s:.1f}s)")

    # ---- front end (GPU) -----------------------------------------------------------------------------------
    @staticmethod
    def _bucket(n: int, pad_s: float) -> int:
        b = int(pad_s * SAMPLE_RATE)
        return int(math.ceil(max(n, 1) / b) * b)

    def _features(self, wave: np.ndarray):
        torch = self.torch
        n = len(wave); Lmax = self._bucket(n, self.pad_s) if self.pad_s > 0 else n
        x = torch.zeros((1, Lmax), device="cuda", dtype=torch.float32)
        x[0, :n] = torch.as_tensor(np.asarray(wave, dtype=np.float32), device="cuda")
        lens = torch.tensor([n], device="cuda")
        tmask = torch.arange(Lmax, device="cuda")[None, :] < lens[:, None]
        x = torch.cat([x[:, :1], x[:, 1:] - PREEMPH * x[:, :-1]], dim=1).masked_fill(~tmask, 0.0)
        st = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=self.window, return_complex=True, pad_mode="constant")
        mag = torch.view_as_real(st); mag = torch.sqrt(mag.pow(2).sum(-1)).pow(2)
        mel = torch.log(self.melf @ mag + LOG_GUARD).permute(0, 2, 1)
        flen = torch.floor_divide(lens + N_FFT // 2 * 2 - N_FFT, HOP)
        am = torch.arange(mel.shape[1], device="cuda")[None, :] < flen[:, None]
        m = am.unsqueeze(-1); melm = mel * m
        mean = (melm.sum(1) / flen.unsqueeze(-1)).unsqueeze(1)
        var = (((melm - mean) ** 2) * m).sum(1) / (flen - 1).unsqueeze(-1)
        mel = (mel - mean) / (torch.sqrt(var).unsqueeze(1) + 1e-5)
        return (mel * m).to(self.dtype), am

    # ---- decode (CUDA graph over K steps) --------------------------------------------------------------------
    def _lstm(self, x, h, cc):
        torch = self.torch; hs, cs = [], []
        for l in range(self.L):
            g = x @ self.Wih[l].T + h[l] @ self.Whh[l].T
            if self.bih[l] is not None:
                g = g + self.bih[l] + self.bhh[l]
            i, f, gg, o = g.chunk(4, -1)
            c2 = torch.sigmoid(f) * cc[l] + torch.sigmoid(i) * torch.tanh(gg)
            x = torch.sigmoid(o) * torch.tanh(c2); hs.append(x); cs.append(c2)
        return x, torch.stack(hs, 0), torch.stack(cs, 0)

    def _step(self, encp, t, last, h, cc, nsym, lens, T):
        torch = self.torch; m = self.model
        emb = m.decoder.embedding(last)
        out, h2, c2 = self._lstm(emb, h, cc)
        dec = m.decoder.decoder_projector(out)
        f = encp[0, torch.clamp(t, max=T - 1)]
        logits = m.joint.head(m.joint.activation(f + dec))
        tok = logits[:, :self.V].argmax(-1); dur = self.durations[logits[:, self.V:].argmax(-1)]
        active = t < lens; is_blank = tok == self.blank
        dur = torch.where(is_blank & (dur == 0), torch.ones_like(dur), dur)
        emit = active & ~is_blank
        nsym_n = torch.where(dur == 0, nsym + 1, torch.zeros_like(nsym))
        hit = (dur == 0) & (nsym_n >= self.max_sym)
        dur = torch.where(hit, torch.ones_like(dur), dur); nsym_n = torch.where(hit, torch.zeros_like(nsym_n), nsym_n)
        last_n = torch.where(emit, tok, last); e3 = emit[None, :, None]
        # the frame this step read (t) and the predicted duration (dur) are returned beside the token so the server can
        # place every word in time.
        return (torch.where(emit, tok, torch.full_like(tok, -1)), torch.where(active, t + dur, t), last_n,
                torch.where(e3, h2, h), torch.where(e3, c2, cc), torch.where(active, nsym_n, nsym), t, dur)

    def _step_b(self, encp, t, last, h, cc, nsym, lens, T, bstate, table, toks, col, depth, lam):
        """_step with hotword biasing: tokens that continue a listed word get a bonus of `lam` (blank is never biased)."""
        torch = self.torch; m = self.model
        emb = m.decoder.embedding(last)
        out, h2, c2 = self._lstm(emb, h, cc)
        dec = m.decoder.decoder_projector(out)
        f = encp[0, torch.clamp(t, max=T - 1)]
        logits = m.joint.head(m.joint.activation(f + dec))
        row, bias = self._hot_bias(bstate, table, toks, depth, lam)
        tok = (logits[:, :self.V].float() + bias).argmax(-1); dur = self.durations[logits[:, self.V:].argmax(-1)]
        active = t < lens; is_blank = tok == self.blank
        dur = torch.where(is_blank & (dur == 0), torch.ones_like(dur), dur)
        emit = active & ~is_blank
        nsym_n = torch.where(dur == 0, nsym + 1, torch.zeros_like(nsym))
        hit = (dur == 0) & (nsym_n >= self.max_sym)
        dur = torch.where(hit, torch.ones_like(dur), dur); nsym_n = torch.where(hit, torch.zeros_like(nsym_n), nsym_n)
        last_n = torch.where(emit, tok, last); e3 = emit[None, :, None]
        bstate_n = torch.where(emit, self._hot_next(row, col, tok, bstate), bstate)
        return (torch.where(emit, tok, torch.full_like(tok, -1)), torch.where(active, t + dur, t), last_n,
                torch.where(e3, h2, h), torch.where(e3, c2, cc), torch.where(active, nsym_n, nsym), t, dur, bstate_n)

    def _hot_bias(self, bstate, table, toks, depth, lam):
        """(row, bias): the automaton row of the current state (1, Kcap) and the (1, V) float32 bonus on the token logits."""
        torch = self.torch
        row = table[bstate]                                                    # (1, Kcap) next state per piece
        gain = (depth[row] - depth[bstate][:, None]).float().clamp(min=0) * lam  # (1, Kcap); bonus only
        bias = torch.zeros((1, self.V + 1), device=row.device, dtype=torch.float32)
        bias.index_add_(1, toks, gain)                                         # slot V = the padded columns' sink
        return row, bias[:, :self.V]

    def _hot_next(self, row, col, tok, bstate):
        torch = self.torch
        k = col[tok]
        return torch.where(k >= 0, row.gather(1, k.clamp(min=0)[:, None])[:, 0], torch.zeros_like(bstate))

    @staticmethod
    def _cap(n: int, lo: int = 64) -> int:
        c = lo
        while c < n:
            c *= 2
        return c

    def _graph_for_b(self, Tcap, Scap, Kcap):
        torch = self.torch
        key = (Tcap, Scap, Kcap)
        if key in self.graphs_b:
            return self.graphs_b[key]
        K = self.K
        S = {"encp": torch.zeros(1, Tcap, self.H, device="cuda", dtype=self.dtype), "lens": torch.zeros(1, dtype=torch.long, device="cuda"),
             "t": torch.zeros(1, dtype=torch.long, device="cuda"), "last": torch.full((1,), self.blank, dtype=torch.long, device="cuda"),
             "h": torch.zeros(self.L, 1, self.H, device="cuda", dtype=self.dtype), "c": torch.zeros(self.L, 1, self.H, device="cuda", dtype=self.dtype),
             "nsym": torch.zeros(1, dtype=torch.long, device="cuda"), "toks": torch.full((K, 1), -1, dtype=torch.long, device="cuda"),
             "frames": torch.zeros((K, 1), dtype=torch.long, device="cuda"), "durs": torch.zeros((K, 1), dtype=torch.long, device="cuda"),
             "active": torch.zeros((), dtype=torch.bool, device="cuda"),
             "bstate": torch.zeros(1, dtype=torch.long, device="cuda"), "table": torch.zeros((Scap, Kcap), dtype=torch.long, device="cuda"),
             "ptoks": torch.full((Kcap,), self.V, dtype=torch.long, device="cuda"), "col": torch.full((self.V,), -1, dtype=torch.long, device="cuda"),
             "depth": torch.zeros(Scap, dtype=torch.long, device="cuda"), "lam": torch.zeros((), dtype=torch.float32, device="cuda")}

        def block():
            t, last, h, cc, nsym, bs = S["t"], S["last"], S["h"], S["c"], S["nsym"], S["bstate"]
            for k in range(K):
                tok_out, t, last, h, cc, nsym, t_at, dur, bs = self._step_b(S["encp"], t, last, h, cc, nsym, S["lens"], Tcap, bs,
                                                                             S["table"], S["ptoks"], S["col"], S["depth"], S["lam"])
                S["toks"][k].copy_(tok_out); S["frames"][k].copy_(t_at); S["durs"][k].copy_(dur)
            S["t"].copy_(t); S["last"].copy_(last); S["h"].copy_(h); S["c"].copy_(cc); S["nsym"].copy_(nsym); S["bstate"].copy_(bs)
            S["active"].copy_((t < S["lens"]).any())
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                block()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            block()
        self.graphs_b[key] = (g, S)
        return self.graphs_b[key]

    def _arm(self, S, auto) -> None:
        torch = self.torch
        Sn, Kn = auto.table.shape
        S["table"].zero_(); S["table"][:Sn, :Kn].copy_(torch.from_numpy(auto.table.astype(np.int64)))
        S["ptoks"].fill_(self.V); S["ptoks"][:Kn].copy_(torch.from_numpy(auto.toks.astype(np.int64)))
        S["col"].copy_(torch.from_numpy(auto.col[: self.V].astype(np.int64)))
        S["depth"].zero_(); S["depth"][:Sn].copy_(torch.from_numpy(auto.depth.astype(np.int64)))
        S["lam"].fill_(float(auto.lam)); S["bstate"].zero_()

    # ---- hotwords ------------------------------------------------------------------------------------------------
    def set_hotwords(self, words, lam=None) -> int:
        """The engine's own hotword list ([] = off); returns how many words are active (at most hotwords.MAX_WORDS)."""
        n, given = self.hw.set(words, lam)
        if given > n:
            log(f"note: {given} hotwords given; the first {n} are used")
        return n

    def _automaton(self, hotwords=None, hotword_lambda=None):
        """The automaton for one call: the engine's own list when both are None, else this call's (not stored)."""
        if hotwords is None and hotword_lambda is None:
            return self.hw.automaton(self.vocab)
        from hotwords import HotwordState, clean_words
        words = self.hw.words if hotwords is None else hotwords
        lam = self.hw.lam if hotword_lambda is None else hotword_lambda
        key = (repr(clean_words(words)[0]), float(lam))
        st = self._hw_cache.get(key)
        if st is None:
            st = HotwordState(); n, given = st.set(words, lam)
            if given > n:
                log(f"note: {given} hotwords given; the first {n} are used")
            if len(self._hw_cache) > 64:
                self._hw_cache.clear()
            self._hw_cache[key] = st
        return st.automaton(self.vocab)

    def _graph_for(self, Tcap):
        torch = self.torch
        if Tcap in self.graphs:
            return self.graphs[Tcap]
        K = self.K
        S = {"encp": torch.zeros(1, Tcap, self.H, device="cuda", dtype=self.dtype), "lens": torch.zeros(1, dtype=torch.long, device="cuda"),
             "t": torch.zeros(1, dtype=torch.long, device="cuda"), "last": torch.full((1,), self.blank, dtype=torch.long, device="cuda"),
             "h": torch.zeros(self.L, 1, self.H, device="cuda", dtype=self.dtype), "c": torch.zeros(self.L, 1, self.H, device="cuda", dtype=self.dtype),
             "nsym": torch.zeros(1, dtype=torch.long, device="cuda"), "toks": torch.full((K, 1), -1, dtype=torch.long, device="cuda"),
             "frames": torch.zeros((K, 1), dtype=torch.long, device="cuda"), "durs": torch.zeros((K, 1), dtype=torch.long, device="cuda"),
             "active": torch.zeros((), dtype=torch.bool, device="cuda")}

        def block():
            t, last, h, cc, nsym = S["t"], S["last"], S["h"], S["c"], S["nsym"]
            for k in range(K):
                tok_out, t, last, h, cc, nsym, t_at, dur = self._step(S["encp"], t, last, h, cc, nsym, S["lens"], Tcap)
                S["toks"][k].copy_(tok_out); S["frames"][k].copy_(t_at); S["durs"][k].copy_(dur)
            S["t"].copy_(t); S["last"].copy_(last); S["h"].copy_(h); S["c"].copy_(cc); S["nsym"].copy_(nsym)
            S["active"].copy_((t < S["lens"]).any())
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                block()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            block()
        self.graphs[Tcap] = (g, S)
        return self.graphs[Tcap]

    #: seconds per encoder frame = subsampling 8 x hop 160 at 16 kHz
    _SUB, _HOP = 8, 160

    def transcribe(self, wav, hotwords=None, hotword_lambda=None) -> str:
        """One greedy decode of <= 30 s of audio."""
        return self.transcribe_timed(wav, hotwords, hotword_lambda)[0]

    def transcribe_timed(self, wav, hotwords=None, hotword_lambda=None, *, _auto=False):
        """(text, tokens): the text exactly as `transcribe` builds it, plus `(piece, start_s, duration_s)` per emitted token
        from the decoder's own frame index and predicted duration, relative to the start of `wav`.  `hotwords` /
        `hotword_lambda` apply to this call (None = the engine's own `set_hotwords` list; [] = none)."""
        torch = self.torch; m = self.model
        auto = self._automaton(hotwords, hotword_lambda) if _auto is False else _auto
        wav = np.asarray(wav, dtype=np.float32)
        with torch.inference_mode():
            feats, am = self._features(wav)
            enc = m.encoder(input_features=feats, attention_mask=am).last_hidden_state
            T = int(m._get_subsampling_output_length(am.sum(-1))[0]); encp = m.encoder_projector(enc)
            Tcap = ((encp.shape[1] + 63) // 64) * 64
            if auto is None:
                g, S = self._graph_for(Tcap)
            else:
                g, S = self._graph_for_b(Tcap, self._cap(auto.table.shape[0]), self._cap(auto.table.shape[1]))
                self._arm(S, auto)
            S["encp"].zero_(); S["encp"][:, :encp.shape[1]].copy_(encp); S["lens"].fill_(T)
            S["t"].zero_(); S["last"].fill_(self.blank); S["h"].zero_(); S["c"].zero_(); S["nsym"].zero_()
            toks = []; frs = []; dus = []; it = 0; max_iters = self.max_sym * T + 16
            while it < max_iters:
                g.replay(); toks.append(S["toks"].clone()); frs.append(S["frames"].clone()); dus.append(S["durs"].clone()); it += self.K
                if not bool(S["active"]):
                    break
            all_t = torch.cat(toks, 0)[:, 0].cpu().numpy(); all_f = torch.cat(frs, 0)[:, 0].cpu().numpy(); all_d = torch.cat(dus, 0)[:, 0].cpu().numpy()
            keep = all_t >= 0
            ids = [int(x) for x in all_t[keep]]; frames = [int(x) for x in all_f[keep]]; durs = [int(x) for x in all_d[keep]]
        text = "".join(self.vocab[i] for i in ids if i < len(self.vocab) and not _special(self.vocab[i])).replace("▁", " ").strip()
        tokens = [(self.vocab[i].replace("▁", " "), fr * self._SUB / SAMPLE_RATE * self._HOP, du * self._SUB / SAMPLE_RATE * self._HOP)
                  for i, fr, du in zip(ids, frames, durs) if i < len(self.vocab) and not _special(self.vocab[i])]
        return text, tokens

    # ---- the server-facing contract -----------------------------------------------------------------------------
    def segment_decode(self, wav, lock: threading.Lock | None = None, hotwords=None, hotword_lambda=None) -> str:
        return self.segment_decode_timed(wav, lock, hotwords, hotword_lambda)[0]

    def segment_decode_timed(self, wav, lock: threading.Lock | None = None, hotwords=None, hotword_lambda=None, *, _auto=False):
        if wav.size == 0 or float(np.max(np.abs(wav))) < 1e-4:
            return "", []
        if lock is None:
            return self.transcribe_timed(wav, hotwords, hotword_lambda, _auto=_auto)
        with lock:
            return self.transcribe_timed(wav, hotwords, hotword_lambda, _auto=_auto)

    def session(self, *, partials: bool, on_partial=None, on_final=None, lock: threading.Lock | None = None):
        from _live import LiveSession
        return LiveSession(lambda wav: self.segment_decode(wav, lock), on_partial=on_partial, on_final=on_final, partials=partials)

    def transcribe_long(self, wav, *, lock: threading.Lock | None = None, on_final=None, hotwords=None, hotword_lambda=None) -> tuple[str, int]:
        text, segments, _words = self.transcribe_long_timed(wav, lock=lock, on_final=on_final, hotwords=hotwords, hotword_lambda=hotword_lambda)
        return text, len(segments)

    def transcribe_long_timed(self, wav, *, lock: threading.Lock | None = None, on_final=None, hotwords=None, hotword_lambda=None):
        """(text, segments, words): `segments` = one {id, start, end, text} per decoded piece of audio (the whole file up to
        30 s, else the live session's pause-cut segments), `words` = one {text, start, end} per word in seconds from the
        start of the file (words_from_tokens, offsets from the session).  `text` is exactly `transcribe_long`'s."""
        n = len(wav)
        auto = self._automaton(hotwords, hotword_lambda)          # one automaton for every segment of this call
        if n / SAMPLE_RATE <= MAX_SECONDS:
            text, toks = self.segment_decode_timed(wav, lock, _auto=auto)
            return text, [{"id": 0, "start": 0.0, "end": round(n / SAMPLE_RATE, 3), "text": text}], words_from_tokens(toks, 0.0, n / SAMPLE_RATE)
        from _live import LiveSession
        stash: list = []; offsets: list = []
        def decode(audio):
            text, toks = self.segment_decode_timed(audio, lock, _auto=auto); stash.append(toks); return text
        def on_segment(text, start_sample, length):
            offsets.append((start_sample, length, text))
        session = LiveSession(decode, on_final=on_final, partials=False, on_segment=on_segment)
        session.feed_pcm(wav)
        text = session.finish()
        segments = []; words = []
        for idx, ((start, length, seg_text), toks) in enumerate(zip(offsets, stash)):
            off = start / SAMPLE_RATE
            segments.append({"id": idx, "start": round(off, 3), "end": round((start + length) / SAMPLE_RATE, 3), "text": seg_text})
            words.extend(words_from_tokens(toks, off, length / SAMPLE_RATE))
        return text, segments, words

    def describe(self) -> dict:
        return {"model": self.model_key, "model_name": self.entry["name"], "repo": self.entry["repo"], "profile": self.entry["profile"],
                "path": "dense fp16 (exact expansion of the container), CUDA-graph TDT, GPU mel",
                "packed_decode": False, "dtype": str(self.dtype).split(".")[-1], "cuda_graph_steps": self.K, "pad_bucket_s": self.pad_s, "warm_buckets": getattr(self, "warm_buckets", None), "warm_seconds": round(getattr(self, "warm_s", 0.0), 1),
                "decode": {"greedy": True, "temperature": 0.0, "max_symbols_per_step": self.max_sym, "repetition_penalty": None},
                "audio": {"sample_rate": SAMPLE_RATE, "max_utterance_seconds": MAX_SECONDS,
                          "long_audio": "energy-gated segmentation (0.7 s close, 30 s cap)", "language": "English"},
                "gpu": self.device_name,
                "hotwords": self.hw.plain_words(), "hotword_lambda": self.hw.lam if self.hw.words else None,
                "hotword_bias": "bonus per matching piece on the decoder's token scores; blank never biased; at most 25 words"}
