"""Phonon CPU speech engine — loads a published model and decodes with the
transcript-gated configuration.

One ``CpuSpeech`` instance owns one loaded model plus the packed kernel
library's thread pool for the process lifetime. The execution path is the
gated packed configuration on every call: LUT packed decode projections,
one C call per decode step, int8-tiered vocabulary head with exact rescore,
fused C attention for decode and prefill, fused prefill GEMMs, and the
conv-stem chunk setting — greedy decode, temperature 0.0, max 512 new
tokens, no repetition penalty. There is no sampling path and no
non-gated fallback in this image.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# ---- the gated decode configuration (do not edit) --------------------------
EOS_IDS = (151643, 151645)
MAX_NEW_TOKENS = 512
SAMPLE_RATE = 16_000
MAX_SECONDS = 30.0            # the single-utterance envelope

# ---- the gated runtime configuration ---------------------------------------
VARIANT = "lut"               # packed-projection variant (shipped quality)
HEAD = "int8t"                # int8-tiered C head, exact top-K rescore
ATTN = "c"                    # fused C decode attention
PREFILL_ATTN = True           # batched causal C prefill attention
PREFILL_FUSE = True           # one GEMM for q/k/v and gate/up at prefill
CONV_CHUNK = 14               # encoder conv-stem chunk split
ATTN_MAX_SEQ = 2048

#: Model profile key -> how that published artifact stores its audio tower /
#: embedding (the decoder format is shared by all three). The manifest
#: format string is what `resolve_profile` checks a directory against.
PROFILES = {
    "audio6": {"manifest_format": "sttg1a-armc2-head8audio6-v1",
               "tower": "6-bit affine", "embedding": "8-bit affine"},
    "parity": {"manifest_format": "sttg1a-armc2-two-plane-ternary-v1"
                                  "+slim-metadata-v1+ten-base5-per-24bit-v1",
               "tower": "dense", "embedding": "dense"},
    "micro": {"manifest_format": "sttg1a-armc2-hybrid4-v1"
                                 "+slim-metadata-v1+ten-base5-per-24bit-v1",
              "tower": "4-bit affine", "embedding": "4-bit affine"},
}
DEFAULT_PROFILE = "audio6"


def resolve_profile(model_dir: Path, profile: str | None = None) -> str:
    """The profile key a model directory holds, cross-checked with the
    caller's claim — refuse, never guess, on a mismatch or an unknown
    layout."""
    manifest_path = Path(model_dir) / "packed_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"{model_dir} does not look like a Phonon-1 model directory "
            f"(missing packed_manifest.json)")
    manifest = json.loads(manifest_path.read_text())
    fmt = str(manifest.get("format", ""))
    found = next((k for k, v in PROFILES.items()
                  if v["manifest_format"] == fmt), None)
    if found is None:
        raise ValueError(
            "this model directory is not one of the published Phonon-1 "
            "models (Phonon-1, Phonon-1 Big, Phonon-1 Micro)")
    if profile is not None and profile != found:
        raise ValueError(
            f"the requested model is the {profile!r} build but the "
            f"directory holds {found!r} — point at the matching directory")
    return found


# ------------------------------------------------------------------ threads
def detect_cpus() -> int:
    """Usable CPU count: container quota aware, never zero."""
    quota = None
    try:  # cgroup v2
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text and text[0] != "max":
            quota = max(1, int(float(text[0]) / float(text[1])))
    except Exception:
        pass
    if quota is None:
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0 and p > 0:
                quota = max(1, q // p)
        except Exception:
            pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 1
    return max(1, min(quota or affinity, affinity))


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
        return value if value > 0 else default
    except ValueError:
        return default


def thread_plan() -> dict:
    """Kernel / torch / decode-phase torch thread counts, env-overridable.

    Kernel and torch pools default to min(cpus, 16); the decode phase runs
    Torch single-threaded (the packed kernel pool owns the cores during a
    decode step, and a spinning Torch pool would fight it under container
    CPU quotas). Overrides: PHONON_CPU_THREADS, PHONON_TORCH_THREADS,
    PHONON_DECODE_TORCH_THREADS.
    """
    cpus = detect_cpus()
    base = max(1, min(cpus, 16))
    return {
        "cpus": cpus,
        "kernel": _env_int("PHONON_CPU_THREADS", base),
        "torch": _env_int("PHONON_TORCH_THREADS", base),
        "decode_torch": _env_int("PHONON_DECODE_TORCH_THREADS", 1),
    }


# ------------------------------------------------------------------- engine
class CpuSpeech:
    """One loaded Phonon model on the packed CPU stack.

    NOT thread-safe: callers serialize `transcribe` (the server holds one
    decode lock, exactly as one GPU worker would).
    """

    def __init__(self, model_dir: str | Path, *, profile: str | None = None,
                 threads: dict | None = None, log=None):
        import torch

        log = log or (lambda message: None)
        self.model_dir = Path(model_dir)
        self.profile = resolve_profile(self.model_dir, profile)
        self.threads = threads or thread_plan()
        torch.set_num_threads(self.threads["torch"])

        from driver import (CpuDriverLibrary, DriverDecoder,
                            build_fused_handles, enable_fold4_cache,
                            install_quant_head)
        from levers import (install_c_prefill_attention, install_conv_fix,
                            install_prefill_fusion)

        started = time.perf_counter()
        loader, fold4_cache = enable_fold4_cache()
        from graph import _qwen_backend_modules
        from model import STATS, load_phonon_cpu

        _, _, processing = _qwen_backend_modules()
        self._feat_out_lengths = processing._get_feat_extract_output_lengths

        bundle = load_phonon_cpu(
            self.model_dir, variant=VARIANT,
            nthreads=self.threads["kernel"], packed_decode=False)
        lib = CpuDriverLibrary(nthreads=self.threads["kernel"])
        fused = build_fused_handles(lib, self.model_dir, loader=loader)
        decoder = DriverDecoder(bundle, lib, fused, VARIANT, STATS,
                                attn=ATTN, attn_max_seq=ATTN_MAX_SEQ)
        fold4_cache.clear()
        install_quant_head(bundle, lib, HEAD)
        if CONV_CHUNK:
            install_conv_fix(bundle, CONV_CHUNK)
        if PREFILL_FUSE:
            install_prefill_fusion(bundle)
        self.pattn_counters = (install_c_prefill_attention(
            bundle, lib, max_seq=ATTN_MAX_SEQ) if PREFILL_ATTN else None)

        self.torch = torch
        self.bundle = bundle
        self.lib = lib
        self.decoder = decoder
        self.stats = STATS
        self.model = bundle.model
        self.tokenizer = bundle.processor.tokenizer
        self.feature_extractor = bundle.processor.feature_extractor
        self.load_s = time.perf_counter() - started
        log(f"loaded {self.model_dir.name} ({self.profile}) in "
            f"{self.load_s:.1f}s — kernel threads "
            f"{self.threads['kernel']}, torch threads "
            f"{self.threads['torch']} (decode {self.threads['decode_torch']})")

    def transcribe(self, wav) -> str:
        """One gated decode of <= 30 s of 16 kHz mono float32 audio."""
        torch = self.torch
        with torch.inference_mode():
            feats = self.feature_extractor(
                wav, sampling_rate=SAMPLE_RATE, return_attention_mask=True,
                truncation=False, padding=True, return_tensors="pt")
            input_features = feats["input_features"].to(torch.float32)
            feature_attention_mask = feats["attention_mask"]
            num_audio_tokens = int(self._feat_out_lengths(
                feature_attention_mask.sum(-1)).item())
            prompt = (
                "<|im_start|>system\n<|im_end|>\n"
                "<|im_start|>user\n<|audio_start|>"
                + "<|audio_pad|>" * num_audio_tokens
                + "<|audio_end|><|im_end|>\n"
                "<|im_start|>assistant\nlanguage English<asr_text>"
            )
            input_ids = torch.tensor([self.tokenizer.encode(prompt)],
                                     dtype=torch.long)
            prompt_len = input_ids.shape[1]
            out = self.model.thinker(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                use_cache=True,
            )
            first = int(torch.argmax(out.logits[0, -1]).item())
            self.decoder.start_utterance(out.past_key_values, prompt_len,
                                         max_new_tokens=MAX_NEW_TOKENS)
            self.stats.phase = "decode"
            decode_tt = self.threads["decode_torch"]
            switch = decode_tt != self.threads["torch"]
            if switch:
                torch.set_num_threads(decode_tt)
            try:
                new_tokens = self.decoder.generate_greedy(
                    first, prompt_len, max_new_tokens=MAX_NEW_TOKENS)
            finally:
                if switch:
                    torch.set_num_threads(self.threads["torch"])
            return self.tokenizer.decode(
                torch.tensor(new_tokens), skip_special_tokens=True).strip()

    def describe(self) -> dict:
        """The decode configuration in force — the /health contract."""
        info = PROFILES[self.profile]
        return {
            "path": "packed-cpu (gated)",
            "profile": self.profile,
            "artifact": {"tower": info["tower"],
                         "embedding": info["embedding"],
                         "decoder": "packed five-value, group 128"},
            "decode": {"greedy": True, "temperature": 0.0,
                       "max_new_tokens": MAX_NEW_TOKENS,
                       "repetition_penalty": None, "eos": list(EOS_IDS)},
            "audio": {"sample_rate": SAMPLE_RATE,
                      "max_utterance_seconds": MAX_SECONDS,
                      "long_audio": "energy-gated segmentation "
                                    "(0.7 s close, 30 s cap)",
                      "language": "English"},
            "threads": dict(self.threads),
            "kernel_library": self.lib.path.name,
        }


def load(model_dir: str | Path, *, profile: str | None = None,
         log=None) -> CpuSpeech:
    return CpuSpeech(model_dir, profile=profile, log=log)
