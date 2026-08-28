#!/usr/bin/env python3
"""Phonon: local streaming speech-to-text CLI for Apple silicon.

The default backend is the exact base-5-packed two-plane decoder. Microphone mode is
rolling/endpointed streaming: audio capture never stops while partial hypotheses
are decoded, and silence is rejected before it reaches the generative model.
"""

from __future__ import annotations

import argparse
import math
import queue
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent
PACKED_MODEL = ROOT / "model_v18_mlx_packed"
PARITY_MODEL = ROOT / "model_v18_mlx_quint5"
COMPACT_MODEL = ROOT / "model_v18_mlx_hybrid8_quint5"
NANO_MODEL = ROOT / "model_v18_mlx_hybrid6_quint5"
MICRO_MODEL = ROOT / "model_v18_mlx_hybrid4_quint5"
DENSE_MODEL = ROOT / "model_v18_mlx"
PARAKEET4_MODEL = ROOT / "bench" / "models" / "parakeet_0.6b_v3_4bit"
WARMUP_AUDIO = ROOT / "samples" / "1089-134691-0000.flac"
SAMPLE_RATE = 16_000
# Display-only punctuation/casing restoration.  See punctuate.py.
# Off unless --punctuate is passed.
PUNCTUATE_CHOICES = ("english", "multi", "xlmr", "rules")
PUNCTUATE_DEFAULT = "english"

# The three profiles published in the public release.  These are the only values
# advertised in `--help`.
PUBLIC_BACKENDS = ("parity", "audio6", "micro")
# Accelerated decode variants.  Derived at load time from the same on-disk
# artifacts, documented in the README, and deliberately not listed in the usage
# line to keep it readable -- the help text points at them instead.
ACCELERATED_BACKENDS = (
    "parity-fold4",
    "parity-tiered4",
    "parity-fold4-tiered4",
    "audio6-fold4",
)
INTERNAL_BACKENDS = ()
ALL_BACKENDS = PUBLIC_BACKENDS + ACCELERATED_BACKENDS + INTERNAL_BACKENDS


class Parakeet4Adapter:
    """Make the local MLX Parakeet model conform to the V18 CLI interface."""

    def __init__(self, path: Path):
        from mlx_audio.stt import load

        self.model = load(path)

    def generate(self, audio, *, stream: bool = False, **_kwargs):
        # Parakeet accepts paths and MLX arrays, while microphone/file paths in
        # this CLI pass NumPy arrays after resampling.
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio)
        return self.model.generate(audio, stream=stream)


V18_PROFILE_PATHS = {
    "parity": PARITY_MODEL,
    "compact": COMPACT_MODEL,
    "nano": NANO_MODEL,
    "micro": MICRO_MODEL,
    "packed": PACKED_MODEL,
}
# Same runtime tensors as the profile above, but through the untouched
# mlx_audio generate path.  Kept as the parity oracle for every optimization
# gate; not a product backend.
V18_LEGACY_BACKENDS = {f"{name}-legacy": path for name, path in V18_PROFILE_PATHS.items()}
# Round-3 opt-in research backends.  Each is a lossy re-encoding derived at load
# time (the artifacts on disk are unchanged) and is gated on the 400-item WER
# set, never byte-identically.  See INFERENCE_OPTIMIZATION_ROUND3_20260818.md.
V18_FOLD_BACKENDS = {
    "parity-fold4": (PARITY_MODEL, 4, None),
    "compact-fold4": (COMPACT_MODEL, 4, None),
    "parity-tiered4": (PARITY_MODEL, None, 4),
    "parity-fold4-tiered4": (PARITY_MODEL, 4, 4),
    "compact-fold4-tiered4": (COMPACT_MODEL, 4, 4),
    "audio6": (ROOT / "model_v18_mlx_head8audio6_quint5", None, None),
    "audio6-fold4": (ROOT / "model_v18_mlx_head8audio6_quint5", 4, None),
}


def prewarm_tokenizer(model_dir):
    """Start building the HF tokenizer/feature extractor on a worker thread.

    Cold start is import-bound, not weight-bound (round 3 §6).  Two costs
    dominate and they are independent: the MLX weight load, and
    ``transformers`` (~1.2 s, which drags in ``torch`` purely to resolve
    ``WhisperFeatureExtractor``).  Running them concurrently turns a sum into a
    max.  The objects produced are exactly the ones ``Model.post_load_hook``
    would have built on the main thread.
    """

    import threading

    box: dict = {}

    def build():
        try:
            import transformers
            from transformers import AutoTokenizer, WhisperFeatureExtractor

            previous = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
            try:
                box["tokenizer"] = AutoTokenizer.from_pretrained(
                    str(model_dir), trust_remote_code=True
                )
                box["feature_extractor"] = WhisperFeatureExtractor.from_pretrained(
                    str(model_dir)
                )
            finally:
                transformers.logging.set_verbosity(previous)
        except Exception as exc:  # fall back to the normal in-line hook
            box["error"] = exc

    thread = threading.Thread(target=build, daemon=True, name="phonon-tokenizer")
    thread.start()
    return thread, box


def load_model(backend: str = "parity", *, fast_boot: bool = False):
    """Load a backend.  ``fast_boot`` only changes *when* work happens.

    It (a) skips two eager upstream package ``__init__`` bodies the inference
    path never uses and (b) overlaps the tokenizer build with the weight load.
    The resulting model object is the same one either way; ``fast_boot`` is
    opt-in so the CLI default is bit-for-bit the path every gate was run on.
    """

    started = time.perf_counter()
    prewarm = None
    if fast_boot:
        try:
            import fast_import

            fast_import.enable_and_verify()
        except Exception:
            pass
        directory = V18_PROFILE_PATHS.get(backend)
        if directory is None and backend in V18_FOLD_BACKENDS:
            directory = V18_FOLD_BACKENDS[backend][0]
        if directory is None:
            directory = V18_LEGACY_BACKENDS.get(backend)
        if directory is not None:
            thread, box = prewarm_tokenizer(directory)
            box["model_dir"] = directory
            prewarm = (thread, box)
            _install_deferred_hook(box)
    try:
        model, elapsed = _load_model_inner(backend, started)
    finally:
        if prewarm is not None:
            _restore_hook()
    return model, elapsed


def _install_deferred_hook(box):
    """Make ``post_load_hook`` consume the prewarmed objects instead of
    building its own.  Restored immediately after the load."""
    from mlx_audio.stt.models.qwen3_asr import Model

    global _REAL_POST_LOAD_HOOK
    if _REAL_POST_LOAD_HOOK is None:
        _REAL_POST_LOAD_HOOK = Model.post_load_hook

    def hook(model, model_path):
        import threading as _t

        for thread in _t.enumerate():
            if thread.name == "phonon-tokenizer":
                thread.join()
        if "tokenizer" in box and "feature_extractor" in box:
            # ``Model`` is a thin wrapper that delegates the real hook to
            # ``model._model``; attach to the same object it would have.
            target = getattr(model, "_model", model)
            target._tokenizer = box["tokenizer"]
            target._feature_extractor = box["feature_extractor"]
            return model
        return _REAL_POST_LOAD_HOOK(model, model_path)

    Model.post_load_hook = staticmethod(hook)


def _restore_hook():
    if _REAL_POST_LOAD_HOOK is None:
        return
    from mlx_audio.stt.models.qwen3_asr import Model

    Model.post_load_hook = _REAL_POST_LOAD_HOOK


_REAL_POST_LOAD_HOOK = None


def _load_model_inner(backend: str, started: float):
    if backend in ("parity-opt", "parity-fixed", "parity-opt-fused"):
        from optimized_v18 import load_optimized_v18

        model = load_optimized_v18(
            PARITY_MODEL,
            fused_qmv=backend == "parity-opt-fused",
            compiled_decode=backend == "parity-fixed",
        )
    elif backend in V18_FOLD_BACKENDS:
        from optimized_v18 import load_optimized_v18

        path, fold_bits, tiered_bits = V18_FOLD_BACKENDS[backend]
        model = load_optimized_v18(path, fold_bits=fold_bits,
                                   tiered_head_bits=tiered_bits)
    elif backend in V18_PROFILE_PATHS:
        # Prompt-head elimination is the default for every shipping V18
        # profile: exact greedy decoding, full 151,936 vocabulary, no
        # projection of audio-prefix positions through the tied head.
        from optimized_v18 import load_optimized_v18

        model = load_optimized_v18(V18_PROFILE_PATHS[backend])
    elif backend in V18_LEGACY_BACKENDS:
        from v18_runtime import load_v18

        model = load_v18(V18_LEGACY_BACKENDS[backend])
    elif backend == "dense":
        from mlx_audio.stt import load

        model = load(str(DENSE_MODEL), strict=True)
    elif backend == "parakeet4":
        model = Parakeet4Adapter(PARAKEET4_MODEL)
    else:
        raise ValueError(f"unknown backend: {backend}")
    return model, time.perf_counter() - started


def token_budget(seconds: float) -> int:
    # English speech rarely needs >5 decoder tokens/s.  The modest headroom is
    # deliberate: it stops a bad silence/loop decode before it can run away.
    return max(24, min(384, int(math.ceil(seconds * 7.0)) + 20))


def transcribe_array(model, audio: np.ndarray, *, repetition_penalty: float = 1.05):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    seconds = len(audio) / SAMPLE_RATE
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4:
        from mlx_audio.stt.models.base import STTOutput

        return STTOutput(text="", segments=[], language=["English"]), 0.0
    started = time.perf_counter()
    output = model.generate(
        audio,
        language="English",
        max_tokens=token_budget(seconds),
        temperature=0.0,
        repetition_penalty=repetition_penalty,
        repetition_context_size=96,
        min_chunk_duration=0.5,
    )
    mx.synchronize()
    wall = time.perf_counter() - started
    return output, wall


def transcribe_file(model, path: str, *, stream_tokens: bool = False):
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)

    if stream_tokens:
        started = time.perf_counter()
        first = None
        for token in model.generate(
            audio,
            language="English",
            max_tokens=token_budget(len(audio) / SAMPLE_RATE),
            temperature=0.0,
            repetition_penalty=1.05,
            stream=True,
            min_chunk_duration=0.5,
        ):
            if first is None:
                first = time.perf_counter()
            piece = token.text if hasattr(token, "text") else str(token)
            print(piece, end="", flush=True)
        print()
        end = time.perf_counter()
        return None, end - started, None if first is None else first - started

    output, wall = transcribe_array(model, audio)
    return output, wall, None


def warm_up(model) -> float:
    if not WARMUP_AUDIO.exists():
        return 0.0
    started = time.perf_counter()
    # This compiles/caches the exact Metal graph without feeding model silence.
    model.generate(
        str(WARMUP_AUDIO),
        language="English",
        max_tokens=32,
        temperature=0.0,
        repetition_penalty=1.05,
    )
    mx.synchronize()
    return time.perf_counter() - started


def _looping_or_implausible(text: str, seconds: float) -> bool:
    words = text.lower().split()
    if len(words) > max(18, int(seconds * 5.8) + 8):
        return True
    if len(words) < 8:
        return False
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) < len(grams) * 0.72


def _clear_line() -> None:
    print("\r\033[2K", end="", flush=True)


# --------------------------------------------------------------------------
# Silence-hallucination guard (microphone path only).
#
# The decoder cannot represent "no speech" (6/6 raw hallucination failures in
# the held-out suite), so a breath, a mouth click or a keyboard tap that trips the
# VAD can reach the decoder and be finalized as a one-word utterance -- "it",
# "in", "the".  The VAD/endpoint/repetition guards contain most of this; this
# guard closes the residue at the *product* layer.
#
# It is deliberately the narrowest rule that covers the observed defect and it
# only ever fires on the conjunction of two independent signals:
#
#   (a) the finalized text is 1-2 words, all of which are in a fixed, closed
#       lexicon of function words and non-lexical fillers -- nothing a user
#       would ever dictate as a standalone utterance; and
#   (b) the finalized audio contains less than `min_voiced_ms` of energy at or
#       above the VAD's own speech threshold, i.e. the segment never actually
#       held speech-level energy for a plausible word duration.
#
# Repetition is recorded for diagnosis but is NOT part of the rule: a
# repetition-only rule would eat a real "yes. yes. yes." dictation, and a rule
# that waits for a repeat would let the first hallucination through.
#
# Words a user dictates as commands or answers -- yes/no/okay/stop/yeah/sure/
# next/delete/undo/send/hello/hi/right/wait/done -- are deliberately absent
# from the lexicon and can never be suppressed, whatever the energy.
#
# *** OFF BY DEFAULT.  See SILENCE_GUARD_20260817.md. ***
#
# The measured verdict is DO-NOT-SHIP-BY-DEFAULT.  `voiced_ms` counts frames
# above an ABSOLUTE VAD floor, so it moves with microphone gain and speaker
# distance -- variables the guard cannot observe.  Measured on the mic path:
# the shortest real one-word finalization falls 100 -> 80 -> 60 ms as capture
# peak falls 0.25 -> 0.10 -> 0.05, while non-speech hallucinations occupy
# 20-260 ms.  The distributions overlap and slide into each other, so no
# threshold both covers the defect and is safe.  At 200 ms it suppresses 10/13
# hallucinations but silently deletes a real spoken "the" and "it"; at 60 ms
# (the setting below) it is false-positive-free on all 90 measured real
# utterances but reaches only 7/13 hallucinations, with zero margin at quiet
# gain.  Silently deleting a dictated word is worse than a visible spurious
# line, so this stays opt-in until the signal is spectral rather than absolute.
# --------------------------------------------------------------------------
SILENCE_GUARD_LEXICON = frozenset(
    {
        # closed-class function words: never meaningful as a lone dictation
        "a", "an", "and", "as", "at", "be", "but", "by", "can", "do", "for",
        "he", "her", "him", "his", "i", "if", "in", "is", "it", "its", "me",
        "my", "of", "on", "or", "she", "so", "that", "the", "them", "then",
        "there", "they", "this", "to", "was", "we", "were", "what", "when",
        "which", "who", "will", "with", "would", "you", "your",
        # non-lexical fillers and breath artifacts
        "ah", "eh", "er", "hm", "hmm", "huh", "mm", "mmm", "oh", "uh", "uhm",
        "um",
        # observed whisper-class silence hallucinations
        "applause", "bye", "canto", "english", "language", "music", "thank",
        "thanks",
    }
)
SILENCE_GUARD_MAX_WORDS = 2
# The only operating point that was false-positive-free on every measured real
# utterance, at every measured capture gain.  Covers 7/13 hallucinations.
SILENCE_GUARD_MIN_VOICED_MS = 60.0
SILENCE_GUARD_FRAME_MS = 20.0
_GUARD_WORD_RE = re.compile(r"[a-z']+")


def _voiced_profile(audio: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """Return (voiced_ms, peak_frame_rms, segment_ms) at the VAD's threshold."""
    hop = max(1, int(SAMPLE_RATE * SILENCE_GUARD_FRAME_MS / 1000))
    count = len(audio) // hop
    segment_ms = 1000.0 * len(audio) / SAMPLE_RATE
    if count == 0:
        return 0.0, float(np.sqrt(np.mean(audio * audio) + 1e-12)) if len(audio) else 0.0, segment_ms
    frames = audio[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    voiced_ms = float(np.count_nonzero(rms >= threshold)) * SILENCE_GUARD_FRAME_MS
    return voiced_ms, float(rms.max()), segment_ms


def silence_guard_decision(
    audio: np.ndarray,
    text: str,
    vad_threshold: float,
    *,
    min_voiced_ms: float = SILENCE_GUARD_MIN_VOICED_MS,
) -> tuple[bool, dict]:
    """Decide whether a finalized segment is a silence hallucination.

    Pure function: no I/O, no model access.  Returns ``(suppress, stats)`` and
    the stats are what the guard logs, so every suppression is diagnosable.
    """
    words = _GUARD_WORD_RE.findall(text.lower())
    voiced_ms, peak_rms, segment_ms = _voiced_profile(
        np.asarray(audio, dtype=np.float32).reshape(-1), vad_threshold
    )
    lexical = bool(words) and len(words) <= SILENCE_GUARD_MAX_WORDS and all(
        word in SILENCE_GUARD_LEXICON for word in words
    )
    low_energy = voiced_ms < min_voiced_ms
    stats = {
        "text": text,
        "words": len(words),
        "lexical_match": lexical,
        "low_energy": low_energy,
        "voiced_ms": round(voiced_ms, 1),
        "min_voiced_ms": min_voiced_ms,
        "peak_frame_rms": round(peak_rms, 6),
        "vad_threshold": round(vad_threshold, 6),
        "segment_ms": round(segment_ms, 1),
    }
    return (lexical and low_energy), stats


def load_punctuator(name: str | None):
    """Opt-in display polish.  Returns None when the flag is not used.

    The returned callable is applied to *displayed* text only.  Nothing that
    feeds WER gates, benchmarks or receipts ever passes through it.
    """
    if not name:
        return None
    try:
        import punctuate
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(
            f"[punctuate] disabled: {exc}. Install with "
            f"`./.venv/bin/pip install -r requirements-punctuate.txt`.",
            file=sys.stderr,
        )
        return None
    started = time.perf_counter()
    punctuator = punctuate.load(name)
    print(
        f"Punctuation/casing pass '{name}' ready in "
        f"{time.perf_counter() - started:.2f}s (display text only).",
        flush=True,
    )
    return punctuator


def run_microphone(
    model,
    *,
    device: int | str | None,
    block_ms: int,
    first_partial_ms: int,
    partial_ms: int,
    endpoint_ms: int,
    min_rms: float,
    max_utterance_s: float,
    punctuator=None,
    silence_guard: bool = False,
    silence_guard_min_voiced_ms: float = SILENCE_GUARD_MIN_VOICED_MS,
    guard_trace: bool = False,
) -> None:
    import sounddevice as sd

    block = max(160, int(SAMPLE_RATE * block_ms / 1000))
    first_partial_frames = int(SAMPLE_RATE * first_partial_ms / 1000)
    partial_frames = int(SAMPLE_RATE * partial_ms / 1000)
    endpoint_blocks = max(1, endpoint_ms // block_ms)
    max_frames = int(max_utterance_s * SAMPLE_RATE)
    incoming: queue.Queue[np.ndarray] = queue.Queue()
    pre_roll: deque[np.ndarray] = deque(maxlen=max(1, 240 // block_ms))
    utterance: list[np.ndarray] = []
    utterance_frames = 0
    frames_at_decode = 0
    silent_blocks = 0
    noise_rms = 0.0015
    vad_threshold = min_rms
    speaking = False
    speech_started_at = 0.0
    first_hypothesis_at: float | None = None
    last_partial = ""

    def callback(indata, frames, timing, status):
        del frames, timing
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        incoming.put(indata[:, 0].copy())

    def decode(final: bool = False) -> None:
        nonlocal frames_at_decode, first_hypothesis_at, last_partial
        if utterance_frames < first_partial_frames:
            return
        audio = np.concatenate(utterance)
        output, wall = transcribe_array(model, audio)
        text = output.text.strip()
        seconds = len(audio) / SAMPLE_RATE
        if _looping_or_implausible(text, seconds):
            if final:
                print("\n[guard] rejected an implausible/repetitive decode")
            return
        if final and (silence_guard or guard_trace) and text:
            # Product-layer silence-hallucination guard.  Final decodes only:
            # partial emissions are never inspected or altered.
            suppress, guard_stats = silence_guard_decision(
                audio, text, vad_threshold, min_voiced_ms=silence_guard_min_voiced_ms
            )
            if guard_trace:
                print(f"[guard-trace] {guard_stats}", file=sys.stderr, flush=True)
            if suppress and silence_guard:
                print(
                    "[silence-guard] suppressed non-speech finalization: "
                    f"text={text!r} voiced_ms={guard_stats['voiced_ms']} "
                    f"(< {guard_stats['min_voiced_ms']}) "
                    f"peak_frame_rms={guard_stats['peak_frame_rms']} "
                    f"vad_threshold={guard_stats['vad_threshold']} "
                    f"segment_ms={guard_stats['segment_ms']}",
                    file=sys.stderr,
                    flush=True,
                )
                text = ""
        frames_at_decode = utterance_frames
        if text and first_hypothesis_at is None:
            first_hypothesis_at = time.perf_counter()
        if final:
            _clear_line()
            if text:
                ttft = (
                    None
                    if first_hypothesis_at is None
                    else first_hypothesis_at - speech_started_at
                )
                timing = f"decode {wall * 1000:.0f} ms"
                if ttft is not None:
                    timing += f", first partial {ttft * 1000:.0f} ms"
                display = text
                if punctuator is not None:
                    # Display polish only, and only once the utterance is
                    # committed: partials stay raw so streaming latency and the
                    # recognized word sequence are untouched.
                    polish_started = time.perf_counter()
                    display = punctuator(text)
                    timing += f", punctuate {(time.perf_counter() - polish_started) * 1000:.0f} ms"
                print(f"{display}\n\033[2m[{timing}]\033[0m")
            else:
                print("\033[2m[no speech recognized]\033[0m")
        elif text != last_partial:
            _clear_line()
            print(f"\033[36m{text}\033[0m", end="", flush=True)
        last_partial = text

    print("Listening. Speak naturally; pause to finalize. Ctrl-C stops.")
    print(
        f"16 kHz mono | first partial >= {first_partial_ms} ms | "
        f"endpoint {endpoint_ms} ms"
    )
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block,
            device=device,
            callback=callback,
            latency="low",
        ):
            while True:
                frame = incoming.get()
                rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                threshold = max(min_rms, noise_rms * 3.2)
                vad_threshold = threshold
                is_voice = rms >= threshold

                if not speaking:
                    if is_voice:
                        speaking = True
                        speech_started_at = time.perf_counter()
                        first_hypothesis_at = None
                        last_partial = ""
                        utterance = list(pre_roll) + [frame]
                        utterance_frames = sum(len(x) for x in utterance)
                        frames_at_decode = 0
                        silent_blocks = 0
                        _clear_line()
                        print("\033[2m[speech]\033[0m", end="", flush=True)
                    else:
                        noise_rms = noise_rms * 0.985 + rms * 0.015
                        pre_roll.append(frame)
                    continue

                utterance.append(frame)
                utterance_frames += len(frame)
                silent_blocks = silent_blocks + 1 if not is_voice else 0

                should_partial = (
                    utterance_frames >= first_partial_frames
                    and utterance_frames - frames_at_decode >= partial_frames
                    and silent_blocks < endpoint_blocks
                )
                if should_partial:
                    decode(final=False)

                endpoint = silent_blocks >= endpoint_blocks
                forced = utterance_frames >= max_frames
                if endpoint or forced:
                    decode(final=True)
                    reset_stream = getattr(model, "reset_stream", None)
                    if reset_stream is not None:
                        reset_stream()
                    speaking = False
                    utterance = []
                    utterance_frames = 0
                    frames_at_decode = 0
                    silent_blocks = 0
                    pre_roll.clear()
    except KeyboardInterrupt:
        _clear_line()
        print("Stopped.")


def run_benchmark(model, path: str, runs: int) -> None:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError("benchmark sample must be 16 kHz")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    seconds = len(audio) / SAMPLE_RATE
    rows = []
    expected = None
    for index in range(runs):
        output, wall = transcribe_array(model, audio)
        expected = expected or output.text
        if output.text != expected:
            raise RuntimeError("nondeterministic transcript during benchmark")
        rows.append(wall)
        print(
            f"run {index + 1}: {wall * 1000:.1f} ms, "
            f"RTF {wall / seconds:.3f}, {output.text!r}"
        )
    steady = rows[1:] if len(rows) > 1 else rows
    print(
        f"steady median {np.median(steady) * 1000:.1f} ms | "
        f"{seconds / np.median(steady):.1f}x realtime | "
        f"active {mx.get_active_memory() / 1e9:.2f} GB | "
        f"peak {mx.get_peak_memory() / 1e9:.2f} GB"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Apple-Silicon STT")
    parser.add_argument(
        "--backend",
        # Validation still covers every backend; only the public four are
        # displayed.  `metavar` controls the help text, `choices` controls what
        # is accepted, so the accelerated and internal backends stay callable
        # without crowding the usage line.
        choices=ALL_BACKENDS,
        metavar="{" + ",".join(PUBLIC_BACKENDS) + "}",
        default="parity",
        help=(
            "deployment profile (default: parity). Accelerated decode variants "
            "are also accepted: " + ", ".join(ACCELERATED_BACKENDS) + " -- see "
            "the README."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_punctuate(target: argparse.ArgumentParser) -> None:
        # Opt-in, off by default, and deliberately absent from `benchmark`:
        # nothing that produces a measured number is ever polished.
        target.add_argument(
            "--punctuate",
            nargs="?",
            const=PUNCTUATE_DEFAULT,
            default=None,
            choices=PUNCTUATE_CHOICES,
            metavar="MODEL",
            help=(
                "restore punctuation and capitalization on displayed text only; "
                f"choices {'/'.join(PUNCTUATE_CHOICES)} (default {PUNCTUATE_DEFAULT})"
            ),
        )

    transcribe = sub.add_parser("transcribe", help="transcribe an audio file")
    transcribe.add_argument("audio")
    transcribe.add_argument("--stream-tokens", action="store_true")
    add_punctuate(transcribe)

    bench = sub.add_parser("benchmark", help="benchmark a local audio file")
    bench.add_argument("audio", nargs="?", default=str(WARMUP_AUDIO))
    bench.add_argument("--runs", type=int, default=5)

    mic = sub.add_parser("mic", help="live rolling microphone transcription")
    mic.add_argument("--device", default=None)
    mic.add_argument("--block-ms", type=int, default=20)
    mic.add_argument("--first-partial-ms", type=int, default=800)
    mic.add_argument("--partial-ms", type=int, default=550)
    mic.add_argument("--endpoint-ms", type=int, default=700)
    mic.add_argument("--min-rms", type=float, default=0.007)
    mic.add_argument("--max-utterance-s", type=float, default=30.0)
    # Product mic path only.  `transcribe` and `benchmark` never construct the
    # guard: nothing that produces a measured number can be suppressed.
    # OFF by default -- it did not clear its false-positive gate.  See
    # SILENCE_GUARD_20260817.md.
    mic.add_argument(
        "--silence-guard",
        dest="silence_guard",
        action="store_true",
        help=(
            "EXPERIMENTAL, off by default: suppress a finalization that is 1-2 "
            "closed-class words AND came from a segment with less than "
            f"{SILENCE_GUARD_MIN_VOICED_MS:.0f} ms of speech-level energy. "
            "Covers 7/13 measured silence hallucinations; it can silently drop "
            "a genuinely dictated short function word at low microphone gain. "
            "Every suppression is logged to stderr"
        ),
    )
    mic.add_argument(
        "--no-silence-guard",
        dest="silence_guard",
        action="store_false",
        help="explicitly disable the silence guard (already the default)",
    )
    mic.add_argument(
        "--silence-guard-min-voiced-ms",
        type=float,
        default=SILENCE_GUARD_MIN_VOICED_MS,
        help=(
            "energy arm of the silence guard: suppress only below this many "
            "milliseconds of speech-level energy (higher covers more "
            "hallucinations and deletes more real words)"
        ),
    )
    mic.set_defaults(silence_guard=False)
    add_punctuate(mic)

    sub.add_parser("devices", help="list microphone devices")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    label = "Parakeet 4-bit" if args.backend == "parakeet4" else f"Phonon {args.backend}"
    print(f"Loading {label} MLX/Metal backend…", flush=True)
    model, load_seconds = load_model(args.backend)
    warm_seconds = warm_up(model)
    print(
        f"Ready in {load_seconds + warm_seconds:.2f}s "
        f"(model {mx.get_active_memory() / 1e9:.2f} GB, Metal warmed)."
    )

    punctuator = load_punctuator(getattr(args, "punctuate", None))

    if args.command == "transcribe":
        output, wall, first = transcribe_file(
            model, args.audio, stream_tokens=args.stream_tokens
        )
        if output is not None:
            print(output.text if punctuator is None else punctuator(output.text))
        suffix = f", first token {first * 1000:.0f} ms" if first is not None else ""
        print(f"[{wall * 1000:.0f} ms{suffix}]", file=sys.stderr)
    elif args.command == "benchmark":
        run_benchmark(model, args.audio, args.runs)
    elif args.command == "mic":
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        run_microphone(
            model,
            device=device,
            block_ms=args.block_ms,
            first_partial_ms=args.first_partial_ms,
            partial_ms=args.partial_ms,
            endpoint_ms=args.endpoint_ms,
            min_rms=args.min_rms,
            max_utterance_s=args.max_utterance_s,
            punctuator=punctuator,
            silence_guard=args.silence_guard,
            silence_guard_min_voiced_ms=args.silence_guard_min_voiced_ms,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
