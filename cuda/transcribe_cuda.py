#!/usr/bin/env python3
"""Phonon-1 on NVIDIA — experimental CUDA transcriber (parity build).

Usage:
    python transcribe_cuda.py audio.wav --model-dir /path/to/Phonon-1-Big

The model directory is a local download of
https://huggingface.co/FermionResearch/Phonon-1-Big (the parity build —
the published five-state packed artifact; this script derives the runtime
weights from it at load, exactly like the Mac runtime does).

Scope of this preview, deliberately narrow:
  * NVIDIA GPU required (no CPU fallback).
  * Single utterances up to 30 seconds, 16 kHz audio (mono or stereo).
  * Greedy decoding, English. Identical decode configuration to the gated
    benchmark runs (temperature 0.0, max 512 tokens, no repetition penalty).
Longer audio, streaming, other sample rates, and a Docker image follow.

Transcripts go to stdout (one line per input file); everything else to
stderr.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

MAX_SECONDS = 30.0
SAMPLE_RATE = 16_000
EOS_IDS = (151643, 151645)


def fail(message: str) -> "NoReturn":  # noqa: F821
    print(f"transcribe_cuda: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_waveform(path: Path):
    import numpy as np
    import soundfile as sf

    try:
        wav, sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        fail(f"could not read {path}: {exc}")
    if sr != SAMPLE_RATE:
        fail(
            f"{path}: sample rate is {sr} Hz; this preview requires 16 kHz.\n"
            f"  Convert first, e.g.: ffmpeg -i {path.name} -ar 16000 -ac 1 out.wav"
        )
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    duration = len(wav) / SAMPLE_RATE
    if duration > MAX_SECONDS:
        fail(
            f"{path}: {duration:.1f}s audio; this preview handles single "
            f"utterances up to {MAX_SECONDS:.0f}s. Long-form chunking ships "
            "in the follow-up release."
        )
    return np.ascontiguousarray(wav, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phonon-1 experimental CUDA transcriber (parity build)."
    )
    parser.add_argument("audio", nargs="+", help="16 kHz audio file(s), <= 30 s each")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="local download of FermionResearch/Phonon-1-Big",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "packed_manifest.json").is_file():
        fail(
            f"{model_dir} does not look like a Phonon-1-Big download "
            "(missing packed_manifest.json)"
        )
    paths = [Path(p) for p in args.audio]
    for path in paths:
        if not path.is_file():
            fail(f"no such file: {path}")

    try:
        import torch
    except ImportError:
        fail("PyTorch is not installed; see cuda/requirements-cuda.txt")
    if not torch.cuda.is_available():
        fail(
            "no CUDA device available. This preview requires an NVIDIA GPU; "
            "on Apple silicon use the MLX runtime in the repository root."
        )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phonon_cuda_model import _qwen_backend_modules, load_phonon_cuda

    _, _, processing = _qwen_backend_modules()
    feat_out_lengths = processing._get_feat_extract_output_lengths

    started = time.perf_counter()
    print(f"loading {model_dir.name} ...", file=sys.stderr)
    bundle = load_phonon_cuda(model_dir, device="cuda")
    # Match the Mac runtime's numerics class: FP32 activations through the
    # BF16-valued audio tower (exact upcast), BF16 decoder.
    bundle.model.thinker.audio_tower.float()
    model = bundle.model
    tokenizer = bundle.processor.tokenizer
    feature_extractor = bundle.processor.feature_extractor
    print(
        f"loaded in {time.perf_counter() - started:.1f}s on "
        f"{torch.cuda.get_device_name(0)}",
        file=sys.stderr,
    )

    @torch.inference_mode()
    def transcribe(wav) -> str:
        feats = feature_extractor(
            wav,
            sampling_rate=SAMPLE_RATE,
            return_attention_mask=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        input_features = feats["input_features"].to("cuda", torch.float32)
        feature_attention_mask = feats["attention_mask"].to("cuda")
        num_audio_tokens = int(
            feat_out_lengths(feature_attention_mask.sum(-1)).item()
        )
        prompt = (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|>"
            + "<|audio_pad|>" * num_audio_tokens
            + "<|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\nlanguage English<asr_text>"
        )
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device="cuda"
        )
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=512,
            eos_token_id=list(EOS_IDS),
            pad_token_id=151643,
        )
        sequences = (
            generated.sequences if hasattr(generated, "sequences") else generated
        )
        new_tokens = sequences[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for path in paths:
        wav = load_waveform(path)
        t0 = time.perf_counter()
        text = transcribe(wav)
        print(
            f"{path.name}: {len(wav) / SAMPLE_RATE:.1f}s audio in "
            f"{time.perf_counter() - t0:.2f}s",
            file=sys.stderr,
        )
        print(text, flush=True)


if __name__ == "__main__":
    main()
