# Phonon-1 on NVIDIA — experimental CUDA preview

**Status: experimental.** This directory runs the full Phonon-1 model on an
NVIDIA GPU. It is the first non-Apple runtime for Phonon and is deliberately
narrow; the supported paths remain the Mac runtime in the repository root.

## What it is

`transcribe_cuda.py` loads the **parity build** — the published
[Phonon-1-Big](https://huggingface.co/FermionResearch/Phonon-1-Big) artifact,
unchanged — and decodes it end-to-end on CUDA. The packed decoder
weights are expanded from the published artifact at load time by the same
derivation the Mac runtime uses (`phonon_cuda_artifact.py`, verified
byte-identical to the MLX derivation on every decoder shape); the projections
then run as dense BF16 reconstructed from those codes.

## Verified

On an A100, against the published Mac reference for the same checkpoint, same
utterances, same greedy decode: **the full LibriSpeech test set — all 5,559
utterances (test-clean + test-other) — scores 4.12% WER on CUDA vs 4.19% on
the Mac reference** (2.65/5.60 vs 2.67/5.72 by split), scored identically
with the Whisper English text normalizer. No degenerate decodes: 0 runaway
generations, 0 repetition loops across the set.

One honest caveat: transcripts across backends are **same-accuracy, not
word-identical** — BF16 floating-point reduction order differs between Metal
and CUDA, which flips occasional borderline word choices on hard audio in
both directions (90.2% of the 5,559 transcripts matched the Mac output
word-for-word; the rest scored equivalently against the references).

## Usage

```bash
pip install -r requirements-cuda.txt
pip install qwen-asr==0.0.6 --no-deps   # official Qwen3-ASR graph, Apache-2.0

# download the model, then:
python transcribe_cuda.py audio.wav --model-dir /path/to/Phonon-1-Big
```

Current limits:

- NVIDIA GPU required; no CPU fallback in this preview.
- Single utterances up to 30 seconds, 16 kHz input, English.
- Batch-size 1, greedy decoding, unoptimized (no CUDA graphs yet).

## The Docker container

This bare script deliberately stays single-utterance (≤ 30 s). The Docker
image ([docker/README.md](../docker/README.md)) is where the serving
capabilities live: its `0.2.0` release adds long-audio
transcription (energy-gated segmentation mirroring the Mac engine), live
streaming over the same `/v1/audio/stream` WebSocket protocol as the Mac
server, and bounded request queueing for concurrent clients, all on the same
dense decode path.

## What follows

The smaller builds (Phonon-1 / Phonon-1-Micro store a pre-quantized audio
tower this loader does not yet mount) are follow-up work.
