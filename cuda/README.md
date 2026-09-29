# Phonon-1 on NVIDIA

This directory runs the full Phonon-1 model on an NVIDIA GPU with a single
script.

## What it is

`transcribe_cuda.py` loads the published
[Phonon-1-Big](https://huggingface.co/FermionResearch/Phonon-1-Big) artifact,
unchanged, and decodes it end-to-end on CUDA. The decoder weights are
expanded from the published artifact at load time
(`phonon_cuda_artifact.py`), and the projections run as dense BF16.

Transcripts can differ by a word here and there from other backends.

## Usage

```bash
pip install -r requirements-cuda.txt
pip install qwen-asr==0.0.6 --no-deps   # official Qwen3-ASR graph, Apache-2.0

# download the model, then:
python transcribe_cuda.py audio.wav --model-dir /path/to/Phonon-1-Big
```

Current limits:

- NVIDIA GPU required; no CPU fallback.
- Single utterances up to 30 seconds, 16 kHz input, English.
- Batch size 1, greedy decoding.

## The Docker container

This script stays single-utterance (≤ 30 s). The Docker image
([docker/README.md](../docker/README.md)) adds long-audio transcription
(energy-gated segmentation), live streaming over the `/v1/audio/stream`
WebSocket protocol, and bounded request queueing for concurrent clients, all
on the same dense decode path.

## Other models

This loader runs Phonon-1-Big. The smaller builds (Phonon-1 and
Phonon-1-Micro) run in the Docker image.
