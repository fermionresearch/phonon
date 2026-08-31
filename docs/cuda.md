# Running on NVIDIA (experimental)

An experimental CUDA runtime exists for machines without Apple silicon:
English, 16 kHz input, greedy decode, NVIDIA GPU required. The supported
paths remain the Mac runtime and the `fermion` CLI. Accuracy is matched
against the Mac reference on paired test sets, but transcripts are not
word-identical across backends (floating-point reduction order differs
between Metal and CUDA).

Two forms: a bare single-utterance script, and a Docker image whose `0.2.0`
release adds long-audio transcription, live streaming over WebSocket, and
bounded request queueing (the earlier `0.1.0-preview` tag is
single-utterance transcribe + serve).

## Bare script

Full documentation: [cuda/README.md](../cuda/README.md). In short: install
`cuda/requirements-cuda.txt` plus `qwen-asr==0.0.6 --no-deps`, download the
model it targets (the largest published build; see that README), and run:

```bash
python transcribe_cuda.py audio.wav --model-dir /path/to/Phonon-1-Big
```

It expands the published packed artifact to dense BF16 at load time and
decodes end to end on the GPU. No CPU fallback in this preview.

## Docker image

Full documentation: [docker/README.md](../docker/README.md). The image
(`ghcr.io/fermionresearch/phonon-cuda:0.2.0`) needs the NVIDIA
Container Toolkit and offers two subcommands. Transcribe a file:

```bash
docker run --rm --gpus all \
  -v /path/to/Phonon-1-Big:/model -v /path/to/audio:/audio \
  ghcr.io/fermionresearch/phonon-cuda:0.2.0 \
  transcribe /audio/utterance.wav --model-dir /model
```

Or serve the same OpenAI-compatible `/v1/audio/transcriptions` endpoint
documented in [docs/server.md](server.md) (`serve --host 0.0.0.0 --port 8000
--api-key …`; non-loopback binds refuse to start without a key). Without
`--model-dir` the entrypoint downloads the model from Hugging Face. The
optional `PHONON_CUDA_PACKED=1` kernel path is experimental and not
transcript-gated; leave it off for anything that matters.

From image `0.2.0` the container also transcribes long
recordings (energy-gated segmentation mirroring the Mac engine's constants,
finals joined with single spaces), serves the `/v1/audio/stream` WebSocket
with the identical protocol to the Mac server — one client works against
both backends — and queues concurrent HTTP requests behind a bounded queue
(503 + `Retry-After` when full, never a silent hang). Enterprise deployment
notes (one GPU = one worker, TLS via reverse proxy, key management, measured
throughput) are in [docker/README.md](../docker/README.md).
