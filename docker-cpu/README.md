# Phonon-1 CPU Docker image

Runs [Phonon-1](https://huggingface.co/FermionResearch/Phonon-1),
[Phonon-1 Big](https://huggingface.co/FermionResearch/Phonon-1-Big) and
[Phonon-1 Micro](https://huggingface.co/FermionResearch/Phonon-1-Micro) on
plain x86-64 CPUs (AVX2 — any mainstream CPU from roughly 2014 on). No GPU
is required. The execution path is the configuration behind the published
CPU accuracy numbers (greedy decode, temperature 0.0, max 512 tokens, no
repetition penalty); accuracy is matched against the Mac reference on
paired test sets, though transcripts are not word-identical across
backends.

## Transcribe files

```sh
docker run --rm \
  -v /path/to/audio:/audio \
  ghcr.io/fermionresearch/phonon-cpu:latest \
  transcribe /audio/recording.wav
```

`--model phonon-1` (the default), `--model phonon-1-big` and
`--model phonon-1-micro` select the published models. Without `--model-dir`
the model's release archive is downloaded from Hugging Face, verified
against its published SHA-256 pin, and unpacked; with
`-v /path/to/model:/model … --model-dir /model` a local copy runs fully
offline.

Input envelope: English, 16 kHz audio (mono or stereo). Longer recordings
are segmented and stitched exactly as the other runtimes do; anything
outside the envelope is refused with an actionable message.

## Serve (OpenAI-compatible)

```sh
docker run --rm -p 127.0.0.1:8000:8000 \
  ghcr.io/fermionresearch/phonon-cpu:latest \
  serve --host 0.0.0.0 --port 8000 --api-key change-me --model phonon-1
```

```sh
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F file=@recording.wav -F model=phonon-1
```

The endpoints are the same as the GPU image's: `POST
/v1/audio/transcriptions` (the Whisper API multipart shape, so `openai`
client libraries work unmodified with
`base_url=http://127.0.0.1:8000/v1`), `GET /v1/audio/stream` (RFC 6455
WebSocket, the same protocol as the Mac server), and `GET /health`. The
same API-key, bounded-queue (503 + `Retry-After`) and SIGTERM-drain
behaviour applies, so clients written against either image work unmodified
against both. The `model` form field names the served model (`phonon` is
accepted on any server). Binding a non-loopback host requires `--api-key`
(or `PHONON_API_KEY`); the server refuses to start otherwise.

## Threads

The runtime picks its own thread counts (up to sixteen cores, container
CPU quotas respected); `PHONON_CPU_THREADS` and `PHONON_TORCH_THREADS`
override them when needed.

## What is in the image

The CPU runtime adapters, the CLI/server entrypoint, stdlib multipart and
WebSocket modules, and the compiled CPU kernel binary
(`/opt/phonon/bin/libphonon_cpu-linux-x86_64.so`, binary only). No kernel
source ships in this image. Model weights are not baked in — mount them or
let the entrypoint download them.
