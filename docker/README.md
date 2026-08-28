# Phonon-1 CUDA Docker image

Runs [Phonon-1 Big](https://huggingface.co/FermionResearch/Phonon-1-Big) on
NVIDIA GPUs. The image's default execution path is **dense-from-fold4**:
standard Torch matmuls over BF16 weights reconstructed at load time from the
published packed artifact. That is the configuration behind the published
CUDA accuracy numbers (greedy decode, temperature 0.0, max 512 tokens, no
repetition penalty) — measured on LibriSpeech at parity with the Apple/MLX
flagship path (same accuracy; transcripts are not word-identical across
backends).

Requires an NVIDIA GPU (Ampere or newer recommended) and the NVIDIA
Container Toolkit.

## Versions

- **`0.2.0`** — rolling out now: long-audio transcription (files over 30 s
  are segmented and stitched), live streaming over WebSocket
  (`/v1/audio/stream`, the same protocol as the Mac server), and bounded
  request queueing for concurrent clients. Each capability below is marked
  with the version it belongs to.
- **`0.1.0-preview`** — available today: single utterances up to 30 s,
  `transcribe` + `serve` with `POST /v1/audio/transcriptions`.

`latest` tracks the newest gated tag.

## Transcribe files

```sh
docker run --rm --gpus all \
  -v /path/to/Phonon-1-Big:/model -v /path/to/audio:/audio \
  ghcr.io/fermionresearch/phonon-cuda:latest \
  transcribe /audio/recording.wav --model-dir /model
```

Without `--model-dir` the model is downloaded from Hugging Face
(`--repo FermionResearch/Phonon-1-Big` is the default).

Input envelope: English, 16 kHz audio (mono or stereo). In `0.1.0-preview`,
single utterances up to 30 seconds; from `0.2.0`, longer recordings are
handled by energy-gated segmentation — the same segmentation constants the
Mac engine uses (a segment closes after ~0.7 s below the adaptive gate, or
at a 30 s cap) — with each segment decoded by the gated configuration and
the finals joined with single spaces. Anything outside the envelope is
refused with an actionable message — no unvalidated fallback paths.

## Serve (OpenAI-compatible)

```sh
docker run --rm --gpus all -p 127.0.0.1:8000:8000 \
  -v /path/to/Phonon-1-Big:/model \
  ghcr.io/fermionresearch/phonon-cuda:latest \
  serve --host 0.0.0.0 --port 8000 --api-key change-me --model-dir /model
```

```sh
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F file=@recording.wav -F model=phonon-1
```

`POST /v1/audio/transcriptions` takes the Whisper API multipart shape
(`file`, optional `model`, `response_format` = json | text | verbose_json),
so `openai` client libraries work unmodified with
`base_url=http://127.0.0.1:8000/v1`. `GET /health` reports the exact decode
configuration in force. Binding a non-loopback host requires `--api-key`
(or `PHONON_API_KEY`); the server refuses to start otherwise.

### Streaming (`0.2.0`)

`GET /v1/audio/stream` is a standard RFC 6455 WebSocket speaking the exact
protocol documented in [docs/server.md](../docs/server.md): send one JSON
config frame, then binary PCM frames; receive `partial` / `final` / `done`
JSON events (first partial after ~0.35 s of speech, then roughly one per
0.5 s of audio; segments finalize after ~0.7 s of silence or at the 30 s
cap). Auth accepts `Authorization: Bearer` or `?api_key=` (the query form
exists because the browser `WebSocket()` constructor cannot set headers).
One live stream per GPU worker; a second concurrent stream is refused
immediately with close code 1013 rather than queued behind a session that
may run for minutes. A client written against the Mac server works
unmodified against the container.

### Concurrency and the queue contract (`0.2.0`)

One GPU worker decodes one request at a time. Concurrent HTTP requests wait
in a bounded queue (`--max-queue`, default 8). When the queue is full the
server answers **503 with a `Retry-After` header** — it never hangs
silently. `GET /health` reports live queue depth, uptime, and the model id,
so a load balancer can steer on it.

## Enterprise deployment

- **One GPU = one worker.** The model is held by a single process that owns
  the GPU; scale horizontally by running one container per GPU behind a load
  balancer, exactly as the Mac server documents. Health-check on
  `GET /health` (unauthenticated by design).
- **TLS via reverse proxy.** The container speaks plain HTTP; put nginx or
  Caddy in front for TLS, timeouts, and access control (config examples in
  [docs/server.md](../docs/server.md)). Keep `--api-key` set as well: TLS
  protects the transport, the key protects the endpoint.
- **API keys.** `--api-key` / `PHONON_API_KEY` requires
  `Authorization: Bearer` on every `/v1/*` route (constant-time compare);
  `/health` stays open for monitoring. Rotate by restarting the container —
  keys are process-lifetime, not persisted.
- **Queue/503 contract.** Steer new traffic away when `/health` queue depth
  approaches `--max-queue`; on 503, honor `Retry-After`.
- **Throughput expectation, measured not promised:** the gated dense path
  measured ~6x realtime at batch 1 on an A100-40GB across the full 5,559-
  utterance LibriSpeech test set (p50 0.96 s per ~7 s utterance). Size
  fleets from your own audio mix.
- **Graceful shutdown (`0.2.0`).** SIGTERM stops accepting new requests,
  drains in-flight work, then exits — safe behind rolling deploys.
- Request logging goes to stderr and never includes audio contents.

## PHONON_CUDA_PACKED=1 — experimental packed kernel

The image ships a compiled packed-arithmetic CUDA kernel
(`/opt/phonon/bin/libphonon_fold4_cuda.so`, binary only; fatbin for
sm_80/86/89/90). Setting `PHONON_CUDA_PACKED=1` uses it for single-token
decode.

**It is experimental and NOT transcript-gated**: it has not passed the
transcript parity gates the default dense path passed, and its output can
differ from the published accuracy numbers. Leave it off for anything that
matters. It exists in the image so the packed path can be evaluated
like-for-like ahead of a gated release.

## What is in the image

The Phonon runtime adapters (`phonon_cuda_model.py`,
`phonon_cuda_artifact.py`, `phonon_cuda_runtime.py`), the CLI/server
entrypoint, stdlib multipart and WebSocket modules, and the compiled kernel
binary. No kernel source ships in this image. Model weights are not baked
in — mount them or let the entrypoint download them.
