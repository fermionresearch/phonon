# HTTP API reference (speech)

`fermion serve --model phonon` starts an OpenAI-compatible speech server.
It is standard library only (no FastAPI, no uvicorn), binds
`127.0.0.1:8000` by default, and mounts exactly three routes:

```
POST /v1/audio/transcriptions   one-shot file transcription
GET  /v1/audio/stream           live transcription over WebSocket
GET  /v1/models                 (also /v1/models/{id})
GET  /health                    (also /v1/health, /healthz)
```

`GET /` returns a small service descriptor listing the mounted endpoints, so
a client can discover what exists in one request. The chat/completions
endpoints are not mounted on a speech server; requests to them get a 404
whose message names the endpoint that does exist here. (Conversely, an LLM
server answers the audio routes with a 404 pointing at
`fermion serve --model phonon`.)

```bash
fermion serve --model phonon
fermion serve --model phonon --port 8080 --api-key "$(openssl rand -hex 24)"
```

Startup notes print to stderr: the base URL, the served profile and decode
backend, and whether an API key is required.

---

## POST /v1/audio/transcriptions

The Whisper API shape: a `multipart/form-data` body. OpenAI client libraries
work unmodified against `base_url=http://127.0.0.1:8000/v1`.

### Form fields

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | The audio. Anything libsndfile decodes: wav, flac, ogg, aiff (any rate/channels; resampled to 16 kHz mono). mp3/m4a are rejected with a message that includes the ffmpeg conversion line. |
| `model` | no | Accepted and checked, never silently ignored: if it does not name the model this process serves (repo id, alias, or `--served-model-name`), the request gets a 404 `model_not_found` rather than a transcript from a model it did not ask for. |
| `response_format` | no | `json` (default), `text`, or `verbose_json`. Anything else is a 400. |

The OpenAI fields `language`, `prompt`, `temperature` and
`timestamp_granularities` are accepted but not implemented; each is noted
once per process on the server's stderr (the model is English-only,
transcription is deterministic, and word/segment timestamps are not
returned).

Request bodies are capped at 32 MB, and chunked transfer encoding is not
supported. Long recordings should be split; the model is designed for
utterances up to 30 s and decodes long audio in chunks.

### Responses

`response_format=json` (default):

```json
{"text": "The transcript."}
```

`response_format=text`: the bare transcript as `text/plain`, with a trailing
newline.

`response_format=verbose_json`:

```json
{
  "text": "The transcript.",
  "task": "transcribe",
  "language": "english",
  "duration": 4.2,
  "segments": [],
  "x_fermion": {
    "model": "FermionResearch/Phonon-1",
    "profile": "audio6",
    "decode_seconds": 0.31,
    "kind": "speech",
    "backend": "audio6",
    "model_dir": "…",
    "audio_tower_bits": "…",
    "audio_tower_group_size": "…",
    "fold_bits": "…",
    "tiered_head_bits": null,
    "load_seconds": "…",
    "sample_rate": 16000
  }
}
```

`duration` is the audio length in seconds. `segments` is always empty
(segment timestamps are not produced). `x_fermion` is not in OpenAI's
schema and is included deliberately: it names the exact decode configuration
that produced the words, so a result can never be attributed to a
configuration that did not produce it. Treat its exact keys as informational
rather than stable API.

### Errors

Errors use the OpenAI envelope on every route:

```json
{"error": {"message": "…", "type": "invalid_request_error",
           "param": "file", "code": "missing_file"}}
```

| Status | When |
|---|---|
| 400 | No `file` part, malformed multipart, undecodable audio (`code: invalid_audio`), unsupported `response_format`, chunked body. |
| 401 | `--api-key` set and the bearer token is missing or wrong (`code: invalid_api_key`). |
| 404 | Unknown route, or `model` names something this process does not serve. |
| 413 | Body over 32 MB. |

### curl examples

```bash
# Default JSON
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@clip.wav

# Bare text, with an API key
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_KEY" \
  -F file=@clip.wav -F model=phonon-1 -F response_format=text

# Verbose JSON
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@clip.flac -F response_format=verbose_json
```

OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="YOUR_KEY")
out = client.audio.transcriptions.create(
    model="phonon-1", file=open("clip.wav", "rb"))
print(out.text)
```

---

## GET /health

Always unauthenticated (it is matched before the API-key check, so
monitoring works without credentials). A speech server answers:

```json
{
  "status": "ok",
  "model": "FermionResearch/Phonon-1",
  "kind": "speech",
  "version": "0.1.17",
  "repo": "FermionResearch/Phonon-1",
  "profile": "audio6",
  "sha256": "…",
  "sha256_short": "214c3b45",
  "download_bytes": 415077202,
  "model_dir": "…",
  "decode": { "…": "the exact decode configuration in force" },
  "endpoints": ["/v1/audio/transcriptions", "/v1/audio/stream",
                "/v1/models", "/health"],
  "tool_calling": false,
  "stats": {"requests": 12, "streams": 2, "audio_seconds": 51.4,
            "decode_seconds": 2.2, "errors": 0, "realtime_factor": 23.4}
}
```

`sha256` is the pinned archive digest the download was verified against.
`stats` counts what this process has done since startup.

## GET /v1/models

The standard OpenAI list shape with one entry, the served model id
(`--served-model-name` overrides it).

---

## Authentication and the loopback rule

- The default bind is `127.0.0.1`: only this machine can connect, and no key
  is required.
- `--api-key KEY` requires `Authorization: Bearer KEY` on every `/v1/*`
  request (compared in constant time). `/health` stays open.
- Binding a non-loopback host (`--host 0.0.0.0`) prints a warning at
  startup; without `--api-key` that exposes an unauthenticated model server
  beyond this machine. Set a key before you widen the bind, or better, keep
  the loopback bind and put a reverse proxy in front (below).

## One request at a time

The model is held by a single long-lived worker thread that owns the Metal
command queue, and every decode is serialised behind one lock. Concurrent
`POST /v1/audio/transcriptions` requests are not rejected: the second
request queues and is served when the first finishes. Audio decoding
(reading the upload) happens on the request thread, so a malformed upload is
rejected without ever occupying the model. The WebSocket endpoint is
stricter: only one live stream may be open at a time (below). If you need to
keep p99 latency flat under concurrent load, run one `fermion serve` process
per port and balance across them.

---

## GET /v1/audio/stream (WebSocket)

Live streaming transcription. Added in 0.1.18. The endpoint drives the exact
live session behind `fermion listen` (same segmentation, same partial
cadence, same deterministic decode), so the wire endpoint and the local
command cannot drift apart.

It is a standard RFC 6455 WebSocket (version 13). A plain GET without an
upgrade handshake gets a 426 explaining what to send. Mounted only when the
served model is a speech model.

### Authentication

With `--api-key` set, a stream must present the key either as the usual
`Authorization: Bearer KEY` header or as a query parameter,
`ws://127.0.0.1:8000/v1/audio/stream?api_key=KEY`. The query form exists
because the browser `WebSocket()` constructor cannot set headers.

### Client to server

1. First message: one JSON text frame with the stream configuration.
   Both fields are optional:

   ```json
   {"sample_rate": 16000, "format": "pcm_f32le"}
   ```

   `format` is `pcm_f32le` (default) or `pcm_s16le`. `sample_rate` must be
   16000; anything else is refused with an instruction to resample
   client-side.
2. Then: binary frames of raw mono PCM in the declared format. Frame
   boundaries need not align to sample boundaries; the server buffers
   sub-sample tails.
3. To finish: a text frame `{"type":"end"}`, or simply a clean close. Both
   finalize whatever audio is buffered.

Fragmented WebSocket messages (FIN=0/continuation frames) are not supported;
send each message in a single frame. Pings are answered with pongs.

### Server to client

All server messages are JSON text frames:

| Message | Meaning |
|---|---|
| `{"type": "partial", "text": "…"}` | The current whole hypothesis for the in-flight phrase. Each partial replaces the previous one; it is not a delta. The first arrives after ~0.35 s of speech, then roughly one per 0.5 s of audio. |
| `{"type": "final", "text": "…", "segment": N}` | A segment finalized (after ~0.7 s of silence, or at the 30 s segment cap). `N` counts from 1. |
| `{"type": "done", "text": "…"}` | Sent after `end` or close: `text` is the full transcript (finals joined with single spaces, the same shape `transcribe` gives one file). Followed by a clean close (code 1000). |
| `{"type": "error", "message": "…"}` | Any refusal, followed by a close. |

### Limits and close codes

- **One stream at a time.** A second concurrent stream is refused
  immediately with `{"type":"error","message":"engine busy — one stream at
  a time"}` and close code 1013 (try again later); it is never queued behind
  a session that may run for minutes.
- **Idle timeout.** A stream that sends nothing for 90 s is closed with an
  error and code 1001.
- Protocol violations close with 1002, internal errors with 1011, other
  refusals with 1008.
- The first stream after server start pays a one-time warm-up decode (Metal
  graph compile) before partials begin.

`/health` counts completed streams and their audio seconds in its `stats`
block.

---

## Reverse proxy and TLS

The server speaks plain HTTP and is designed to sit behind a reverse proxy
when exposed beyond localhost: keep `fermion serve` on its loopback default
and let the proxy own TLS, timeouts and access control. Keep `--api-key` set
as well; TLS protects the transport, the key protects the endpoint.

nginx:

```nginx
server {
    listen 443 ssl;
    server_name stt.example.com;
    ssl_certificate     /etc/ssl/certs/stt.pem;
    ssl_certificate_key /etc/ssl/private/stt.key;

    client_max_body_size 32m;          # match the server's body cap

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 120s;       # long clips take time to decode
        proxy_request_buffering on;
    }
}
```

Caddy:

```caddy
stt.example.com {
    reverse_proxy 127.0.0.1:8000
    request_body {
        max_size 32MB
    }
}
```

---

## The CUDA container speaks this API too

The NVIDIA Docker image (`ghcr.io/fermionresearch/phonon-cuda`, see
[docker/README.md](../docker/README.md)) serves this same API from image
`0.2.0`, which is rolling out now: `POST /v1/audio/transcriptions` with the
same multipart shape and error envelope, and `GET /v1/audio/stream` with the
**identical WebSocket protocol** — same config frame, same
`partial`/`final`/`done`/`error` events and cadence, same
`Authorization: Bearer` / `?api_key=` auth, same one-stream-at-a-time
refusal (close code 1013) per GPU worker. A client written against
`fermion serve` works unmodified against the container.

Container-specific serve semantics (from `0.2.0`): concurrent HTTP requests
wait in a bounded queue (`--max-queue`, default 8) and get **503 +
`Retry-After`** when it is full — never a silent hang; `GET /health`
additionally reports live queue depth and uptime; SIGTERM drains in-flight
requests before exit. One GPU = one worker — scale horizontally behind a
load balancer with the same reverse-proxy/TLS shape documented above.
