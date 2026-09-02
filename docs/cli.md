# CLI reference

The `fermion` command ships in the `fermion-research` pip package
(version 0.1.23 at the time of writing). One CLI runs two model families:
**Phonon** (speech recognition, Apple silicon and x86-64 CPUs) and **Neutrino** (language
models, all platforms). This page covers the speech commands in full and the
Neutrino commands in brief.

```
fermion transcribe   one-shot file transcription (Apple silicon, x86-64 CPU)
fermion listen       live microphone transcription (Apple silicon)
fermion serve        OpenAI-compatible HTTP server (speech or LLM)
fermion models       list published models and what is installed
fermion chat         Neutrino REPL
fermion generate     Neutrino one-shot completion
```

`fermion --help` and `fermion <command> --help` are the authoritative flag
listings for your installed version. On a machine without a GPU the same
speech commands run on the CPU; [docs/cpu.md](cpu.md) carries the platform
detail, speed and memory figures.

---

## fermion transcribe

One-shot speech-to-text: an audio file in, a line of text out.

```bash
fermion transcribe meeting.wav
```

```
usage: fermion transcribe [-h] [--model MODEL] [--json] [--verbose]
                          [--download-only]
                          audio
```

| Argument | Meaning |
|---|---|
| `audio` | Path to an audio file. Anything libsndfile reads: wav, flac, ogg, aiff. Any sample rate and channel count (resampled to 16 kHz mono internally). mp3/m4a are not read; convert first: `ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav`. |
| `--model MODEL` | Speech model repo id, alias, or a local unpacked model directory. Default: `FermionResearch/Phonon-1`. See [Model selection](#model-selection). |
| `--json` | Emit a JSON object instead of bare text: text, timings, per-segment timestamps, `truncated` flag. |
| `--verbose` | Print the decode configuration and timings to stderr (decode-only and wall-clock, separately, plus the segment count). |
| `--download-only` | Fetch and verify the model, print its local directory, then stop without decoding. |

### stdout/stdin discipline

stdout carries **only the transcript** (or, with `--json`, only the JSON
object; with `--download-only`, only the model directory path). Every note,
progress bar, warning and timing goes to stderr. So this writes exactly the
transcript and nothing else:

```bash
fermion transcribe clip.wav > out.txt
```

Audio is read from a file path, not from stdin. There is no `-` argument.

### `--json` output shape

```json
{"text": "...", "model": "FermionResearch/Phonon-1", "profile": "audio6",
 "backend": "audio6", "engine": "mlx",
 "duration_seconds": 4.2, "decode_seconds": 0.31, "wall_seconds": 2.4,
 "segment_count": 1,
 "segments": [{"id": 0, "start": 0.0, "end": 4.2, "text": "..."}],
 "truncated": false}
```

`decode_seconds` is the decode alone; `wall_seconds` is the whole command
from model resolution to output, including the model load. Audio up to 35 s
is one segment. Longer files are decoded in 25-35 s windows cut at pauses,
one `segments` entry each (start and end in seconds), and the window
transcripts are joined with single spaces in `text`. `truncated` is true if
any window used its whole token budget, which means part of that window's
audio may be missing from the transcript.

### Determinism

Transcription always decodes at `temperature 0.0` with a fixed repetition
penalty (1.05 over a 96-token context window, the configuration the published
numbers were measured at). There is no `--temperature` flag and no sampler flags, because no
supported configuration uses them. The same file produces the same transcript
on the same machine and version.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (including `--download-only`). |
| 1 | Refusal or failure: unsupported platform, missing speech runtime, unknown model, unreadable audio, download or checksum failure, disk full. Always one plain message on stderr, never a traceback. |
| 2 | Usage error (argparse: unknown flag, missing argument). |

---

## fermion listen

Live streaming microphone transcription. Speak; the current hypothesis
updates on one terminal line; finalized segments print permanently; Ctrl-C
stops and prints the full transcript. Added in 0.1.17.

```bash
fermion listen
fermion listen > note.txt      # captures exactly the words spoken
fermion listen --wav clip.wav  # the same live path, from a file
```

```
usage: fermion listen [-h] [--wav FILE] [--model MODEL] [--verbose]
```

| Flag | Meaning |
|---|---|
| `--wav FILE` | Stream this audio file through the identical live code path, paced to real time, instead of capturing the microphone. The whole streaming stack (segmentation, partial cadence, final decode, rendering) runs headless; only microphone capture is skipped. |
| `--model MODEL` | Same semantics as `transcribe --model`, byte for byte. |
| `--verbose` | Print the decode configuration and one timed stderr line per partial/final instead of the animated live display. |

### Output discipline

Identical to `transcribe`: stdout carries **only the final transcript**,
printed once when the session ends. All live rendering happens on stderr.
When stderr is a tty (and `--verbose` is not set) the current partial
overwrites one line and finals print permanently; when stderr is redirected,
finals print as plain lines (partials only under `--verbose`).

### Partial cadence and segmentation

- The first partial hypothesis appears after about 0.35 s of speech, then a
  new one after every further ~0.5 s of audio. Each partial replaces the
  previous one; it is the whole current hypothesis, not a delta.
- A segment finalizes after about 0.7 s of trailing silence, or at a hard
  30 s cap. The trailing quiet is included in the segment, so the final
  decode sees a contiguous copy of what came in.
- Silence detection is an adaptive energy gate: the louder of an absolute
  room-tone floor and a fraction of the segment's own peak.
- Every partial and final runs the exact `transcribe` decode
  (temperature 0.0). On a single-utterance file, `fermion listen --wav f.wav`
  prints a transcript byte-identical to `fermion transcribe f.wav`.
- Before listening starts, one throwaway decode is run to pay the Metal graph
  compile up front, so the first partial lands on cadence rather than
  stalling. `--verbose` prints how long that warm-up took.

### Stopping

Ctrl-C stops capture, finalizes whatever audio is still buffered, and prints
the full transcript (all finals joined with single spaces) to stdout. A
second Ctrl-C during that last decode skips it and keeps only the segments
already finalized. `listen` never exits with a traceback on Ctrl-C.

### Microphone permission on macOS

The microphone is opened via `sounddevice` (installed transitively by
`mlx-audio`). If the device cannot be opened, `listen` exits with one plain
message. On macOS the usual cause is that your terminal application has no
microphone permission: grant it under
**System Settings, Privacy & Security, Microphone**, then retry.
`fermion listen --wav file.wav` runs the same live path without a microphone.

---

## fermion serve

`fermion serve` starts an HTTP server on `127.0.0.1:8000` by default. **The
model you pass determines which endpoints are mounted.**

### Speech mode

```bash
fermion serve --model phonon
```

With a speech model, the server mounts:

- `POST /v1/audio/transcriptions` (the Whisper API multipart shape)
- `GET /v1/audio/stream` (live transcription over WebSocket; `fermion
  listen`'s session on the wire)
- `GET /v1/models`
- `GET /health`

The chat/completions endpoints are **not** mounted; a POST to them returns a
404 naming the endpoint that does exist. Transcription over HTTP is
deterministic, exactly like the CLI. Concurrent file requests are serialised
behind a single decode worker (one Metal command queue); a second request
waits, it is not rejected. The WebSocket endpoint allows one live stream at
a time.

Flags that matter in speech mode: `--model`, `--host` (default `127.0.0.1`),
`--port` (default `8000`), `--api-key` (require this bearer token on `/v1/*`
requests), `--cors`, `--served-model-name`. The LLM sampler and backend flags
(`--temperature`, `--draft`, `--kv-dtype`, `--backend`, `--session-ctx`,
`--tool-profile`, `--max-new-ceiling`, `--yarn-factor`) are accepted by the
shared parser but have no meaning for a speech model; if you set one, the
server says so once on stderr at startup and ignores it.

Full API documentation, including request and response shapes, auth, and
reverse-proxy guidance: [docs/server.md](server.md).

### LLM mode

Started without `--model` (or with a Neutrino model or a local TRTC
container), `serve` is an OpenAI-compatible language-model server:
`POST /v1/chat/completions` (streaming and non-streaming, with tool calling),
`POST /v1/completions`, `GET /v1/models`, `GET /health`. It defaults to the
published sampler config (temp 0.01, top-p 1.0, rep-pen 1.05, window
256) and adds serve-side scaffolds for agent workloads: `--stuck-detector`
(default `enforce`), `--tool-profile`, `--max-new-ceiling`, `--session-ctx`,
`--no-session`, plus the decode flags shared with `chat`/`generate`
(`--device`, `--dtype`, `--backend`, `--native-bin`, `--kv-dtype`,
`--yarn-factor`, `--yarn-orig-max`, `--draft`, `--max-new`, `--temperature`,
`--top-p`, `--rep-penalty`, `--pen-window`). See `fermion serve --help` and
the package README for the full story.

---

## fermion models

Lists every model the lab publishes, marks the ones already on this machine
(`*`), and reports whether the speech runtime is available here. It reads no
network: everything comes from the built-in catalog and a local directory
test.

```bash
fermion models
fermion models --json   # machine-readable
fermion models --all    # also show profiles retained but never published
```

---

## Model selection

Every speech verb takes `--model`, which accepts a repo id, a short alias, or
a local directory holding an unpacked model. There is deliberately no
`--profile` flag: each model is its own repository, so the model is the
profile.

| Model (repo id) | Aliases | Profile | Download | On disk |
|---|---|---|---|---|
| `FermionResearch/Phonon-1` (default) | `phonon`, `phonon-1`, `speech`, `stt`, `asr` | `audio6` | 415 MB | 455 MB |
| `FermionResearch/Phonon-1-Big` | `phonon-1-big`, `phonon-big`, `big` | `parity` | 581 MB | 822 MB |
| `FermionResearch/Phonon-1-Micro` | `phonon-1-micro`, `phonon-micro`, `micro` | `micro` | 285 MB | 331 MB |

- Aliases and repo ids are case-insensitive
  (`--model fermionresearch/phonon-1` works).
- The bare family aliases (`phonon`, `speech`, `stt`, `asr`) resolve to the
  default model, so `fermion transcribe clip.wav` with no `--model` does the
  expected thing.
- A local directory is accepted anywhere a repo id is:
  `--model /path/to/model_v18_mlx_head8audio6_quint5`. The directory must
  hold `config.json` and `packed_manifest.json` side by side.

### What a fresh machine downloads, and where it lands

On first use of a model, the CLI:

1. Fetches the repo's small metadata files first (`config.json`, `README.md`,
   `LICENSE`, `NOTICE`, `verify_install.py`, `package_release_bps.py`) and
   cross-checks the published facts against its own pinned SHA-256 before
   spending the transfer. A disagreement refuses the download.
2. Downloads the model archive (a `.tar.zst`), verifies the whole file
   against the pinned SHA-256, and unpacks it with a per-file SHA-256 check
   on every member. A corrupt download fails at install time, not at
   inference time.
3. Caches the unpacked model under
   `~/.cache/fermion/speech/<Org__Repo>/<unpack_dir>/`, for example
   `~/.cache/fermion/speech/FermionResearch__Phonon-1/model_v18_mlx_head8audio6_quint5/`.
   The downloaded archive itself sits in the Hugging Face hub cache
   (`~/.cache/huggingface/hub` by default).

Later runs load from the cache with no network access.
`fermion transcribe --download-only clip-not-needed` fetches and verifies
without decoding, and prints the model directory.

Two environment variables move the caches:

- `FERMION_CACHE_DIR=/big/disk/fermion` relocates this CLI's model downloads
  only (speech models unpack under `$FERMION_CACHE_DIR/speech/…`).
- `HF_HOME=/big/disk/hf` relocates the whole Hugging Face cache, including
  the downloaded archives and metadata.

Disk preflight is automatic: a download that cannot fit on the cache volume
is refused before it starts, with the directory named and both fixes printed.

---

## Neutrino commands in brief

The language-model side of the CLI is documented in the
[`fermion-research` package README](https://github.com/fermionresearch) and
in each command's `--help`.

- `fermion chat` starts an interactive REPL against
  `fermionresearch/Neutrino-8B` by default (aliases: `neutrino`, `8b`;
  smaller SKUs: `neutrino-0.6b`, `0.6b-chat`). Sampled at the published default
  config.
- `fermion generate "prompt"` is the one-shot, scriptable surface:
  deterministic greedy by default, so scripts reproduce byte for byte.
- `fermion info`, `fermion verify`, `fermion bench` and `fermion inspect`
  verify and measure containers; see their `--help`.

Passing a speech model to a language verb (or the reverse) is caught before
any download and answered with the right verb to use.
