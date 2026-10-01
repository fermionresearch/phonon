# CLI reference

The `fermion` command ships in the `fermion-research` pip package
(version 0.2.5 at the time of writing). The fermion command runs the Phonon speech models and the Neutrino
language models from one install. This page covers the speech commands in full and the
Neutrino commands in brief.

```
fermion transcribe   one-shot file transcription (Apple silicon; Linux, Windows and macOS CPUs)
fermion listen       live microphone transcription (Apple silicon)
fermion serve        OpenAI-compatible HTTP server (speech or LLM)
fermion models       list published models and what is installed
fermion describe     this machine's CPU features and the speech kernel tier it runs
fermion speech bench speed of a speech model on this machine
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
fermion transcribe phonon-2 meeting.wav
```

```
usage: fermion transcribe [-h] [--json] [--verbose] [--threads N]
                          [--download-only] [--hotwords WORDS]
                          [--hotword-strength LOGITS]
                          [MODEL AUDIO ...]

positional arguments:
  MODEL AUDIO           the model (phonon-2, phonon-1, phonon-1-big,
                        phonon-1-micro, a repo id or a local model directory)
                        followed by the audio file (wav/flac/ogg/aiff);
                        `fermion transcribe phonon-2 meeting.wav`
```

| Argument | Meaning |
|---|---|
| `audio` | Path to an audio file. Anything libsndfile reads: wav, flac, ogg, aiff. Any sample rate and channel count (resampled to 16 kHz mono internally). mp3/m4a are not read; convert first: `ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav`. |
| `MODEL` | First positional argument: `phonon-2`, `phonon-1`, `phonon-1-big`, `phonon-1-micro`, a repo id or a local model directory. Required. See [Model selection](#model-selection). |
| `--threads N` | CPU engine threads (default one per physical core from six cores up, every logical cpu on smaller parts, performance cores on Apple silicon); same as `FERMION_CPU_THREADS`. |
| `--json` | Emit a JSON object instead of bare text: text, timings, per-segment timestamps, `truncated` flag. |
| `--verbose` | Print the decode configuration and timings to stderr (decode-only and wall-clock, separately, plus the segment count). |
| `--download-only` | Fetch and verify the model, print its local directory, then stop without decoding. |

### stdout/stdin discipline

stdout carries **only the transcript** (or, with `--json`, only the JSON
object; with `--download-only`, only the model directory path). Every note,
progress bar, warning and timing goes to stderr. So this writes exactly the
transcript and nothing else:

```bash
fermion transcribe phonon-2 clip.wav > out.txt
```

Audio is read from a file path, not from stdin. There is no `-` argument.

### `--json` output shape

```json
{"text": "...", "model": "FermionResearch/Phonon-2", "profile": "five-value",
 "backend": "phonon2-five-value", "engine": "mlx",
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
fermion listen phonon-2
fermion listen phonon-2 > note.txt      # captures exactly the words spoken
fermion listen phonon-2 --wav clip.wav  # the same live path, from a file
```

```
usage: fermion listen [-h] [--wav FILE] [--verbose] [--hotwords WORDS]
                      [--hotword-strength LOGITS]
                      [MODEL]

positional arguments:
  MODEL                 the speech model: phonon-2, phonon-1, phonon-1-big,
                        phonon-1-micro, a repo id or a local model directory;
                        `fermion listen phonon-2`
```

| Flag | Meaning |
|---|---|
| `--wav FILE` | Stream this audio file through the identical live code path, paced to real time, instead of capturing the microphone. The whole streaming stack (segmentation, partial cadence, final decode, rendering) runs headless; only microphone capture is skipped. |
| `MODEL` | Same semantics as the first argument of `transcribe`, byte for byte. |
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
  (temperature 0.0). On a single-utterance file, `fermion listen phonon-2 --wav f.wav`
  prints a transcript byte-identical to `fermion transcribe phonon-2 f.wav`.
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
`fermion listen phonon-2 --wav file.wav` runs the same live path without a microphone.

---

## fermion serve

`fermion serve` starts an HTTP server on `127.0.0.1:8000` by default. **The
model you pass determines which endpoints are mounted.**

### Speech mode

```bash
fermion serve phonon
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

Flags that matter in speech mode: `--host` (default `127.0.0.1`), `--unix-socket PATH` (owner-only socket in place of a port and key, macOS and Linux), `--threads N`,
`--port` (default `8000`), `--api-key` (require this bearer token on `/v1/*`
requests), `--cors`, `--served-model-name`. The LLM sampler and backend flags
(`--temperature`, `--draft`, `--kv-dtype`, `--backend`, `--session-ctx`,
`--tool-profile`, `--max-new-ceiling`, `--yarn-factor`) are accepted by the
shared parser but have no meaning for a speech model; if you set one, the
server says so once on stderr at startup and ignores it.

Full API documentation, including request and response shapes, auth, and
reverse-proxy guidance: [docs/server.md](server.md).

### LLM mode

Started with a Neutrino model (or a local TRTC
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

## fermion describe

What this machine is and which Phonon-2 CPU kernel tier it runs (added in 0.2.5). Nothing is downloaded and no model
is loaded. `phonon describe` and `phonon --describe` print the same report.

```bash
fermion describe
fermion describe --json
```

```
usage: fermion describe [-h] [--json] [--no-load]

options:
  --json      emit one JSON object
  --no-load   do not open the selected kernel library for its own report
```

The report names the host, the processor features the engine found (dotprod, i8mm and SVE on Arm; SSE4.1, AVX2,
AVX-512 BW and VNNI, AMX on x86-64), how they were read, the tier chosen (`i8mm`, `dotprod` or `neon` on Arm; `amx`,
`avx512-vnni`, `avx512bw`, `avx2`, `sse4.1` or `scalar` on x86-64) and the kernel binaries that tier loads. Processors
without dotprod or AVX2 run the baseline tier, which gives the same transcripts and is slower; [docs/cpu.md](cpu.md)
has the supported-CPU list.

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

Every speech verb takes the model as its first argument, a short alias, a repo id, or
a local directory holding an unpacked model. Name the model. Phonon never guesses. There is deliberately no
`--profile` flag: each model is its own repository, so the model is the
profile.

| Model (repo id) | Aliases | Profile | Download | On disk |
|---|---|---|---|---|
| `FermionResearch/Phonon-2` | `phonon-2`, `phonon2`, `phonon`, `speech`, `stt`, `asr` | `five-value` | 164 MB | 178 MB |
| `FermionResearch/Phonon-1` | `phonon-1` | `audio6` | 415 MB | 455 MB |
| `FermionResearch/Phonon-1-Big` | `phonon-1-big`, `phonon-big`, `big` | `parity` | 581 MB | 822 MB |
| `FermionResearch/Phonon-1-Micro` | `phonon-1-micro`, `phonon-micro`, `micro` | `micro` | 285 MB | 331 MB |

- Aliases and repo ids are case-insensitive
  (`fermion transcribe fermionresearch/phonon-1 clip.wav` works).
- Every speech verb takes the model first, for example `fermion transcribe phonon-2 clip.wav`.
  A command without a model prints the model names and exits.
- A local directory is accepted anywhere a repo id is:
  `fermion transcribe /path/to/model_phonon2_c4c_int6 clip.wav`. The directory must
  hold `config.json` and `packed_manifest.json` side by side.

### A command without a model

Nothing is downloaded; the command prints the model list and exits 2:

```
fermion transcribe: name the model. Phonon never guesses.

  fermion transcribe phonon-2 meeting.wav

Models (alias, kind, repo):
  phonon-2, phonon2, phonon          speech   FermionResearch/Phonon-2
  phonon-1                           speech   FermionResearch/Phonon-1
  phonon-1-big, phonon-big, big      speech   FermionResearch/Phonon-1-Big
  phonon-1-micro, phonon-micro, micro speech   FermionResearch/Phonon-1-Micro
  neutrino, neutrino-8b, 8b          language fermionresearch/Neutrino-8B
  neutrino-0.6b, 0.6b, draft         language fermionresearch/Neutrino-0.6B
  neutrino-0.6b-chat, 0.6b-chat      language fermionresearch/Neutrino-0.6B-Chat

`fermion models` lists everything and marks what is already on this machine.
[exit 2]
```

`--model MODEL` still works in this release and prints one note line saying the model now comes first.

### The `phonon`, `phonon-2` and `phonon-1` commands

Each is the `fermion` CLI with its model fixed:

```
phonon: the fermion CLI with the model fixed to phonon-2.

  phonon transcribe meeting.wav
  phonon listen
  phonon serve [--port 8000] [--unix-socket PATH]
  phonon bench --audio clip.wav
  phonon describe

Everything after the verb is passed to `fermion <verb> phonon-2`; `phonon <verb> -h` shows that verb's options.
```

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
   `~/.cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6/`.
   The downloaded archive itself sits in the Hugging Face hub cache
   (`~/.cache/huggingface/hub` by default).

Later runs load from the cache with no network access.
`fermion transcribe phonon-2 --download-only clip-not-needed` fetches and verifies
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
