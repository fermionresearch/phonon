# Troubleshooting

Speech commands are designed to fail with one plain message and exit code 1,
never a traceback. This page lists the messages you may see and what to do.

## "Phonon runs on Apple silicon (arm64 macOS) only"

Full message (from `fermion transcribe` on an unsupported machine):

```
`fermion transcribe` is unavailable: Phonon runs on Apple silicon (arm64
macOS) only — this is <OS>/<arch>. The speech model decodes through MLX,
which is Apple-silicon-only; there is no CUDA or x86 build. The language
models (`fermion chat`, `fermion generate`, `fermion serve`) run here
normally.
```

There is nothing to fix on this machine: the speech model decodes through
MLX, which only exists on Apple silicon. The Neutrino commands still work.
For NVIDIA GPUs there is a separate experimental preview; see
[docs/cuda.md](cuda.md).

## "the speech runtime needs mlx, …, which are not installed"

```
speech is unavailable: the speech runtime needs <names>, which are not
installed. These are Apple-silicon-only wheels, so they are not dependencies
of `fermion-research` itself — Linux and Intel users must not be forced to
resolve them. Install them with:
    pip install mlx mlx-audio mlx-lm soundfile scipy zstandard
```

Run the printed line in the same Python environment the `fermion` command
lives in (check with `which fermion` and `pip -V`). The probe checks for
`mlx`, `mlx_audio`, `mlx_lm`, `soundfile` and `scipy`.

## "live microphone capture needs sounddevice"

```
`fermion listen` is unavailable: live microphone capture needs sounddevice,
which is not installed. The speech install line provides it (via mlx-audio):
    pip install mlx mlx-audio mlx-lm soundfile scipy zstandard
```

Only `fermion listen` needs `sounddevice`; `transcribe` and `serve` do not
probe for it. It normally arrives transitively with `mlx-audio`.

## "could not open the microphone" (macOS permission)

```
could not open the microphone (<error>). On macOS, grant this terminal
microphone access in System Settings → Privacy & Security → Microphone and
retry. `fermion listen --wav file.wav` runs the same live path from a file.
```

Open **System Settings, Privacy & Security, Microphone** and enable the
terminal application you run `fermion` from (Terminal, iTerm2, VS Code and
so on), then start `fermion listen` again. macOS prompts once per
application; if the prompt was dismissed, the toggle stays off until you set
it by hand. To verify everything above the microphone works, run
`fermion listen --wav clip.wav`, which exercises the identical live path
from a file.

## Server: "engine busy" and queued requests

Two different behaviours under concurrency:

- `POST /v1/audio/transcriptions`: requests queue. `fermion serve` holds the
  model on a single worker thread and serialises decodes behind one lock, so
  a request that arrives while another is decoding is not rejected and gets
  no 429/503; it waits its turn and then runs. Clients that appear to hang
  under load are in that queue, not hitting a fault.
- `GET /v1/audio/stream` (WebSocket): one live stream at a time. A second
  concurrent stream is refused immediately with
  `{"type":"error","message":"engine busy — one stream at a time"}` and
  WebSocket close code 1013 (try again later). Retry after the first stream
  closes. An idle stream is closed by the server after 90 s.

`GET /health` shows cumulative requests, streams, audio seconds and decode
seconds in its `stats` block. For parallel throughput, run several
`fermion serve` processes on different ports and balance across them.

## zstd / unpack problems

- ```
  cannot decompress the model archive: none of the `zstandard` Python
  package, Python 3.14's `compression.zstd`, or the `zstd` command-line
  tool is available. Install either one:
      pip install zstandard
      brew install zstd
  ```
  macOS ships no zstd decompressor. Either install fixes it;
  `pip install zstandard` is the documented route.

- `could not read <archive> as a zstd-compressed tar (…)` usually means a
  truncated or corrupted download. Delete the archive from the Hugging Face
  cache and re-run the command; the download resumes fresh and is verified
  again.

- `checksum mismatch unpacking <file> — the archive is corrupt; delete it
  and re-run` means a member failed its per-file SHA-256 during unpack.
  Same fix: delete and re-download. Interrupted unpacks never leave a
  half-installed model (the tree is staged and renamed only on success), so
  re-running is always safe.

- `<filename> sha256 … != the pinned …` before unpacking means the
  downloaded archive does not match the digest pinned in this release.
  Delete it from the Hugging Face cache and retry; if it recurs, do not use
  that download.

- A disagreement between the client's pins and the repo's published
  `config.json` refuses before downloading and suggests
  `pip install -U fermion-research`. The pinned digest in your installed
  version is authoritative; the client never adopts a published value that
  disagrees with it.

## Slow first run

Three one-time costs stack on a fresh machine:

1. The model download and SHA-verified unpack (hundreds of MB; progress
   bars on stderr).
2. The very first `fermion` invocation after installing compiles the
   dependency stack to bytecode, roughly ten seconds once; later
   invocations start fast.
3. The first decode after a model loads pays the Metal graph compile,
   roughly four to five seconds. `fermion listen` pays this up front as an
   explicit warm-up decode so the first live partial is not delayed;
   `transcribe` pays it inside the first file. Subsequent decodes in the
   same process run at full speed.

A long blank pause with no output is not one of these: downloads always show
progress, and every stage prints to stderr.

## Disk space

Downloads are preflighted: if the cache volume cannot hold the archive plus
the unpacked model, nothing is downloaded and the message names the
directory and both fixes:

```
export HF_HOME=/big/disk/hf                  # the standard Hugging Face knob
export FERMION_CACHE_DIR=/big/disk/fermion   # this CLI's model downloads only
```

The cache lives on your home volume by default, wherever a bigger disk is
mounted.

## Verifying artifact integrity

Verification is automatic on every install: the archive is checked against
the SHA-256 pinned inside the CLI before unpacking, and every unpacked file
is checked against the manifest's per-file SHA-256 before it is written. A
model that installed is a model that verified.

To re-check independently:

- `fermion transcribe --download-only unused.wav` prints the model directory
  (and re-verifies if anything needs fetching; the audio argument is not
  read).
- Each model repo publishes `verify_install.py`, which re-hashes every shard
  of an unpacked tree against `packed_manifest.json` without loading or
  mutating it. Place it next to the unpacked model directory and run, for
  example:

  ```bash
  python verify_install.py --profile audio6
  ```

  It prints one `PASS` line per shard and a summary, and exits non-zero on
  any size or SHA-256 mismatch.
- The pinned digests also appear on `GET /health` (`sha256`) and in each
  repo's `config.json`, so all three sources can be compared by hand.

## mp3 / m4a input

`soundfile` reads what libsndfile reads: wav, flac, ogg, aiff. Compressed
mp3/m4a is refused with the conversion line:

```
ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav
```

## Where to file issues

GitHub: <https://github.com/fermionresearch> (the repository's issue
tracker). Include the full stderr of the failing command, `fermion
--version`, `python -V` and your macOS/chip (`uname -m` should print
`arm64`). For server issues, include the startup banner and the response of
`GET /health`.
