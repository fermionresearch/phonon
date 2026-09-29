# Running on CPUs

The CPU engine runs Phonon-2 and the Phonon-1 family, on Linux (x86-64 and Arm), Windows and
macOS with no GPU. Supported machines are x86-64 with SSE4.1 or newer and 64-bit Arm with NEON under Linux, x86-64
Windows, Apple silicon Macs, and Intel Macs (Phonon-2). The engine reads the processor's features before it loads a kernel and picks the
tier they allow: on x86-64 AVX2, AVX-512 VNNI and AMX processors run faster tiers; on Arm, dotprod and i8mm processors
do. Processors without those instructions (Raspberry Pi 3 and 4, Cortex-A53/A72 boards, x86-64 parts before AVX2) run
the baseline tier, which gives the same transcripts and is slower. `fermion describe` prints the features found and the
tier chosen. The models are the same published archives the other
engines use (164 MB, 415 MB, 581 MB and 285 MB downloads), decoded with the same configuration as the other engines (greedy
decode, temperature 0.0). Every command names its model, on every platform. Name the model. Phonon never guesses.

Input envelope: English, 16 kHz audio (mono or stereo). Longer recordings
are segmented and stitched exactly as the other runtimes do. Anything
outside the envelope is refused with an actionable message.

## Transcribe with the `fermion` CLI

```sh
pip install fermion-research
fermion transcribe recording.wav                       # Phonon-1
fermion transcribe recording.wav --model phonon-1-big
fermion transcribe recording.wav --model phonon-1-micro
```

On a machine that still needs the CPU speech runtime, `fermion transcribe`
prints the exact install line for that platform and exits — it never fails
with a traceback and never downloads a model it cannot run.

On Windows, the whole install is:

```sh
pip install fermion-research torch safetensors soundfile scipy zstandard
```

The plain torch wheel is already the CPU build there; a clean Windows
machine may also need Microsoft's `vc_redist.x64.exe` (the fix when
`import torch` fails with WinError 126). On Linux, installing torch from
its CPU wheel index (`pip install torch --index-url
https://download.pytorch.org/whl/cpu`) skips the much larger GPU build.

## Threads

The runtime picks its own thread counts: six performance cores on Apple
silicon, up to sixteen cores elsewhere. There is nothing to configure.

## Run the container

```sh
docker run --rm \
  -v /path/to/audio:/audio \
  ghcr.io/fermionresearch/phonon-cpu:latest \
  transcribe /audio/recording.wav
```

`serve` exposes the same OpenAI-compatible endpoints as the GPU image
(`POST /v1/audio/transcriptions`, `GET /v1/audio/stream`, `GET /health`),
with the same API-key and queue behaviour, so clients written against
either work unmodified against both. Without `--model-dir` the model is
downloaded from Hugging Face; `-v /path/to/model:/model … --model-dir
/model` runs fully offline. On Windows, run the container with Docker
Desktop; no GPU is required for the CPU image. The container sources live in
[docker-cpu/](../docker-cpu/).

## Verify an install

`verify_install.py` checks every downloaded file against its published
SHA-256 pin and decodes a short set of clips, so a broken or tampered
install is caught before it ever transcribes your audio.
