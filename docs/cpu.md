# Running on CPUs

The CPU engine runs Phonon-2 by default, and the Phonon-1 family by name, on Linux (x86-64 and Arm), Windows and
macOS with no GPU. Supported machines are x86-64 (AVX2, which is any mainstream CPU from roughly 2014 on) and 64-bit
Arm (NEON) under Linux, x86-64 Windows, and Apple silicon Macs. The models are the same published archives the other
engines use (164 MB, 415 MB, 581 MB and 285 MB downloads), decoded with the same configuration as on a Mac (greedy
decode, temperature 0.0). `--model phonon` is Phonon-2, the default, exactly as on a Mac.

Accuracy is matched against the Mac reference on paired test sets, but
transcripts are not word-identical across backends (floating-point
reduction order differs between Metal and CPU BLAS).

Input envelope: English, 16 kHz audio (mono or stereo). Longer recordings
are segmented and stitched exactly as the other runtimes do. Anything
outside the envelope is refused with an actionable message.

## Transcribe with the `fermion` CLI

```sh
pip install fermion-research
fermion transcribe recording.wav                       # Phonon-2, the default
fermion transcribe recording.wav --model phonon-1
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
  ghcr.io/fermionresearch/phonon-cpu:2.0.2 \
  transcribe /audio/recording.wav --model phonon-2
```

`serve` exposes the same OpenAI-compatible endpoints as the GPU image
(`POST /v1/audio/transcriptions`, `GET /v1/audio/stream`, `GET /health`),
with the same API-key and queue behaviour, so clients written against
either work unmodified against both. Without `--model-dir` the model is
downloaded from Hugging Face; `-v /path/to/model:/model … --model-dir
/model` runs fully offline. On Windows, run the container with Docker
Desktop; no GPU is required for the CPU image. The image runs on amd64 and arm64; its sources live in
[docker-cpu-phonon2/](../docker-cpu-phonon2/) (the Phonon-1 CPU image is built from [docker-cpu/](../docker-cpu/)).

## Verify an install

`verify_install.py` checks every downloaded file against its published
SHA-256 pin and decodes a short set of clips, so a broken or tampered
install is caught before it ever transcribes your audio.
