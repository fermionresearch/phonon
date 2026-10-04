# Installation

## Standard install

```bash
pip install fermion-research
```

Python 3.10 or newer. The distribution name is `fermion-research` (bare
`fermion` was already taken on PyPI); the command and the import package are
both `fermion`. This installs the CLI and the language-model runtime on
every platform (its dependencies are `torch`, `transformers`, `numpy`,
`huggingface_hub`).

## The speech runtime (Apple silicon)

Speech needs a small MLX stack that is deliberately not a dependency of the
package (the wheels are Apple-silicon-only, and Linux and Intel installs must
not be forced to resolve them). On an Apple-silicon Mac, run:

```bash
pip install mlx mlx-audio mlx-lm soundfile scipy zstandard
```

What each package is for:

| Package | Role |
|---|---|
| `mlx` | Apple's array framework; the model decodes through MLX's quantized matmuls on the Apple GPU (Metal). |
| `mlx-audio` | The audio model implementation the engine drives; also pulls in `sounddevice`, which `fermion listen` uses for microphone capture. |
| `mlx-lm` | Sampling utilities the decode path imports. |
| `soundfile` | Audio file reading (libsndfile: wav, flac, ogg, aiff). |
| `scipy` | Resampling arbitrary input rates to the model's 16 kHz. |
| `zstandard` | Decompresses the zstd model archive at install time. macOS ships no `zstd` tool, so without this wheel a clean Mac could download an archive it cannot unpack. |

Then:

```bash
fermion models                       # shows what is published and what is installed
fermion transcribe phonon-2 clip.wav # first run downloads Phonon-2 (164 MB)
```

## Linux, Windows and Intel Macs

On Linux (x86-64 and 64-bit ARM), x86-64 Windows and Intel Macs, the speech verbs
run on the CPU:

```bash
pip install fermion-research torch safetensors soundfile scipy zstandard
fermion transcribe phonon-2 recording.wav
```

On Linux, install torch from its CPU wheel index first (`pip install --no-deps
torch --index-url https://download.pytorch.org/whl/cpu`, then the install line
above, which adds torch's dependencies from PyPI) to skip the much larger GPU build. On Windows the plain torch wheel already is the CPU
build; a clean machine may also need Microsoft's `vc_redist.x64.exe` (the
fix when `import torch` fails with WinError 126). If anything is missing,
the command prints the exact install line for this platform and exits.
The CPU engine runs on x86-64 processors with SSE4.1 or newer and on 64-bit Arm processors with NEON; AVX2,
AVX-512 VNNI and AMX on x86-64, and dotprod and i8mm on Arm, run faster tiers of the same kernels, and processors
without them (Raspberry Pi 3 and 4, x86-64 parts before AVX2) run a slower baseline tier with the same transcripts.
`fermion describe` shows the features found and the tier chosen. Container details and CPU usage are in
[docs/cpu.md](cpu.md); both Docker images also run under Docker Desktop on Windows, and the CPU image needs no GPU.

Intel Macs: CPU engine. Phonon-2 runs natively on Intel Macs through the CPU engine;
the install line above is the whole setup (it resolves torch 2.2.2 and transformers 5.0 there), and `fermion transcribe phonon-2`, `--json` word timestamps
and `fermion serve phonon-2` work as on Linux. Python 3.10 through 3.12 (the last Intel torch wheels,
which the package resolves on that platform). The Phonon-1 family is not available on Intel Macs; the
command says so in one line.

On Windows ARM, the speech verbs refuse cleanly, in one
line, before downloading anything. Everything Neutrino (`fermion chat`,
`fermion generate`, `fermion serve` with a language model,
`fermion models`) works on every platform as normal. For running Phonon on
NVIDIA GPUs, see [docs/cuda.md](cuda.md).

Python 3.10 through 3.14 are supported on x86-64 and Arm Linux (the `python:3.10-slim` to
`python:3.14-slim` images all install and transcribe). The CPU engine reads audio through
libsndfile; on Alpine or any musl-based image the command refuses before downloading anything:

```
audio files cannot be read: soundfile could not load libsndfile. Install the system library and retry:
    apk add libsndfile   (Alpine / musl: the manylinux wheel bundles libsndfile, the musl wheel does not)
    [OSError: cannot load library 'libsndfile.so': Error loading shared library libsndfile.so: No such file or directory]
```

On glibc systems without the library the same message names `sudo apt install libsndfile1`
(or the distribution's equivalent).

## Air-gapped / offline install

Two things must be moved to the offline machine: the Python wheels and the
model files.

### 1. Wheels

On a connected machine with the same OS, architecture and Python version:

```bash
pip download fermion-research mlx mlx-audio mlx-lm soundfile scipy zstandard \
    -d wheelhouse/
```

On the offline machine:

```bash
pip install --no-index --find-links wheelhouse/ \
    fermion-research mlx mlx-audio mlx-lm soundfile scipy zstandard
```

### 2. Model files: pre-seed the cache

The simplest route is to fetch and verify on a connected Mac, then copy the
cache directory:

```bash
# connected machine (the audio argument is not read with --download-only)
fermion transcribe phonon-2 --download-only unused.wav   # prints the model directory

# copy the unpacked tree to the offline machine, preserving the layout:
#   ~/.cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6/
```

The cache layout the CLI reads is:

```
<cache root>/speech/<Org__Repo>/<unpack_dir>/
```

- `<cache root>` is `~/.cache/fermion` by default, or `$FERMION_CACHE_DIR`
  if set.
- `<Org__Repo>` is the repo id with `/` replaced by `__`, for example
  `FermionResearch__Phonon-2`.
- `<unpack_dir>` is the model's directory name:
  `model_phonon2_c4c_int6` for Phonon-2, `model_v18_mlx_head8audio6_quint5`
  for Phonon-1, `model_v18_mlx_quint5` and `model_v18_mlx_hybrid4_quint5` for
  Phonon-1 Big and Phonon-1 Micro (`fermion models --json` prints each model's
  exact expected path on your machine).
- A directory is treated as installed when `config.json` and
  `packed_manifest.json` exist side by side inside it.

These directory names are kept deliberately interchangeable: a tree produced
by the CLI or by the reference unpacker (`package_release_bps.py unpack …`,
published in each model repo) is the same tree, so you can also download the `.tar.zst` archive from the model repo by
any means, unpack it with the reference script, and place the result at the
path above.

You do not have to use the cache at all: every speech verb accepts a local
directory directly.

```bash
fermion transcribe /srv/models/model_v18_mlx_head8audio6_quint5 clip.wav
```

Set `HF_HUB_OFFLINE=1` on the offline machine if anything in the environment
still tries to reach the Hugging Face Hub.

## Disk and memory expectations

| Model | Download | Unpacked on disk | Peak during install |
|---|---|---|---|
| `FermionResearch/Phonon-2` | 164 MB | 178 MB | ~342 MB |
| `FermionResearch/Phonon-1` | 415 MB | 455 MB | ~870 MB |
| `FermionResearch/Phonon-1-Micro` | 285 MB | 331 MB | ~616 MB |
| `FermionResearch/Phonon-1-Big` | 581 MB | 822 MB | ~1.4 GB |

Peak is archive plus unpacked tree on the same volume; the CLI checks free
space before starting a download and refuses with the directory named if it
cannot fit. The downloaded archive stays in the Hugging Face cache after
unpacking; delete it from there (or clear the repo with
`huggingface-cli delete-cache`) to reclaim the download size once the model
is installed. Active memory while transcribing is roughly 0.5 to 0.9 GB
depending on the model; the model cards carry the detail.

Cache location knobs (both honoured by every download):

```bash
export FERMION_CACHE_DIR=/big/disk/fermion   # this CLI's model downloads only
export HF_HOME=/big/disk/hf                  # the whole Hugging Face cache
```
