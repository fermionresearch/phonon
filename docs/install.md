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
fermion models              # shows what is published and what is installed
fermion transcribe clip.wav # first run downloads the default model (415 MB)
```

## Linux, Windows and Intel Macs

The speech verbs refuse cleanly, in one line, before downloading anything:

```
`fermion transcribe` is unavailable: Phonon runs on Apple silicon (arm64
macOS) only — this is Linux/x86_64. The speech model decodes through MLX,
which is Apple-silicon-only; there is no CUDA or x86 build. The language
models (`fermion chat`, `fermion generate`, `fermion serve`) run here
normally.
```

Everything Neutrino (`fermion chat`, `fermion generate`, `fermion serve`
with a language model, `fermion models`) works on these platforms as normal.
For running Phonon on NVIDIA GPUs there is a separate experimental preview;
see [docs/cuda.md](cuda.md).

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
fermion transcribe --download-only unused.wav   # prints the model directory

# copy the unpacked tree to the offline machine, preserving the layout:
#   ~/.cache/fermion/speech/FermionResearch__Phonon-1/model_v18_mlx_head8audio6_quint5/
```

The cache layout the CLI reads is:

```
<cache root>/speech/<Org__Repo>/<unpack_dir>/
```

- `<cache root>` is `~/.cache/fermion` by default, or `$FERMION_CACHE_DIR`
  if set.
- `<Org__Repo>` is the repo id with `/` replaced by `__`, for example
  `FermionResearch__Phonon-1`.
- `<unpack_dir>` is the model's historical directory name:
  `model_v18_mlx_head8audio6_quint5` for Phonon-1,
  `model_v18_mlx_quint5` and `model_v18_mlx_hybrid4_quint5` for the other
  two published models (`fermion models --json` prints each model's exact
  expected path on your machine).
- A directory is treated as installed when `config.json` and
  `packed_manifest.json` exist side by side inside it.

These directory names are kept deliberately interchangeable: a tree produced
by the CLI, by the reference unpacker (`package_release_bps.py unpack …`,
published in each model repo), or by the app's developer tree is the same
tree, so you can also download the `.tar.zst` archive from the model repo by
any means, unpack it with the reference script, and place the result at the
path above.

You do not have to use the cache at all: every speech verb accepts a local
directory directly.

```bash
fermion transcribe --model /srv/models/model_v18_mlx_head8audio6_quint5 clip.wav
```

Set `HF_HUB_OFFLINE=1` on the offline machine if anything in the environment
still tries to reach the Hugging Face Hub.

## Disk and memory expectations

| Model | Download | Unpacked on disk | Peak during install |
|---|---|---|---|
| `FermionResearch/Phonon-1` (default) | 415 MB | 455 MB | ~870 MB |
| `FermionResearch/Phonon-1-Micro` | 285 MB | 331 MB | ~616 MB |
| `FermionResearch/Phonon-1-Big` | 581 MB | 822 MB | ~1.4 GB |

Peak is archive plus unpacked tree on the same volume; the CLI checks free
space before starting a download and refuses with the directory named if it
cannot fit. The downloaded archive stays in the Hugging Face cache after
unpacking; delete it from there (or clear the repo with
`huggingface-cli delete-cache`) to reclaim the download size once the model
is installed. Active memory while transcribing is roughly 0.5 to 0.9 GB
depending on the model; the model cards carry the measured figures.

Cache location knobs (both honoured by every download):

```bash
export FERMION_CACHE_DIR=/big/disk/fermion   # this CLI's model downloads only
export HF_HOME=/big/disk/hf                  # the whole Hugging Face cache
```
