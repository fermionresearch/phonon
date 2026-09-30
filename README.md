# Phonon

Phonon is the speech recognition engine from Fermion Research. It runs the Phonon models on Apple silicon through MLX,
on CPUs under Linux (x86-64 and Arm), Windows and macOS, and on NVIDIA GPUs through the CUDA image. It transcribes files
from the command line, transcribes the microphone live, and serves an OpenAI-compatible endpoint. Phonon-2, a 164 MB
download, is the default model, and Phonon-1 (415 MB), Phonon-1 Big (581 MB) and Phonon-1 Micro (285 MB) run by name.

## Models

| Model | Download | Weights |
|---|--:|---|
| **Phonon-2** (default, `--model phonon-2`) | 164 MB | [huggingface.co/FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) |
| Phonon-1 (`--model phonon-1`) | 415 MB | [huggingface.co/FermionResearch/Phonon-1](https://huggingface.co/FermionResearch/Phonon-1) |
| Phonon-1 Micro (`--model phonon-1-micro`) | 285 MB | [huggingface.co/FermionResearch/Phonon-1-Micro](https://huggingface.co/FermionResearch/Phonon-1-Micro) |
| Phonon-1 Big (`--model phonon-1-big`) | 581 MB | [huggingface.co/FermionResearch/Phonon-1-Big](https://huggingface.co/FermionResearch/Phonon-1-Big) |

Accuracy and speed for each model are on its model page ([fermionresearch.com/models/phonon-2](https://fermionresearch.com/models/phonon-2/) and the model cards above).

## Install

```bash
pip install fermion-research
```

**Apple silicon (MLX engine)**

```bash
pip install mlx mlx-audio mlx-lm soundfile scipy zstandard
```

**Linux and Windows CPUs.** On Linux, install torch from its CPU wheel index first, which skips the GPU build.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only
pip install fermion-research torch safetensors soundfile scipy zstandard
```

**CPU container (amd64 and arm64)**

```bash
docker run --rm -v "$PWD":/audio ghcr.io/fermionresearch/phonon-cpu:2.0.2 transcribe /audio/recording.wav --model phonon-2
```

**NVIDIA CUDA container**

```bash
docker run --rm --gpus all -v "$PWD":/audio ghcr.io/fermionresearch/phonon-cuda:1.0.3 transcribe /audio/recording.wav --model phonon-2
```

## Run

```bash
fermion transcribe recording.wav   # transcribe a file (Phonon-2)
fermion listen                     # live microphone transcription (Apple silicon)
fermion serve                      # OpenAI-compatible HTTP server on 127.0.0.1:8000
```

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=phonon-2"
```

`fermion models` lists every model with its aliases and marks the ones already on the machine.

## Documentation

- [docs/cli.md](docs/cli.md), the command line.
- [docs/server.md](docs/server.md), the HTTP server and its API.
- [docs/install.md](docs/install.md), installation on every platform.
- [docs/cpu.md](docs/cpu.md), running on CPUs, no GPU required.
- [docs/cuda.md](docs/cuda.md), the NVIDIA CUDA image.
- [docs/troubleshooting.md](docs/troubleshooting.md), fixes for common problems.

The CUDA image is built from [docker-phonon2/](docker-phonon2/) and the CPU image from [docker-cpu-phonon2/](docker-cpu-phonon2/).

## Licence

The Phonon-2 weights are released under CC-BY-4.0. They are a derivative of NVIDIA's parakeet-tdt-0.6b-v3, with the changes
listed in the [NOTICE file of the weights repository](https://huggingface.co/FermionResearch/Phonon-2/blob/main/NOTICE). The
Phonon-1 family weights and the command line are released under Apache-2.0, and Phonon-1 is built on
[Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), also under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
