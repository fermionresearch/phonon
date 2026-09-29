# Phonon

Open speech recognition models for English from Fermion Research: **Phonon-2** (164 MB,
the current model — see its section below) and the **Phonon-1** family.

Phonon-1 is an open speech recognition model for English. It downloads in
415 MB and runs on a laptop or a datacenter GPU.

Models: [Phonon-1](https://huggingface.co/FermionResearch/Phonon-1) ·
[Phonon-1-Micro](https://huggingface.co/FermionResearch/Phonon-1-Micro) ·
[Phonon-1-Big](https://huggingface.co/FermionResearch/Phonon-1-Big)

## Phonon-2

Phonon-2 is an open speech recognition model for English, a 164 MB download. Accuracy and speed are on its model page
([huggingface.co/FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2)).

```bash
pip install fermion-research
fermion transcribe recording.wav --model phonon-2      # Apple silicon (MLX), Linux, Windows: the CPU engine; NVIDIA GPUs: the CUDA image
docker run --rm -v "$PWD":/audio ghcr.io/fermionresearch/phonon-cpu:2.0.1 transcribe /audio/recording.wav --model phonon-2
docker run --rm --gpus all -v "$PWD":/audio ghcr.io/fermionresearch/phonon-cuda:1.0.2 transcribe /audio/recording.wav --model phonon-2
```

The weights are at [huggingface.co/FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2). Based on
parakeet-tdt-0.6b-v3 by NVIDIA; the tokenizer and output conventions (punctuation, casing, numerals) are the original's.
Licence CC-BY-4.0, same as the original; the weights repository's `NOTICE` lists the changes. The
CUDA image is built from [docker-phonon2/](docker-phonon2/) and the CPU image from [docker-cpu-phonon2/](docker-cpu-phonon2/).

## Install

```bash
pip install fermion-research
```

On an Apple-silicon Mac, add the speech runtime:

```bash
pip install mlx mlx-audio mlx-lm soundfile scipy zstandard
```

## Run it

Phonon runs on Apple silicon through MLX, on NVIDIA GPUs through the
Docker image, and on ordinary CPUs — x86-64 Linux and Windows, and
Apple silicon — through the CPU runtime ([docs/cpu.md](docs/cpu.md)).

```bash
phonon transcribe meeting.wav               # transcribe a file with Phonon-2
phonon transcribe meeting.wav --json        # the same, as JSON with word timestamps
phonon transcribe meeting.wav --hotwords "Ada, Quillon"    # favour names and terms
phonon listen                               # live microphone transcription (Apple silicon)
phonon serve                                # OpenAI-compatible HTTP server on 127.0.0.1:8000
fermion transcribe phonon-2 meeting.wav     # the same, naming the model
```

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=FermionResearch/Phonon-1"
```

```bash
docker run --rm --gpus all -v "$PWD":/audio ghcr.io/fermionresearch/phonon-cuda:latest transcribe /audio/recording.wav
```

The NVIDIA CUDA runtime lives in [cuda/](cuda/).

## Documentation

- [docs/cli.md](docs/cli.md): the command line.
- [docs/server.md](docs/server.md): the HTTP server and its API.
- [docs/install.md](docs/install.md): installation on every platform.
- [docs/troubleshooting.md](docs/troubleshooting.md): fixes for common problems.
- [docs/cuda.md](docs/cuda.md): the NVIDIA CUDA runtime and Docker image.
- [docs/cpu.md](docs/cpu.md): running on CPUs, no GPU required.

## License

**Phonon-2 weights: CC-BY-4.0** (a derivative of NVIDIA's parakeet-tdt-0.6b-v3; the weights repository's `NOTICE` lists the changes). **Phonon-1 family weights and the [command line](https://pypi.org/project/fermion-research/): Apache License 2.0.** See [LICENSE](LICENSE) and [NOTICE](NOTICE). Base model: [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), Apache-2.0.
