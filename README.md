# Phonon-1

Phonon-1 is an open speech recognition model for English. It downloads in
415 MB, runs on a laptop or a datacenter GPU, and transcribes an hour of audio
in about two and a half minutes. It was trained at 2.4 bits per weight from
the start, and it is the second model in the lab's low-bit lane after
Neutrino-1.

Models: [Phonon-1](https://huggingface.co/FermionResearch/Phonon-1) ·
[Phonon-1-Micro](https://huggingface.co/FermionResearch/Phonon-1-Micro) ·
[Phonon-1-Big](https://huggingface.co/FermionResearch/Phonon-1-Big)

## Benchmarks

| Dataset | Phonon-1 (415 MB) | Phonon-1 Micro (285 MB) | Parakeet-0.6B 4-bit (637 MB) | Moonshine base (248 MB) | Whisper large-v3-turbo (1,619 MB) | Whisper small (967 MB) | wav2vec2-large (1,262 MB) | Qwen3-ASR teacher (1,569 MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LibriSpeech test-clean | 2.640 | 3.002 | 2.186 | 3.417 | 2.10 | 3.4† | 2.8† | 2.235 |
| LibriSpeech test-other | 5.699 | 6.511 | 3.937 | 8.262 | 4.07 | 7.6† | 6.3† | 4.618 |
| TED-LIUM | 3.421 | 3.878 | 2.829 | 5.272 | — | — | — | 2.889 |
| SPGISpeech | 4.163 | 4.858 | 4.104 | 5.731 | 2.79† | — | 13.31† | 3.074 |
| VoxPopuli | 8.394 | 9.177 | 6.345 | 10.470 | 11.22† | — | — | 7.151 |
| GigaSpeech | 11.396 | 11.882 | 9.614 | 12.114 | 8.52† | — | — | 9.321 |
| Earnings-22 | 12.571 | 14.771 | 11.190 | 17.872 | 11.07† | — | 36.28† | 11.188 |
| AMI | 13.084 | 14.094 | 12.723 | 17.790 | 15.16† | — | — | 12.560 |
| Macro (eight benchmarks) | 7.67 | 8.52 | 6.62 | 10.1 | — | — | — | 6.63 |

Word error rate, lower is better. Unmarked cells: measured by us — full test sets, Whisper English text normalizer, greedy decoding. † = published figure (model card, paper, or the Open ASR Leaderboard). Dash = no comparable measurement.

Median 23.9× realtime across nine corpora on a base M5 MacBook Air.

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
fermion transcribe recording.wav   # transcribe a file
fermion listen                     # live microphone transcription
fermion serve                      # OpenAI-compatible HTTP server
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

**Apache License 2.0** for the weights and the [command line](https://pypi.org/project/fermion-research/). See [LICENSE](LICENSE) and [NOTICE](NOTICE). Base model: [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), Apache-2.0.
