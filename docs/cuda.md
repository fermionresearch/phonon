# Running on NVIDIA

The CUDA image runs Phonon-2 on NVIDIA GPUs, and the Phonon-1 family has its own earlier image. Input is English,
16 kHz audio, greedy decode, NVIDIA GPU required.

Two images. **Phonon-2** ships as `ghcr.io/fermionresearch/phonon-cuda:1.0.6` (also `:latest`): it downloads the
model from the Hub on first run into the `phonon-cache` volume and serves `phonon-2`; built from [docker-phonon2/](../docker-phonon2/).

```bash
docker run --rm --gpus all -v "$PWD":/audio -v phonon-cache:/home/phonon/.cache ghcr.io/fermionresearch/phonon-cuda:1.0.6 transcribe phonon-2 /audio/recording.wav
docker run --rm --gpus all -p 127.0.0.1:8000:8000 -v phonon-cache:/home/phonon/.cache ghcr.io/fermionresearch/phonon-cuda:1.0.6 serve phonon-2 --host 0.0.0.0 --port 8000 --api-key YOUR_KEY
```

The image takes hotwords like the pip package: `transcribe ... --hotwords "Ada, Quillon"`, and a `hotwords` or `prompt`
field on the server ([hotwords.md](hotwords.md)).

The **Phonon-1 family** image is `ghcr.io/fermionresearch/phonon-cuda:0.3.0` (built from [docker/](../docker/)); it also
comes as a bare single-utterance script. All three Phonon-1 models run in the container: `--model
phonon-1-big` (that image's own default), `--model phonon-1`, `--model phonon-1-micro`.

## Bare script

Full documentation: [cuda/README.md](../cuda/README.md). In short: install
`cuda/requirements-cuda.txt` plus `qwen-asr==0.0.6 --no-deps`, download the
model it targets (the largest published build; see that README), and run:

```bash
python transcribe_cuda.py audio.wav --model-dir /path/to/Phonon-1-Big
```

It expands the published packed artifact to dense weights at load time and
decodes end to end on the GPU, with no CPU fallback. The script runs the
largest build; the container below runs all three.

## Docker image

Full documentation: [docker/README.md](../docker/README.md). The image
(`ghcr.io/fermionresearch/phonon-cuda:0.3.0`) needs the NVIDIA
Container Toolkit and offers two subcommands. Transcribe a file:

```bash
docker run --rm --gpus all \
  -v /path/to/Phonon-1-Big:/model -v /path/to/audio:/audio \
  ghcr.io/fermionresearch/phonon-cuda:0.3.0 \
  transcribe /audio/utterance.wav --model-dir /model
```

`--model phonon-1` and `--model phonon-1-micro` select the other published
models.

Or serve the same OpenAI-compatible `/v1/audio/transcriptions` endpoint
documented in [docs/server.md](server.md) (`serve --host 0.0.0.0 --port 8000
--api-key …`; non-loopback binds refuse to start without a key). Without
`--model-dir` the requested model's release archive is downloaded from
Hugging Face, verified against its published SHA-256 pin, and unpacked.
`PHONON_CUDA_PACKED=1` switches single-token decode to an optional packed
kernel path; the default is the dense path.

The container also transcribes long recordings (energy-gated segmentation
with the pip package's constants, finals joined with single spaces), serves
the `/v1/audio/stream` WebSocket with the same protocol as `fermion serve`
(one client works against both), and queues concurrent HTTP requests behind
a bounded queue (503 + `Retry-After` when full, never a silent hang).
Deployment notes (one GPU = one worker, TLS via reverse proxy, key
management) are in [docker/README.md](../docker/README.md).
