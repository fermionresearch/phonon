#!/bin/bash
# Build the Phonon-2 CUDA image (this directory) and tag it; push with `--push`. `:latest` is moved by hand.
set -euo pipefail; cd "$(dirname "$0")"
IMG=ghcr.io/fermionresearch/phonon-cuda; VER=1.0.6
docker build -t $IMG:$VER .
docker image inspect $IMG:$VER --format 'built {{.Id}} size {{.Size}}'
if [ "${1:-}" != "--push" ]; then echo "built and tagged $IMG:$VER; NOT pushed (pass --push)"; exit 0; fi
docker push $IMG:$VER && echo "pushed $IMG:$VER"
