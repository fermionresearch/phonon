#!/bin/bash
# Build the Phonon-2 CUDA image (this directory), tag 1.0.0 + latest, and push to ghcr.io/fermionresearch/phonon-cuda.
# Refuses to push without FOUNDER_WORD=push.
set -euo pipefail; cd "$(dirname "$0")"
IMG=ghcr.io/fermionresearch/phonon-cuda; VER=1.0.0
docker build -t $IMG:$VER -t $IMG:latest .
docker image inspect $IMG:$VER --format 'built {{.Id}} size {{.Size}}'
if [ "${FOUNDER_WORD:-}" != "push" ]; then echo "built and tagged; NOT pushed (FOUNDER_WORD=push to push)"; exit 0; fi
echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:-fermionresearch}" --password-stdin
docker push $IMG:$VER && docker push $IMG:latest && echo "pushed $IMG:$VER + latest"
