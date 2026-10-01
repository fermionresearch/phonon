#!/bin/bash
# Build the Phonon-2 CPU image (this directory) for the host architecture and tag it; push with `--push`.
# The published image is a linux/amd64 + linux/arm64 manifest list assembled from two native builds.
set -euo pipefail; cd "$(dirname "$0")"
IMG=ghcr.io/fermionresearch/phonon-cpu; VER=2.0.4; ARCH=$(uname -m); case $ARCH in x86_64) DA=amd64;; aarch64|arm64) DA=arm64;; *) echo "unsupported arch $ARCH"; exit 2;; esac
docker build -t $IMG:$VER-$DA .
docker image inspect $IMG:$VER-$DA --format 'built {{.Id}} size {{.Size}}'
if [ "${1:-}" != "--push" ]; then echo "built and tagged $IMG:$VER-$DA; NOT pushed (pass --push)"; exit 0; fi
docker push $IMG:$VER-$DA && echo "pushed $IMG:$VER-$DA"
echo "then, once both architectures are pushed:"
echo "  docker manifest create $IMG:$VER $IMG:$VER-amd64 $IMG:$VER-arm64 && docker manifest push $IMG:$VER"
