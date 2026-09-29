# ghcr.io/fermionresearch/phonon-cpu — Phonon-2 on plain CPUs (x86-64 and arm64), built from the published pip release.
FROM python:3.12-slim
ARG FERMION_VERSION=0.2.2
RUN apt-get update -q && apt-get install -y -q --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "fermion-research==${FERMION_VERSION}" "transformers==5.17.0" "numpy<3" soundfile scipy zstandard safetensors huggingface_hub
COPY entrypoint.sh /usr/local/bin/phonon-entrypoint
RUN chmod 755 /usr/local/bin/phonon-entrypoint && useradd --create-home --uid 1001 phonon
ENV FERMION_DEVICE=cpu \
    HF_HOME=/home/phonon/.cache/huggingface \
    OMP_WAIT_POLICY=PASSIVE \
    PYTHONUNBUFFERED=1
USER phonon
WORKDIR /home/phonon
# `docker run --rm -v "$PWD":/audio ghcr.io/fermionresearch/phonon-cpu transcribe /audio/recording.wav --model phonon-2`
# `docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/fermionresearch/phonon-cpu serve --host 0.0.0.0 --model phonon-2 --api-key ...`
ENTRYPOINT ["/usr/local/bin/phonon-entrypoint"]
CMD ["--help"]
