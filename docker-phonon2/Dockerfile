# Phonon-2 on NVIDIA — `ghcr.io/fermionresearch/phonon-cuda` (same server and endpoints as the pip package).
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update -q && apt-get install -y -q --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir \
        "transformers==5.17.0" \
        "numpy<3" \
        soundfile \
        huggingface_hub \
        zstandard

RUN useradd --create-home --uid 1001 phonon \
    && printf '#!/bin/sh\nexec python /opt/phonon/entrypoint.py "$@"\n' > /usr/local/bin/phonon-cuda \
    && chmod 0755 /usr/local/bin/phonon-cuda

COPY phonon2_cuda_engine.py fermion_container.py _archive.py _multipart.py _ws.py _live.py entrypoint.py /opt/phonon/

ENV HF_HOME=/home/phonon/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PHONON2_CUDA_PAD_S=5 \
    PHONON2_CUDA_K=16 \
    PHONON2_CUDA_DTYPE=bfloat16

USER phonon
WORKDIR /home/phonon

ENTRYPOINT ["phonon-cuda"]
CMD ["--help"]
