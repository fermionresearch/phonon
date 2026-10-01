# Phonon-2 on NVIDIA — `ghcr.io/fermionresearch/phonon-cuda` (Phonon-1 0.3.0 pattern; same server, same endpoints).
#
# Default execution path: the five-value container expanded EXACTLY to dense fp16 in the stock transformers
# ParakeetForTDT graph, greedy TDT decode captured in CUDA graphs, log-mel on the GPU, bucket-padded input shapes.
# PHONON2_CUDA_PATH=packed opts into the packed two-plane encoder (302 MB on the GPU instead of 1,208 MB; Triton
# fused dequant+GEMM; same transcripts within argmax ties; slower at batch >= 8).
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update -q && apt-get install -y -q --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir \
        "transformers==5.17.0" \
        "numpy<3" \
        soundfile \
        huggingface_hub \
        zstandard \
        triton

RUN useradd --create-home --uid 1001 phonon \
    && printf '#!/bin/sh\nexec python /opt/phonon/entrypoint.py "$@"\n' > /usr/local/bin/phonon-cuda \
    && chmod 0755 /usr/local/bin/phonon-cuda \
    && mkdir -p /home/phonon/.cache/huggingface && chown -R phonon:phonon /home/phonon
VOLUME ["/home/phonon/.cache"]

COPY phonon2_cuda_engine.py fermion_container.py _archive.py _multipart.py _ws.py _live.py entrypoint.py /opt/phonon/

ENV HF_HOME=/home/phonon/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PHONON2_CUDA_PATH=dense \
    PHONON2_CUDA_PAD_S=5 \
    PHONON2_CUDA_K=16 \
    PHONON2_CUDA_DTYPE=bfloat16

USER phonon
WORKDIR /home/phonon

ENTRYPOINT ["phonon-cuda"]
CMD ["--help"]
