# Phonon-1 on NVIDIA — experimental CUDA preview image.
#
# Default execution path: dense-from-fold4 (stock Torch matmuls on BF16
# weights reconstructed at load from the published packed artifact) — the
# transcript-gated configuration. PHONON_CUDA_PACKED=1 opts into the packed
# CUDA kernel (binary at /opt/phonon/bin), which is EXPERIMENTAL and not
# transcript-gated.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir \
        transformers==4.57.6 \
        safetensors \
        soundfile \
        numpy \
        huggingface_hub \
    && pip install --no-cache-dir qwen-asr==0.0.6 --no-deps

RUN useradd --create-home --uid 1001 phonon \
    && printf '#!/bin/sh\nexec python /opt/phonon/entrypoint.py "$@"\n' \
        > /usr/local/bin/phonon-cuda \
    && chmod 0755 /usr/local/bin/phonon-cuda

COPY phonon_cuda_model.py phonon_cuda_artifact.py phonon_cuda_runtime.py \
     _multipart.py _ws.py _live.py entrypoint.py /opt/phonon/
COPY bin/libphonon_fold4_cuda.so /opt/phonon/bin/libphonon_fold4_cuda.so

ENV PHONON_CUDA_LIBRARY=/opt/phonon/bin/libphonon_fold4_cuda.so \
    HF_HOME=/home/phonon/.cache/huggingface \
    PYTHONUNBUFFERED=1

USER phonon
WORKDIR /home/phonon

ENTRYPOINT ["phonon-cuda"]
CMD ["--help"]
