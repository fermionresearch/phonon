# Phonon-1 on plain CPUs (x86-64, AVX2).
#
# CPU-only sibling of the CUDA image: slim Python base instead of the CUDA
# runtime base, torch installed from the CPU wheel index, and the packed CPU
# kernel binary in place of the CUDA one. The execution path in this image
# IS the transcript-gated packed configuration (greedy decode, temperature
# 0.0, max 512 tokens, no repetition penalty) — unlike the CUDA image there
# is no experimental opt-in flag, because the packed CPU path is the gated
# path. All three published models run here (phonon-1, phonon-1-big,
# phonon-1-micro).
FROM python:3.12-slim

# transformers pinned to the transcript-gated version.
RUN pip install --no-cache-dir \
        torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        transformers==4.57.6 \
        safetensors \
        soundfile \
        numpy \
        huggingface_hub \
        zstandard \
    && pip install --no-cache-dir qwen-asr==0.0.6 --no-deps

RUN useradd --create-home --uid 1001 phonon \
    && printf '#!/bin/sh\nexec python /opt/phonon/entrypoint_cpu.py "$@"\n' \
        > /usr/local/bin/phonon-cpu \
    && chmod 0755 /usr/local/bin/phonon-cpu

# Runtime adapters (Python source) + the CLI/server entrypoint.
COPY artifact.py graph.py runtime.py model.py driver.py hybrid.py levers.py \
     engine.py _archive.py _multipart.py _ws.py _live.py entrypoint_cpu.py \
     /opt/phonon/
# The ONLY binary in the image; kernel sources never enter the build context.
COPY bin/libphonon_cpu-linux-x86_64.so /opt/phonon/bin/libphonon_cpu-linux-x86_64.so

# Thread policy is the entrypoint's job (kernel threads = min(nproc, 16),
# decode-phase torch threads = 1, full torch count for encoder/prefill);
# no OMP env pins here so cgroup quota detection stays the runtime's call.
ENV PHONON_CPU_LIBRARY=/opt/phonon/bin/libphonon_cpu-linux-x86_64.so \
    HF_HOME=/home/phonon/.cache/huggingface \
    PYTHONUNBUFFERED=1

USER phonon
WORKDIR /home/phonon

ENTRYPOINT ["phonon-cpu"]
CMD ["--help"]
