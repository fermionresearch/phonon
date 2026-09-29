# Phonon-2 on plain CPU — `ghcr.io/fermionresearch/phonon-cpu` (Phonon-1 0.1.21 docker-cpu pattern: the pip package's CPU engine
# behind the same `fermion transcribe` / `fermion serve` commands; no GPU, no MLX).
#
# Engine: the five-value container expanded to dense fp32 in the stock transformers ParakeetForTDT graph (tier 1, day one) with
# the packed int8 five-value GEMM + the C greedy TDT loop available as FERMION_P2_CPU=packed / FERMION_P2_CPU_TDT=c (Linux x86-64
# AVX2 / AVX-512 VNNI and aarch64 NEON binaries ship inside the wheel tree under fermion/_speech/_engine_phonon2_cpu/).
# bf16 autocast switches itself on where the CPU has AMX-bf16 (Sapphire Rapids+), measured 30x -> 46.6x at the same WER.
FROM python:3.12-slim

RUN apt-get update -q && apt-get install -y -q --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers==5.17.0" "numpy<3" soundfile scipy zstandard safetensors huggingface_hub

# the staged fermion tree (0.2.0 with patches 0006b + 0007 applied, binaries + SHA256SUMS in _engine_phonon2_cpu/)
COPY pip_tree /opt/fermion_src
RUN pip install --no-cache-dir /opt/fermion_src && rm -rf /opt/fermion_src

RUN useradd --create-home --uid 1001 phonon
ENV FERMION_DEVICE=cpu \
    HF_HOME=/home/phonon/.cache/huggingface \
    OMP_WAIT_POLICY=PASSIVE \
    PYTHONUNBUFFERED=1
USER phonon
WORKDIR /home/phonon

# `docker run --rm -v $PWD:/audio ghcr.io/fermionresearch/phonon-cpu transcribe /audio/clip.wav --model phonon-2`
# `docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/fermionresearch/phonon-cpu serve --host 0.0.0.0 --model phonon-2 --api-key ...`
ENTRYPOINT ["fermion"]
CMD ["--help"]
