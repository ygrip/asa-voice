FROM python:3.11-slim AS model-converter

WORKDIR /work
COPY requirements-convert.txt .
RUN pip install --no-cache-dir -r requirements-convert.txt
COPY scripts/convert_whisper_ct2.sh /usr/local/bin/convert_whisper_ct2.sh
RUN chmod +x /usr/local/bin/convert_whisper_ct2.sh

ARG STT_SOURCE_MODEL
ARG STT_SOURCE_REVISION
ARG STT_QUANTIZATION=int8
ARG STT_OUTPUT_DIR=/models/whisper-ct2-int8
RUN if [ -n "${STT_SOURCE_MODEL}" ] && [ -n "${STT_SOURCE_REVISION}" ]; then \
        convert_whisper_ct2.sh \
            "${STT_SOURCE_MODEL}" \
            "${STT_OUTPUT_DIR}" \
            "${STT_QUANTIZATION}" \
            "${STT_SOURCE_REVISION}"; \
    else \
        mkdir -p /models; \
    fi

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Prefer the CPU-only torch wheel (avoids the ~2GB CUDA wheel on x86_64); fall back to the default
# index where the CPU index has no wheel (e.g. linux/arm64). Then pocket-tts reuses the installed torch.
RUN pip install --upgrade pip \
    && (pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5.0" \
        || pip install "torch>=2.5.0") \
    && pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /tmp/asa-voice /models

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--ws-ping-interval", "0"]
