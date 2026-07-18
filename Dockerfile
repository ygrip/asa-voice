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

# --- base: deps + OS packages shared by every runtime profile (hosted/local/hybrid) ---------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg/ffprobe: audio_service duration probing runs on every STT upload regardless of provider.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt .
RUN pip install --upgrade pip && pip install -r requirements-base.txt

# --- hosted: OpenAI STT+TTS only. torch/pocket-tts/faster-whisper/ctranslate2/scipy stay absent
# (scripts/assert_hosted_deps_absent.py asserts this in CI) - the lean production default. -----------
FROM base AS hosted

COPY app ./app
# Cue pack embedded at build time (setara-nx07.4). build/cues/.gitkeep keeps this dir present even
# when no pack has been generated yet - CueService treats an empty dir as "no embedded pack".
COPY build/cues ./app/generated/cues
RUN mkdir -p /tmp/asa-voice /models

EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--ws-ping-interval", "0"]

# --- local: local Whisper STT + local Pocket TTS, CPU-only torch. ----------------------------------
FROM base AS local

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-local-stt.txt requirements-local-tts.txt .
# Prefer the CPU-only torch wheel (avoids the ~2GB CUDA wheel on x86_64); fall back to the default
# index where the CPU index has no wheel (e.g. linux/arm64). pocket-tts then reuses this torch.
RUN (pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5.0" \
        || pip install "torch>=2.5.0") \
    && pip install -r requirements-local-stt.txt -r requirements-local-tts.txt

COPY app ./app
COPY build/cues ./app/generated/cues
RUN mkdir -p /tmp/asa-voice /models

EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--ws-ping-interval", "0"]

# --- hybrid: identical dependency set to `local` - provider routing (plan §11) is controlled
# entirely by STT_PROVIDER/TTS_PROVIDER/*_FALLBACK_PROVIDER env vars, not the image. -----------------
FROM local AS hybrid
