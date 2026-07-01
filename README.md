<h1 align="center">asa-voice</h1>

<p align="center">Self-hosted voice sidecar for ASA — faster-whisper STT + Pocket TTS, packaged for Docker</p>

<p align="center">
  <a href="https://github.com/ygrip/asa-voice/pkgs/container/asa-voice"><img src="https://img.shields.io/badge/GHCR-asa--voice-blue?logo=docker" alt="Docker" /></a>
  <img src="https://github.com/ygrip/asa-voice/actions/workflows/ci.yml/badge.svg" alt="CI" />
</p>

---

Free, self-hosted, lightweight voice service for ASA: **STT** via faster-whisper, **TTS** via Pocket TTS.
FastAPI + Uvicorn. Optional — Setara works without it (text-only).

Setara's Quarkus backend (`setara-core`) is the only caller; the sidecar is **internal-only** in
production (no public port). Browser → core `/api/asa/voice/*` → sidecar.

---

## Run locally

```bash
cp .env.example .env
docker compose up --build
# or, without Docker:
pip install -r requirements.txt
uvicorn app.main:app --port 8090
```

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | `{status, sttLoaded, ttsLoaded}` |
| `GET` | `/models` | limits + STT info + TTS voice catalog |
| `POST` | `/stt` | multipart `file` (wav/webm/mp3/m4a) → transcript JSON |
| `WS` | `/stt/stream` | WebSocket rolling-window streaming STT |
| `POST` | `/tts` | JSON `{text, voiceId, format}` → `audio/wav` |
| `POST` | `/tts/stream` | streaming PCM chunks via chunked HTTP |

### Quick test

```bash
curl http://localhost:8090/health
curl -X POST http://localhost:8090/stt -F "file=@sample.wav"
curl -X POST http://localhost:8090/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Build created and assigned to the plan.","voiceId":"asa_default"}' \
  --output asa-output.wav
```

---

## Resource guards

- 1 concurrent STT + 1 concurrent TTS (extra requests get `429`).
- Max audio 20 s, max upload 15 MB (`413` otherwise).
- Models preloaded at startup (`/health` unhealthy until loaded).

---

## Engines

- **STT**: faster-whisper `distil-small.en` int8 (CTranslate2). Downgrade to `base.en` if memory-tight.
- **TTS**: Kyutai **pocket-tts** (100 M params, CPU, PyTorch 2.5+), used via its in-process Python API
  (`TTSModel.load_model()` once at startup, voice states cached, `generate_audio()` per request, converted
  to 16-bit PCM wav). Isolated behind `app/services/tts_service.py` — the `/tts` contract is engine-agnostic,
  so swapping engines (Piper fallback) touches only that file.
  Voice catalog maps our IDs → Kyutai voices: `asa_default` → anna, `asa_bright` → eve, `asa_calm` → george.

### First start downloads models

On first boot, faster-whisper and pocket-tts fetch their models from HuggingFace into `/root/.cache`
(persisted via the `asa_voice_cache` volume). Needs network on first run and can take a minute or two
(`/health` returns `sttLoaded: false` / `ttsLoaded: false` until complete; healthcheck `start_period` is 180 s).

### Footprint

torch is installed **CPU-only** (`--index-url .../whl/cpu`) to avoid the ~2 GB CUDA wheel.
pocket-tts uses ~2 CPU cores. If RAM is tight against the 3 GB cap, set `STT_MODEL=base.en`.

---

## Docker image

Published to GHCR on every `v*` tag push:

```bash
docker pull ghcr.io/ygrip/asa-voice:latest
```

Multi-arch: `linux/amd64` and `linux/arm64`.

---

## Configuration

All env keys are in `.env.example` and map 1:1 to `app/config.py` fields.

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `distil-small.en` | faster-whisper model. `base.en` (lightest) → `distil-small.en` → `small.en` (most accurate) |
| `STT_DEVICE` | `cpu` | Inference device |
| `STT_COMPUTE_TYPE` | `int8` | CTranslate2 quantization |
| `STT_CPU_THREADS` | `4` | Decode thread count — use all cores for best latency |
| `STT_VAD_FILTER` | `true` | Skip silent segments |
| `STT_STREAM_INTERVAL_MS` | `600` | Rolling-window re-decode interval for `/stt/stream` |
| `TTS_ENGINE` | `pocket-tts` | TTS backend (only `pocket-tts` currently) |
| `TTS_DEFAULT_VOICE` | `asa_default` | Voice ID: `asa_default` (anna), `asa_bright` (eve), `asa_calm` (george) |
| `TTS_DEFAULT_MODEL` | `pocket-low` | Pocket TTS model variant |
| `TTS_SAMPLE_RATE` | `24000` | Output sample rate (Hz) |
| `TTS_COMPILE` | `true` | `torch.compile` the model at startup — first call is slow, subsequent calls faster |
| `TTS_STREAM_COALESCE` | `1` | Chunks to buffer before flushing in `/tts/stream`. `1` = lowest latency |
| `MAX_AUDIO_SECONDS` | `20` | Max STT input duration before `413` |
| `MAX_UPLOAD_MB` | `15` | Max upload body size before `413` |
| `MAX_CONCURRENT_STT` | `1` | Max parallel STT requests |
| `MAX_CONCURRENT_TTS` | `1` | Max parallel TTS requests |
| `TMP_DIR` | `/tmp/asa-voice` | Scratch dir for audio processing |

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
