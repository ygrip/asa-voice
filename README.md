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
cp .env.example .env  # optional: create local overrides
docker compose up --build
# or, without Docker:
pip install -r requirements.txt
uvicorn app.main:app --port 8090
```

For development and contract tests, install `requirements-dev.txt` or run `make install-dev`, then `make test`.

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | `{status, mode, sttLoaded, ttsLoaded, stt:{provider,...}, tts:{provider,...}}` |
| `GET` | `/models` | mode + limits + STT/TTS active+fallback+available providers |
| `POST` | `/stt` | multipart `file` (wav/webm/mp3/m4a) → transcript JSON |
| `POST` | `/stt/raw` | final PCM16 mono 16 kHz decode without temporary files |
| `WS` | `/stt/stream` | WebSocket rolling-window streaming STT |
| `POST` | `/tts` | JSON `{text, voiceId, format}` → `audio/wav` |
| `POST` | `/tts/stream` | `format=pcm|l16`; streaming PCM16 little-endian chunks via chunked HTTP |

## WebSocket `/stt/stream`

The client sends binary PCM16 mono 16 kHz little-endian frames. Send a configuration message once
after connecting, before audio frames:

```json
{
  "type": "config",
  "language": "en",
  "prompt": "Voice command for ASA on the Raksara build page.",
  "hotwords": ["ASA", "Setara", "Raksara", "build 1.0.1"],
  "requestId": "019voice-request"
}
```

Control messages are `{"type":"flush"}` at the end of an utterance and `{"type":"reset"}` to
discard it. Plain text `flush` remains supported for older clients. Server messages are:

```json
{"type":"partial","text":"open build raksara"}
{"type":"final","text":"open build Raksara 1.0.1"}
{"type":"error","detail":"STT model not loaded"}
```

Partials are display-only. Only a `final` transcript should trigger an ASA command.

For HTTP finalization, `POST /stt/raw` requires `Content-Type: audio/l16`, `X-Sample-Rate: 16000`,
`X-Channels: 1`, and `X-Sample-Format: s16le`. Optional bounded decode context is base64url-encoded
JSON in `X-Stt-Context`.

### Browser PCM capture example

Use an `AudioWorklet` in production so capture does not depend on the main thread. The worklet must
resample microphone audio to 16 kHz and post `Float32Array` frames. Convert and send each frame like
this:

```js
function float32ToPcm16(samples) {
  const pcm = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm.buffer;
}

const socket = new WebSocket("ws://localhost:8090/stt/stream");
socket.addEventListener("open", () => {
  socket.send(JSON.stringify({ type: "config", hotwords: ["ASA", "Setara"] }));
});
worklet.port.onmessage = ({ data }) => {
  if (socket.readyState === WebSocket.OPEN) socket.send(float32ToPcm16(data));
};
vad.onSpeechEnd = () => socket.send(JSON.stringify({ type: "flush" }));
socket.addEventListener("message", ({ data }) => {
  const event = JSON.parse(data);
  if (event.type === "partial") renderGhostTranscript(event.text);
  if (event.type === "final") submitFinalTranscript(event.text);
});
```

`POST /tts` accepts only `format=wav` and returns `audio/wav`. `POST /tts/stream` accepts only
`format=pcm` or `format=l16` and returns `audio/l16` with `X-Sample-Rate`, `X-Channels`, and
`X-Sample-Format: s16le`. Raw PCM is not a WAV file and must be scheduled through Web Audio rather
than passed to `decodeAudioData()`.

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
- Rolling WebSocket and raw PCM buffers are capped at 20 s by default.
- Multipart uploads are capped independently at 300 s and 15 MB (`413` otherwise).
- Models preloaded at startup (`/health` unhealthy until loaded).

---

## Engines

- **STT**: faster-whisper `distil-small.en` int8 (CTranslate2). Downgrade to `base.en` if memory-tight.
- **TTS**: Kyutai **pocket-tts** (100 M params, CPU, PyTorch 2.5+), used via its in-process Python API
  (`TTSModel.load_model()` once at startup, voice states cached, `generate_audio()` per request, converted
  to 16-bit PCM wav). Isolated behind `app/services/tts_service.py` — the `/tts` contract is engine-agnostic,
  so swapping engines (Piper fallback) touches only that file.
  Voice catalog maps our IDs → Kyutai voices: `asa_default` → anna, `asa_bright` → eve, `asa_calm` → george.

### Provider mode (`ASA_VOICE_MODE`)

STT/TTS engines sit behind provider adapters (`app/providers/`) selected through `STT_PROVIDER`/`TTS_PROVIDER` +
`ASA_VOICE_MODE` (`local` | `hosted` | `hybrid`). `faster_whisper` and OpenAI STT plus `pocket_tts` TTS are
implemented; an unrecognized provider name fails the process at startup instead of silently no-op'ing. The mode is
descriptive metadata - provider and fallback variables perform the actual routing. See
`asa-local-openai-hosted-mode-plan.md` (repo root) and this file's `KNOWLEDGE.md` for the adapter/router pattern.

Provider routing currently covers multipart `POST /stt`. The file-free `POST /stt/raw` path needs an adapter with
`transcribe_array()`, and the rolling WebSocket path directly uses the Faster Whisper service. Setara Core uses both
of those paths for its full voice-session flow, so keep `STT_PROVIDER=faster_whisper` for production voice sessions
until provider-neutral raw and streaming contracts are implemented. OpenAI can be evaluated through `POST /stt` or
as a file-based fallback without implying that hosted mode covers the complete Core flow.

### achetronic/parakeet assessment

Parakeet is feasible as a future STT provider, but it is not a Faster Whisper model value. Setting
`STT_MODEL=parakeet-tdt-0.6b` would make CTranslate2 try to load a non-Whisper architecture and fail. The upstream
service is a separate Go/ONNX server exposing `POST /v1/audio/transcriptions`, so the correct boundary is a dedicated
`parakeet` adapter behind `SttProviderRouter`.

The recommended integration sequence is:

1. Add explicit Parakeet URL, API key, model label, and timeout settings, plus a `ParakeetSttAdapter` that normalizes
   the OpenAI-compatible response into `SttResult`.
2. Register `parakeet` as a primary/fallback provider and health-check the upstream server independently. Do not label
   it `openai`, even if the OpenAI SDK is reused as the HTTP client.
3. Support multipart `/stt` first. For `/stt/raw`, encode bounded PCM16 to an in-memory WAV upload or add a provider
   byte-stream method. Preserve the current size, duration, quota, and fallback policy before making it selectable.
4. Treat WebSocket support as separate work. Parakeet's current streaming API uploads the complete file and emits SSE
   text deltas; it is not the bidirectional PCM WebSocket protocol used by `/stt/stream`.
5. Benchmark final command accuracy, entity preservation, peak RSS, latency, and mixed Indonesian-English input before
   promotion. The current NVIDIA v3 model lists 25 European languages and does not list Indonesian.

Operational gates also matter. Upstream documents about 2 GB RAM for the int8 model, a 25 MB request limit, ignored
`prompt`, and whole-file encoding up to 400 seconds unless long-audio mode is enabled. Prompt/hotword behavior therefore
cannot be assumed equivalent to Whisper, and a Parakeet container adds memory beside this sidecar rather than replacing
Pocket TTS. The model is CC BY 4.0. Upstream's README code-license statement and GitHub license metadata should be
reconciled against the exact pinned release before distribution. Sources: [achetronic/parakeet](https://github.com/achetronic/parakeet),
[NVIDIA Parakeet TDT 0.6B v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

### LiteASR assessment

[efeslab/LiteASR](https://github.com/efeslab/LiteASR) is a Whisper encoder-compression method, not a server or a
Faster Whisper checkpoint format. Its published checkpoints use Hugging Face Transformers with custom model code,
PyTorch, and optionally Triton or MLX. They therefore cannot be selected through the current `STT_MODEL` CTranslate2
loader. The safe integration boundary would be a separate `lite_asr` adapter and runtime image, or an independently
deployed HTTP service.

LiteASR is most relevant when Whisper Large v3 or Large v3 Turbo accuracy is required on GPU or Apple MLX hardware.
The reported accurate Large v3 Turbo variant reduces the encoder from 635M to 421M parameters with average WER moving
from 10.1 to 10.2, but the complete model remains about 0.7B parameters and the reference quick start uses CUDA FP16.
The repository pins PyTorch 2.6, Transformers 4.49, and Triton 3.2. This does not match the current CPU-only 2 GB
sidecar target, and LiteASR retains Whisper's fixed-window/live-streaming disadvantages. Smaller compressed checkpoints
exist, but they need independent CTranslate2 compatibility, CPU RSS, latency, multilingual, and entity-accuracy proof.
Repository and published model metadata are Apache 2.0. Source: [LiteASR repository](https://github.com/efeslab/LiteASR),
[Lite Whisper Large v3 Turbo model card](https://huggingface.co/efficient-speech/lite-whisper-large-v3-turbo-acc).

Decision: keep LiteASR as a GPU/MLX batch benchmark lane, not the default CPU sidecar or hands-free engine.

### Moonshine assessment

[moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine) is the strongest architectural candidate for ASA's
live voice path. Unlike Whisper, its streaming models process flexible-duration audio incrementally and cache work while
the speaker is talking. The native API accepts mono float PCM at arbitrary input rates, emits partial/final events, and
also exposes non-streaming array transcription. That maps naturally to both `/stt/raw` and `/stt/stream` behind a
dedicated `moonshine` adapter or a small native sidecar service.

Upstream's CPU benchmark reports English Small Streaming at 123M parameters and 165 ms post-utterance latency on Linux
x86, and Medium Streaming at 245M parameters, 269 ms, and 6.65% WER. These are upstream measurements using its own VAD
and must be reproduced on Setara's hardware and command corpus. English code and models are MIT licensed. Non-English
models use Moonshine's non-commercial Community License, and Indonesian is not in the current STT language list, so
mixed Indonesian-English production use remains an explicit blocker. Source: [Moonshine comparison and models](https://github.com/moonshine-ai/moonshine#when-should-you-choose-moonshine-over-whisper).

The current `setara-ui` intentionally has no browser Moonshine runtime. It captures audio and sends it through Core so
authentication, prompt/hotword preparation, transcript provenance, entity authorization, quotas, and provider fallback
remain server-owned. Moonshine should first be evaluated inside or beside `asa-voice-sidecar`; the UI can remain
provider-agnostic and display the returned provider/model diagnostics. A browser-local engine would be separate work
with pinned assets, integrity verification, explicit provenance, and an authorization-safe finalization contract.

Decision: prioritize a server-side Moonshine streaming proof of concept ahead of Parakeet for hands-free latency. Keep
Faster Whisper as the authoritative baseline until English command accuracy, entity preservation, RSS, and mixed-language
requirements pass.

| Candidate | Correct integration boundary | Live voice fit | Current recommendation |
|---|---|---|---|
| Faster Whisper | Existing in-process adapter | Proven current batch/raw/WebSocket path | Production baseline |
| Moonshine | Native/in-process provider or small sidecar service | Strongest streaming architecture; English-first | First alternative proof of concept |
| Parakeet | Separate OpenAI-compatible Go/ONNX service | Batch/SSE upload, not current PCM WebSocket | Batch benchmark and later raw bridge |
| LiteASR | Separate Transformers/Triton or MLX runtime | Retains Whisper live-window constraints | GPU/MLX batch benchmark only |

### First start downloads models

On first boot, Faster Whisper and Pocket TTS fetch their models from Hugging Face into `/root/.cache`
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

## Converting a custom Whisper model

Convert once per checkpoint version. Do not install conversion dependencies or download a model at
container startup. Resolve and record the immutable Hugging Face commit first:

```bash
git ls-remote https://huggingface.co/your-org/your-whisper refs/heads/main
pip install -r requirements-convert.txt
scripts/convert_whisper_ct2.sh \
  your-org/your-whisper \
  /models/your-whisper-ct2-int8 \
  int8 \
  FULL_COMMIT_SHA
STT_MODEL=/models/your-whisper-ct2-int8 uvicorn app.main:app --port 8090
```

The script rejects mutable `main`, converts into a temporary directory, records source metadata,
and moves the artifact into place only after `.asa_model_ready` exists. Mount the resulting model
read-only in production. To produce only a model artifact through Docker:

```bash
docker build --target model-converter \
  --build-arg STT_SOURCE_MODEL=your-org/your-whisper \
  --build-arg STT_SOURCE_REVISION=FULL_COMMIT_SHA \
  --build-arg STT_QUANTIZATION=int8 \
  --output type=local,dest=./converted-model .
```

For a model fine-tuned with Unsloth, first merge/export the adapter as a complete Hugging Face
Whisper checkpoint containing model weights, config, tokenizer, generation config, and feature
extractor. Upload that checkpoint at an immutable revision, run the same conversion command, then
benchmark the CT2 artifact before changing `STT_MODEL`.

### CrisperWhisper evaluation lane

`unsloth/CrisperWhisper` is not a drop-in production default for Setara. It is based on Whisper
Large v3, trained for English and German verbatim transcription, uses a retokenized vocabulary, and
is licensed CC BY-NC 4.0. That makes it both too large for the current combined 4 CPU / 3 GB
sidecar target and unsuitable for commercial deployment. Its special timestamp behavior also relies
on a custom Transformers fork and does not automatically carry over to faster-whisper.

For non-commercial evaluation only, the model revision inspected for this implementation is:

```text
unsloth/CrisperWhisper@4507962bae1df56f2c31bafb0df90ec9d6e0b2f4
```

Attempt conversion with the pinned command below and treat converter or tokenizer incompatibility as
a failed lane, not a reason to patch runtime dynamically:

```bash
scripts/convert_whisper_ct2.sh \
  unsloth/CrisperWhisper \
  /models/crisper-whisper-ct2-int8 \
  int8 \
  4507962bae1df56f2c31bafb0df90ec9d6e0b2f4
```

For Indonesian-English commands under the current resource cap, evaluate multilingual Whisper
`small` or a domain fine-tune of `small` first. CrisperWhisper should only proceed if licensing,
conversion compatibility, peak RSS, final latency, and entity accuracy all pass independently.

`cobrayyxx/whisper-small-indo-eng` is an Apache-2.0 Whisper Small checkpoint and its model card
explicitly documents CTranslate2 conversion. The pinned revision inspected here is:

```text
cobrayyxx/whisper-small-indo-eng@0d3a356eb29177e4d956beb163c14762d8ac0350
```

It is a valid benchmark candidate, but it was trained for Indonesian-to-English speech translation,
not mixed-language verbatim transcription. Its published metrics are BLEU/CHRF translation scores;
it publishes no WER, CER, mixed-command, or entity-preservation result. Converting it is straightforward:

```bash
scripts/convert_whisper_ct2.sh \
  cobrayyxx/whisper-small-indo-eng \
  /models/whisper-small-indo-eng-ct2-int8 \
  int8 \
  0d3a356eb29177e4d956beb163c14762d8ac0350
```

Do not promote it from the benchmark lane unless it preserves Indonesian terms and Setara entity
names while beating multilingual Whisper Small on the command corpus. A translation model can score
well while silently translating or rewriting exactly the words that tool routing needs.

---

## Configuration

All env keys are in `.env.example` and map 1:1 to active `app/config.py` fields. A contract test rejects missing,
extra, or unused settings so Compose and documentation cannot silently retain obsolete knobs.

| Variable | Default | Description |
|---|---|---|
| `ASA_VOICE_MODE` | `local` | Descriptive `local` \| `hosted` \| `hybrid` mode reported by health/models |
| `STT_PROVIDER` | `faster_whisper` | Active STT provider; unrecognized values fail startup |
| `STT_FALLBACK_PROVIDER` | `none` | Secondary STT provider used when the primary raises |
| `STT_ALLOW_PROVIDER_OVERRIDE` | `false` | Allow trusted development/admin/test clients to select a wired provider per `/stt` call |
| `TTS_PROVIDER` | `pocket_tts` | Active TTS provider; unrecognized values fail startup |
| `TTS_FALLBACK_PROVIDER` | `none` | Secondary TTS provider used when the primary raises |
| `STT_MODEL` | `distil-small.en` | faster-whisper model. `base.en` (lightest) → `distil-small.en` → `small.en` (most accurate) |
| `STT_DEVICE` | `cpu` | Inference device |
| `STT_COMPUTE_TYPE` | `int8` | CTranslate2 quantization |
| `STT_CPU_THREADS` | `4` | Decode thread count — use all cores for best latency |
| `STT_PARTIAL_BEAM_SIZE` | `1` | Greedy rolling-partial decode used only for display |
| `STT_FINAL_BEAM_SIZE` | `3` | Accuracy-oriented final decode used for commands |
| `STT_FINAL_VAD_FILTER` | `true` | Skip silent regions during final decoding |
| `STT_STREAM_INTERVAL_MS` | `600` | Rolling-window re-decode interval for `/stt/stream` |
| `STT_STREAM_ENERGY_THRESHOLD` | `0.02` | Skip partial decoding below the RMS speech threshold |
| `OPENAI_STT_MODEL` | `gpt-4o-mini-transcribe` | Hosted model for multipart `/stt`; does not make raw/WS provider-neutral |
| `TTS_ENGINE` | `pocket-tts` | TTS backend (only `pocket-tts` currently) |
| `TTS_DEFAULT_VOICE` | `asa_default` | Voice ID: `asa_default` (anna), `asa_bright` (eve), `asa_calm` (george) |
| `TTS_DEFAULT_MODEL` | `pocket-low` | Pocket TTS model variant |
| `TTS_SAMPLE_RATE` | `24000` | Output sample rate (Hz) |
| `TTS_COMPILE` | `true` | `torch.compile` the model at startup — first call is slow, subsequent calls faster |
| `TTS_STREAM_COALESCE` | `1` | Chunks to buffer before flushing in `/tts/stream`. `1` = lowest latency |
| `MAX_AUDIO_SECONDS` | `20` | Rolling WebSocket and raw PCM duration cap |
| `MAX_UPLOAD_SECONDS` | `300` | Multipart file duration cap |
| `MAX_UPLOAD_MB` | `15` | Max upload body size before `413` |
| `MAX_CONCURRENT_STT` | `1` | Max parallel STT requests |
| `MAX_CONCURRENT_TTS` | `1` | Max parallel TTS requests |
| `TMP_DIR` | `/tmp/asa-voice` | Scratch dir for audio processing |

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
