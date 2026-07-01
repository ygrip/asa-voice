# ASA Voice Sidecar — common dev/ops tasks.
# Usage: make <target>   (run `make help` for the list)

IMAGE        ?= ghcr.io/ygrip/asa-voice-sidecar:latest
PORT         ?= 8090
BASE         ?= http://localhost:$(PORT)
SAMPLE       ?= sample.wav
TTS_TEXT     ?= Build created and assigned to the plan.
TTS_VOICE    ?= asa_default

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## ── Docker ──────────────────────────────────────────────────────────────────
.PHONY: build
build: ## Build the Docker image via compose
	docker-compose build

.PHONY: up
up: ## Start the sidecar (foreground, with build)
	docker-compose up --build

.PHONY: up-d
up-d: ## Start the sidecar (detached, with build)
	docker-compose up --build -d

.PHONY: down
down: ## Stop and remove the sidecar
	docker-compose down

.PHONY: logs
logs: ## Tail sidecar logs
	docker-compose logs -f asa-voice

.PHONY: shell
shell: ## Open a shell in the running container
	docker-compose exec asa-voice bash

.PHONY: image
image: ## Build a tagged image (IMAGE=...)
	docker build -t $(IMAGE) .

.PHONY: push
push: ## Push the tagged image (IMAGE=...)
	docker push $(IMAGE)

## ── Local (no Docker) ───────────────────────────────────────────────────────
.PHONY: install
install: ## Install deps into the active venv (CPU torch)
	pip install --upgrade pip
	pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5.0" || pip install "torch>=2.5.0"
	pip install -r requirements.txt

.PHONY: run
run: ## Run the API locally with reload
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

.PHONY: compile
compile: ## Syntax-check all Python sources
	python -m py_compile app/*.py app/services/*.py app/routers/*.py
	@echo "py_compile OK"

## ── Smoke tests (need the service running) ───────────────────────────────────
.PHONY: health
health: ## GET /health
	curl -fsS $(BASE)/health | (python -m json.tool 2>/dev/null || cat)

.PHONY: models
models: ## GET /models
	curl -fsS $(BASE)/models | (python -m json.tool 2>/dev/null || cat)

.PHONY: test-stt
test-stt: ## POST /stt with SAMPLE=path.wav
	@test -f "$(SAMPLE)" || { echo "Set SAMPLE=<audio file> (default sample.wav)"; exit 1; }
	curl -fsS -X POST $(BASE)/stt -F "file=@$(SAMPLE)" | (python -m json.tool 2>/dev/null || cat)

.PHONY: test-tts
test-tts: ## POST /tts -> asa-output.wav (TTS_TEXT=, TTS_VOICE=)
	curl -fsS -X POST $(BASE)/tts \
		-H "Content-Type: application/json" \
		-d '{"text":"$(TTS_TEXT)","voiceId":"$(TTS_VOICE)"}' \
		--output asa-output.wav
	@echo "wrote asa-output.wav"

.PHONY: smoke
smoke: health models test-tts ## Run health + models + tts

## ── Cleanup ──────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove pyc, build artifacts, generated audio
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f asa-output.wav

.PHONY: clean-volumes
clean-volumes: ## Stop and drop the cache/tmp volumes (re-downloads models next start)
	docker-compose down -v
