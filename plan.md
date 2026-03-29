# local-llm — Development Plan

## Current State
Single-script inference using `mlx-lm` with a hardcoded prompt and model.

---

## Phase 1 — Usable CLI
Make the script actually useful day-to-day.

- [ ] Accept prompt from CLI args or stdin (`argparse`)
- [ ] Make model configurable via `--model` flag
- [ ] Add `--max-tokens` and `--temp` flags
- [ ] Support multi-turn chat (keep message history in memory)
- [ ] Stream tokens to stdout as they generate

---

## Phase 2 — Model Management
Make it easy to switch and manage local models.

- [ ] Config file (`~/.local-llm/config.json`) for default model and settings
- [ ] `list-models` command to show downloaded models
- [ ] `pull` command to download a model by name
- [ ] `remove` command to delete a cached model

---

## Phase 3 — HTTP API
Expose a local REST API so other tools can use it.

- [ ] FastAPI server with `/chat` and `/generate` endpoints
- [ ] OpenAI-compatible `/v1/chat/completions` endpoint (drop-in for existing tools)
- [ ] Streaming responses via SSE
- [ ] Configurable host/port

---

## Phase 4 — System Prompt & Personas
Support customizable behavior.

- [ ] `--system` flag to set a system prompt per session
- [ ] Named personas stored in `~/.local-llm/personas/`
- [ ] `--persona` flag to load a persona by name

---

## Phase 5 — Context & RAG (stretch)
Give the model access to local documents.

- [ ] Ingest local files into a vector store (e.g. `faiss` or `chromadb`)
- [ ] Retrieve relevant chunks and inject into context window
- [ ] `index` command to add files/directories to the knowledge base

---

## Model Shortlist
Small models that work well with MLX on Apple Silicon:

| Model | Size | Notes |
|---|---|---|
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4GB | — |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4.3GB | Current default |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~2GB | Lighter option |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4GB | Strong at coding |
| `mlx-community/phi-4-4bit` | ~8GB | Strong reasoning |
