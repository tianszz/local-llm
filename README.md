# local-llm

Local-first LLM inference for Apple Silicon, built on [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Requirements

- Apple Silicon Mac
- Python 3.10+

```bash
pip install -r requirements.txt
```

## Usage

### Chat

```bash
python main.py                          # use default model
python main.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
python main.py --max-tokens 512 --temp 0.7
python main.py --image /path/to/image.png   # vision models only
python main.py --system "You are a concise assistant."
python main.py --persona coder          # load from ~/.local-llm/personas/coder.txt
```

### Model management

```bash
python main.py pull mlx-community/Qwen2.5-7B-Instruct-4bit
python main.py list
python main.py remove mlx-community/Qwen2.5-7B-Instruct-4bit
```

### HTTP API server

```bash
python main.py serve                    # 127.0.0.1:8080
python main.py serve --host 0.0.0.0 --port 9000
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web chat UI |
| `POST` | `/chat` | Chat (SSE streaming) |
| `POST` | `/generate` | Single-turn generation |
| `POST` | `/v1/chat/completions` | OpenAI-compatible endpoint |
| `GET` | `/models` | List downloaded models |
| `POST` | `/load` | Hot-swap the active model |
| `GET` | `/personas` | List available personas |
| `GET` | `/personas/<name>` | Get persona text |

## Configuration

`~/.local-llm/config.json` controls defaults:

```json
{
  "default_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "max_tokens": 1024,
  "temp": 0.7
}
```

Models are cached in the directory specified by `MODELS_DIR` in config (defaults to `~/.local-llm/models`).

## Personas

Place plain-text system prompts in `~/.local-llm/personas/<name>.txt` and load them with `--persona <name>`.

## Recommended models

| Model | Size | Notes |
|-------|------|-------|
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~2 GB | Lightest option |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4.3 GB | Default; strong all-around |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4 GB | — |
| `mlx-community/phi-4-4bit` | ~8 GB | Strong reasoning |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | ~20 GB | Largest tested; fits 48 GB |
