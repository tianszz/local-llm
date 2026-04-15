import json
import os
from pathlib import Path

MODELS_DIR = Path.home() / ".local-llm" / "models"
CONFIG_PATH = Path.home() / ".local-llm" / "config.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)

DEFAULTS = {
    "default_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "max_tokens": 4096,
    "temp": 0.7,
}


def load():
    if CONFIG_PATH.exists():
        return {**DEFAULTS, **json.loads(CONFIG_PATH.read_text())}
    return DEFAULTS.copy()


def save(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
