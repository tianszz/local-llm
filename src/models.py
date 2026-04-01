import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

from src.config import MODELS_DIR


def hf_token_set():
    from huggingface_hub import get_token
    return get_token() is not None


def save_hf_token(token):
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)


def pull(model_id):
    if not hf_token_set():
        print("HuggingFace token not found.")
        token = input("Enter your HF token (huggingface.co/settings/tokens): ").strip()
        if not token:
            raise SystemExit("Token required to download models.")
        save_hf_token(token)
    print(f"Pulling {model_id}...")
    snapshot_download(repo_id=model_id)
    print("Done.")


def list_models():
    data = list_models_data()
    if not data:
        print("No models downloaded.")
        return
    for m in data:
        print(f"{m['id']}  [{m['kind']}]  {m['size_gb']:.1f} GB")


def list_models_data():
    dirs = sorted(MODELS_DIR.glob("models--*"))
    result = []
    for d in dirs:
        model_id = _dir_to_model_id(d.name)
        size = _dir_size_gb(d)
        kind = "VLM" if is_vision_model(model_id) else "LLM"
        result.append({"id": model_id, "size_gb": round(size, 1), "kind": kind})
    return result


def remove(model_id):
    path = MODELS_DIR / _model_id_to_dir(model_id)
    if not path.exists():
        print(f"Model not found: {model_id}")
        return
    shutil.rmtree(path)
    print(f"Removed {model_id}")


def is_vision_model(model_id):
    cfg = _read_config(model_id)
    if cfg is not None:
        return "vision_config" in cfg or "vl" in cfg.get("model_type", "").lower()
    # Fallback to name heuristic if not downloaded yet
    name = model_id.lower()
    return any(k in name for k in ["-vl", "vision", "vlm"])


def snapshot_path(model_id):
    """Return local snapshot path for a cached model, or None."""
    refs = MODELS_DIR / _model_id_to_dir(model_id) / "refs" / "main"
    if not refs.exists():
        return None
    snap_hash = refs.read_text().strip()
    path = MODELS_DIR / _model_id_to_dir(model_id) / "snapshots" / snap_hash
    return path if path.exists() else None


def _read_config(model_id):
    path = snapshot_path(model_id)
    if path is None:
        return None
    config_file = path / "config.json"
    if not config_file.exists():
        return None
    return json.loads(config_file.read_text())


def _model_id_to_dir(model_id):
    return "models--" + model_id.replace("/", "--")


def _dir_to_model_id(dir_name):
    return dir_name[len("models--"):].replace("--", "/", 1)


def _dir_size_gb(path):
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)
