from pathlib import Path

PERSONAS_DIR = Path.home() / ".local-llm" / "personas"


def load(name):
    path = PERSONAS_DIR / f"{name}.txt"
    if not path.exists():
        raise SystemExit(f"Persona not found: {path}")
    return path.read_text().strip()
