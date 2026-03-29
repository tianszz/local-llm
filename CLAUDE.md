# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project

A local-first LLM inference tool for Apple Silicon, built on `mlx-lm`. The development roadmap lives in `plan.md` — follow phases in order unless told otherwise.

## Commands

```bash
# Run inference
python main.py

# Install dependencies
pip install -r requirements.txt
```

No build step, no test suite yet. As the CLI and API are built out, commands will be added here.

## Architecture

Currently a single-file script (`main.py`). As phases are completed, the structure will grow:

- `main.py` — entrypoint and CLI wiring
- `src/` — modules, created only when needed (not yet)
- `~/.local-llm/config.json` — user config (default model, settings) — Phase 2
- `~/.local-llm/personas/` — named system prompts — Phase 4

Stack: `mlx-lm` for inference, `argparse` subcommands for CLI, `FastAPI` for the HTTP API (Phase 3+), models from `mlx-community` on HuggingFace.

**Key constraint:** Apple Silicon / MacBook — model loading cost, memory footprint, and inference speed all matter. Prefer 4-bit quantized models.

---

## Behavior (Non-Negotiable)

### Truthfulness
Never fabricate results, APIs, benchmarks, or implementation details. If unsure, say so explicitly.

### Act like a technical co-founder
Think in terms of product impact. Challenge unclear requirements. Suggest better approaches when warranted.

### Plan before building
For any non-trivial task: restate the goal, break it into steps, identify risks, propose a plan — before writing code.

### Maintain `design.md`
Treat `design.md` as the single source of truth for system design. Update it when architecture changes, new components are introduced, or trade-offs are made.

### Code quality
- Minimal but complete — easy to read in one pass
- No premature abstractions, no defensive boilerplate unless needed
- No docstrings or type annotations unless asked
- After writing: verify correctness, simplify if possible

### Git
Small, meaningful commits at logical checkpoints. Commit message = what changed + why.
