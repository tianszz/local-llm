import base64
import gc
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()

# --- Global model state ---

_model      = None
_tokenizer  = None   # LLM only
_processor  = None   # VLM only
_vlm_config = None   # VLM only
_model_id   = None
_is_vlm     = False
_system     = None


def _unload():
    global _model, _tokenizer, _processor, _vlm_config
    import mlx.core as mx
    if _model is not None:
        del _model
    if _tokenizer is not None:
        del _tokenizer
    if _processor is not None:
        del _processor
    if _vlm_config is not None:
        del _vlm_config
    _model = _tokenizer = _processor = _vlm_config = None
    gc.collect()
    mx.metal.clear_cache()


def load_model(model_id, system=None):
    global _model, _tokenizer, _processor, _vlm_config, _model_id, _is_vlm, _system
    from src.models import is_vision_model
    _unload()
    print(f"Loading {model_id}...")
    if is_vision_model(model_id):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        _model, _processor = load(model_id)
        _vlm_config = load_config(model_id)
        _is_vlm = True
    else:
        from mlx_lm import load
        _model, _tokenizer = load(model_id)
        _is_vlm = False
    _model_id = model_id
    _system = system
    print("Server ready.")


# --- Inference helpers ---
# Each generator yields token strings, then a final stats dict.

def _stream_llm(prompt_text, max_tokens, temp):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temp)
    last = None
    for chunk in stream_generate(_model, _tokenizer, prompt=prompt_text, max_tokens=max_tokens, sampler=sampler):
        yield chunk.text
        last = chunk
    if last:
        yield {
            "prompt_tokens": last.prompt_tokens,
            "generation_tokens": last.generation_tokens,
            "tps": round(last.generation_tps, 1),
        }


def _stream_vlm(prompt_text, image_b64, max_tokens, temp):
    from mlx_vlm import stream_generate
    img_path = None
    if image_b64:
        raw = image_b64.split(",", 1)[-1] if "," in image_b64 else image_b64
        img_bytes = base64.b64decode(raw)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes)
            img_path = f.name
    try:
        last = None
        for chunk in stream_generate(
            _model, _processor, prompt_text,
            image=img_path, max_tokens=max_tokens, temperature=temp
        ):
            yield chunk.text
            last = chunk
        if last:
            yield {
                "prompt_tokens": getattr(last, "prompt_tokens", 0),
                "generation_tokens": getattr(last, "generation_tokens", 0),
                "tps": round(getattr(last, "generation_tps", 0.0), 1),
            }
    finally:
        if img_path:
            os.unlink(img_path)


def _prepend_system(messages):
    if _system and (not messages or messages[0].get("role") != "system"):
        return [{"role": "system", "content": _system}] + messages
    return messages


def _apply_chat_template(messages, image_b64=None):
    if _is_vlm:
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        num_images = 1 if image_b64 else 0
        return apply_chat_template(_processor, _vlm_config, prompt, num_images=num_images)
    return _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _stream_response(prompt_text, image_b64, max_tokens, temp):
    if _is_vlm:
        return _stream_vlm(prompt_text, image_b64, max_tokens, temp)
    return _stream_llm(prompt_text, max_tokens, temp)


# --- Request models ---

class LoadRequest(BaseModel):
    model_id: str


class PullRequest(BaseModel):
    model_id: str


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temp: float = 0.7


class ChatRequest(BaseModel):
    messages: list
    image: str = None
    max_tokens: int = 512
    temp: float = 0.7


class OpenAIRequest(BaseModel):
    model: str = None
    messages: list
    image: str = None
    max_tokens: int = None
    temperature: float = None
    stream: bool = False


# --- Endpoints ---

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/info")
def info():
    return {"model": _model_id, "kind": "VLM" if _is_vlm else "LLM"}


@app.get("/models")
def models():
    from src.models import list_models_data
    return list_models_data()


@app.post("/load")
def load(req: LoadRequest):
    load_model(req.model_id, _system)
    return {"model": _model_id, "kind": "VLM" if _is_vlm else "LLM"}


@app.post("/pull")
def pull(req: PullRequest):
    from src.models import pull as pull_model, list_models_data
    pull_model(req.model_id)
    return {"model_id": req.model_id, "models": list_models_data()}


@app.get("/personas")
def personas():
    from src.personas import PERSONAS_DIR
    if not PERSONAS_DIR.exists():
        return []
    return [p.stem for p in sorted(PERSONAS_DIR.glob("*.txt"))]


@app.get("/personas/{name}")
def persona_text(name: str):
    from src.personas import PERSONAS_DIR
    from fastapi.responses import PlainTextResponse
    path = PERSONAS_DIR / f"{name}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Persona not found")
    return PlainTextResponse(path.read_text())


@app.post("/generate")
def generate(req: GenerateRequest):
    def event_stream():
        for item in _stream_llm(req.prompt, req.max_tokens, req.temp):
            if isinstance(item, dict):
                yield f"data: {json.dumps({'type': 'stats', **item})}\n\n"
            else:
                yield f"data: {json.dumps({'text': item})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat")
def chat(req: ChatRequest):
    messages = _prepend_system(req.messages)
    prompt_text = _apply_chat_template(messages, req.image)
    def event_stream():
        for item in _stream_response(prompt_text, req.image, req.max_tokens, req.temp):
            if isinstance(item, dict):
                yield f"data: {json.dumps({'type': 'stats', **item})}\n\n"
            else:
                yield f"data: {json.dumps({'text': item})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
def openai_chat(req: OpenAIRequest):
    from src.config import load as load_cfg
    cfg = load_cfg()
    max_tokens = req.max_tokens or cfg["max_tokens"]
    temp = req.temperature if req.temperature is not None else cfg["temp"]

    messages = _prepend_system(req.messages)
    prompt_text = _apply_chat_template(messages, req.image)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_name = req.model or _model_id
    created = int(time.time())

    if req.stream:
        def event_stream():
            stats = None
            for item in _stream_response(prompt_text, req.image, max_tokens, temp):
                if isinstance(item, dict):
                    stats = item
                    continue
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if stats:
                done_chunk["stats"] = stats
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    stats = None
    tokens = []
    for item in _stream_response(prompt_text, req.image, max_tokens, temp):
        if isinstance(item, dict):
            stats = item
        else:
            tokens.append(item)
    response = "".join(tokens)
    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
    }
    if stats:
        result["stats"] = stats
    return result


def serve(model_id, host, port, system=None):
    import uvicorn
    load_model(model_id, system)
    uvicorn.run(app, host=host, port=port)
