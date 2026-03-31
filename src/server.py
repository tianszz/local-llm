import json
import time
import uuid

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()

_model = None
_tokenizer = None
_model_id = None
_system = None


def load_model(model_id, system=None):
    global _model, _tokenizer, _model_id, _system
    from mlx_lm import load
    print(f"Loading {model_id}...")
    _model, _tokenizer = load(model_id)
    _model_id = model_id
    _system = system
    print("Server ready.")


def _stream_tokens(prompt_text, max_tokens, temp):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temp)
    for chunk in stream_generate(_model, _tokenizer, prompt=prompt_text, max_tokens=max_tokens, sampler=sampler):
        yield chunk.text


def _prepend_system(messages):
    if _system and (not messages or messages[0].get("role") != "system"):
        return [{"role": "system", "content": _system}] + messages
    return messages


# --- Request models ---

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temp: float = 0.7


class ChatRequest(BaseModel):
    messages: list
    max_tokens: int = 512
    temp: float = 0.7


class OpenAIRequest(BaseModel):
    model: str = None
    messages: list
    max_tokens: int = None
    temperature: float = None
    stream: bool = False


# --- Endpoints ---

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/info")
def info():
    return {"model": _model_id}


@app.post("/generate")
def generate(req: GenerateRequest):
    def event_stream():
        for token in _stream_tokens(req.prompt, req.max_tokens, req.temp):
            yield f"data: {json.dumps({'text': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat")
def chat(req: ChatRequest):
    messages = _prepend_system(req.messages)
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    def event_stream():
        for token in _stream_tokens(text, req.max_tokens, req.temp):
            yield f"data: {json.dumps({'text': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
def openai_chat(req: OpenAIRequest):
    from src.config import load as load_cfg
    cfg = load_cfg()
    max_tokens = req.max_tokens or cfg["max_tokens"]
    temp = req.temperature if req.temperature is not None else cfg["temp"]

    messages = _prepend_system(req.messages)
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_name = req.model or _model_id
    created = int(time.time())

    if req.stream:
        def event_stream():
            for token in _stream_tokens(text, max_tokens, temp):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = "".join(_stream_tokens(text, max_tokens, temp))
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
    }


def serve(model_id, host, port, system=None):
    import uvicorn
    load_model(model_id, system)
    uvicorn.run(app, host=host, port=port)
