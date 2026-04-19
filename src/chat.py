from src.models import is_vision_model

_MAX_TOOL_ITERS = 10


def run(model_id, max_tokens, temp, image=None, system=None, tools=False):
    if is_vision_model(model_id):
        _chat_vlm(model_id, max_tokens, temp, image, system)
    else:
        if image:
            print(f"Warning: --image ignored, {model_id} is not a vision model.\n")
        _chat_llm(model_id, max_tokens, temp, system, tools)


def _chat_llm(model_id, max_tokens, temp, system=None, tools=False):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"Loading {model_id}...")
    model, tokenizer = load(model_id)
    print("Ready. Ctrl+C to exit.\n")

    tool_defs = None
    if tools:
        from src.tools import BUILTIN_TOOLS, parse_tool_calls, strip_tool_calls, execute_tool
        tool_defs = BUILTIN_TOOLS
        print(f"Tools enabled: {', '.join(t['function']['name'] for t in tool_defs)}\n")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        for _ in range(_MAX_TOOL_ITERS):
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if tool_defs:
                kwargs["tools"] = tool_defs
            try:
                text = tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                kwargs.pop("tools", None)
                text = tokenizer.apply_chat_template(messages, **kwargs)

            print("Assistant: ", end="", flush=True)
            response = ""
            sampler = make_sampler(temp=temp)
            last_chunk = None
            for chunk in stream_generate(model, tokenizer, prompt=text, max_tokens=max_tokens, sampler=sampler):
                print(chunk.text, end="", flush=True)
                response += chunk.text
                last_chunk = chunk

            if last_chunk:
                print(f"\n\n[{last_chunk.prompt_tokens} in | {last_chunk.generation_tokens} out | {last_chunk.generation_tps:.1f} tok/s]\n")
            else:
                print()

            if tool_defs:
                tool_calls = parse_tool_calls(response)
                if tool_calls:
                    content = strip_tool_calls(response)
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {"id": tc["id"], "type": "function",
                             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                            for tc in tool_calls
                        ],
                    })
                    for tc in tool_calls:
                        print(f"[tool: {tc['name']}] {tc['arguments']}")
                        result = execute_tool(tc["name"], tc["arguments"])
                        preview = result[:300] + ("…" if len(result) > 300 else "")
                        print(f"[result] {preview}\n")
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    continue  # re-enter agentic loop

            messages.append({"role": "assistant", "content": response})
            break
        else:
            print("[max tool iterations reached]\n")


def _chat_vlm(model_id, max_tokens, temp, image=None, system=None):
    from mlx_vlm import load, stream_generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"Loading {model_id} (vision)...")
    model, processor = load(model_id)
    config = load_config(model_id)
    print("Ready. Ctrl+C to exit.\n")
    if system:
        print(f"System: {system}\n")
    if image:
        print(f"Image: {image}\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt:
            continue

        num_images = 1 if image else 0
        formatted = apply_chat_template(processor, config, prompt, num_images=num_images)

        print("Assistant: ", end="", flush=True)
        response = ""
        for chunk in stream_generate(
            model, processor, formatted,
            image=image, max_tokens=max_tokens, temperature=temp
        ):
            print(chunk.text, end="", flush=True)
            response += chunk.text
        print(f"\n\n[{chunk.prompt_tokens} in | {chunk.generation_tokens} out | {chunk.generation_tps:.1f} tok/s]\n")
