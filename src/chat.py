from src.models import is_vision_model


def run(model_id, max_tokens, temp, image=None, system=None):
    if is_vision_model(model_id):
        _chat_vlm(model_id, max_tokens, temp, image, system)
    else:
        if image:
            print(f"Warning: --image ignored, {model_id} is not a vision model.\n")
        _chat_llm(model_id, max_tokens, temp, system)


def _chat_llm(model_id, max_tokens, temp, system=None):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"Loading {model_id}...")
    model, tokenizer = load(model_id)
    print("Ready. Ctrl+C to exit.\n")

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
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("Assistant: ", end="", flush=True)
        response = ""
        sampler = make_sampler(temp=temp)
        for chunk in stream_generate(model, tokenizer, prompt=text, max_tokens=max_tokens, sampler=sampler):
            print(chunk.text, end="", flush=True)
            response += chunk.text
        print(f"\n\n[{chunk.prompt_tokens} in | {chunk.generation_tokens} out | {chunk.generation_tps:.1f} tok/s]\n")

        messages.append({"role": "assistant", "content": response})


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
