from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import argparse
import os
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MODELS_DIR = Path.home() / ".local-llm" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Local LLM chat")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)
    print("Ready. Ctrl+C to exit.\n")

    messages = []

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
        sampler = make_sampler(temp=args.temp)
        for chunk in stream_generate(model, tokenizer, prompt=text, max_tokens=args.max_tokens, sampler=sampler):
            print(chunk.text, end="", flush=True)
            response += chunk.text
        print(f"\n\n[{chunk.generation_tokens} tokens, {chunk.generation_tps:.1f} tok/s]\n")

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
