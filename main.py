import src.config  # sets HF_HUB_CACHE before any HF imports
import argparse
from src import chat, models, config


def main():
    cfg = config.load()

    parser = argparse.ArgumentParser(description="Local LLM")
    parser.add_argument("--model", default=cfg["default_model"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--temp", type=float, default=cfg["temp"])
    parser.add_argument("--image", help="Image path (vision models only)")

    sub = parser.add_subparsers(dest="command")

    pull_p = sub.add_parser("pull", help="Download a model from HuggingFace")
    pull_p.add_argument("model_id")

    sub.add_parser("list", help="List downloaded models")

    remove_p = sub.add_parser("remove", help="Delete a local model")
    remove_p.add_argument("model_id")

    serve_p = sub.add_parser("serve", help="Start the HTTP API server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if args.command == "pull":
        models.pull(args.model_id)
    elif args.command == "list":
        models.list_models()
    elif args.command == "remove":
        models.remove(args.model_id)
    elif args.command == "serve":
        from src import server
        server.serve(args.model, args.host, args.port)
    else:
        chat.run(args.model, args.max_tokens, args.temp, args.image)


if __name__ == "__main__":
    main()
