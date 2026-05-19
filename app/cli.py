"""
Legal RAG — Interactive CLI
===========================
Run this script to chat directly with the engine without the API server.

Usage:
    python cli.py --provider openai --model gpt-4o-mini
    python cli.py --provider ollama --model mistral
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import LegalRAGEngine


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          ⚖️           Legal Assistant — RAG Chat             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
Type your question in French or Arabic.
Commands: /ingest <dir>  /clear  /history  /quit
"""


def print_json(label: str, data: dict):
    """Pretty-print a JSON payload for debugging."""
    print(f"\n{'─'*20} {label} {'─'*20}")
    print(json.dumps({k: v for k, v in data.items() if k != "context"}, ensure_ascii=False, indent=2))
    print("─" * 50)


def run_cli(args):
    print(BANNER)

    engine = LegalRAGEngine(
        provider        = args.provider,
        model           = args.model,
        temperature     = args.temperature,
        api_key         = None,  # providers read GOOGLE_API_KEY / OPENAI_API_KEY from .env directly
        ollama_base_url = args.ollama_url,
        vector_store_dir= args.vector_store,
        k_results       = args.k,
    )

    context: list[dict] = []

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # ── Built-in commands ──
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye.")
            break

        if user_input.lower() == "/clear":
            context = []
            print("✅ Conversation cleared.")
            continue

        if user_input.lower() == "/history":
            print(json.dumps(context, ensure_ascii=False, indent=2))
            continue

        if user_input.lower().startswith("/ingest"):
            parts = user_input.split(maxsplit=1)
            source = parts[1] if len(parts) > 1 else "./data/tmp"
            print(f"⏳ Ingesting documents from: {source}")
            try:
                n = engine.ingest_documents(source)
                print(f"✅ Done — {n} chunks indexed.")
            except Exception as e:
                print(f"❌ Ingestion failed: {e}")
            continue

        # ── Normal chat turn ──
        payload = {
            "user_prompt": user_input,
            "context":     context,
        }

        if args.debug:
            print_json("→ SENDING", payload)

        print("⏳ Thinking...", end="\r")
        result = engine.chat(payload)
        print("              ", end="\r")  # clear "Thinking..."

        if args.debug:
            print_json("← RECEIVED", result)

        context = result["context"]  # persist updated history

        print(f"\n⚖️  Assistant:\n{result['response']}")

        if result.get("query") and args.debug:
            print(f"\n🔍 [DB query used: {result['query']}]")


def main():
    parser = argparse.ArgumentParser(description="Legal RAG CLI")
    parser.add_argument("--provider",     default=os.getenv("LLM_PROVIDER", "gemini"),
                                          choices=["openai", "ollama", "gemini"])
    parser.add_argument("--model",        default=os.getenv("LLM_MODEL", "gemini-1.5-flash"))
    parser.add_argument("--temperature",  default=float(os.getenv("LLM_TEMPERATURE", "0.1")), type=float)
    parser.add_argument("--ollama-url",   default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--vector-store", default=os.getenv("VECTOR_STORE_DIR", "./data/vector_stores"))
    parser.add_argument("-k",             default=int(os.getenv("K_RESULTS", "4")), type=int)
    parser.add_argument("--debug",        action="store_true", help="Print full JSON payloads")
    args = parser.parse_args()

    run_cli(args)


if __name__ == "__main__":
    main()
