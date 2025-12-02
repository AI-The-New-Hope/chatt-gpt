#!/usr/bin/env python3
"""Simple endless ChatGPT-like terminal loop using openai-agents."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from agents import Agent, Runner, SQLiteSession

MIN_PY_VERSION: Final[tuple[int, int]] = (3, 10)


def ensure_supported_python() -> None:
    """Exit early if the interpreter is too old for openai-agents."""
    if sys.version_info < MIN_PY_VERSION:
        major, minor = MIN_PY_VERSION
        print(
            "This script requires Python >= "
            f"{major}.{minor} because openai-agents 0.6.1 uses PEP 604 typing.",
            file=sys.stderr,
        )
        print("Install Python 3.10+ (e.g., via Homebrew or pyenv) and rerun.", file=sys.stderr)
        sys.exit(1)


def require_api_key() -> str:
    """Return the OpenAI API key or exit with a helpful message."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    print(
        "OPENAI_API_KEY is not set. Export it before running this script, e.g.\n"
        "  export OPENAI_API_KEY=sk-...",
        file=sys.stderr,
    )
    sys.exit(1)


def build_agent() -> Agent:
    """Create a concise, friendly terminal agent."""
    return Agent(
        name="Terminal ChatGPT",
        instructions=(
            "You are an upbeat assistant in a developer terminal. "
            "Keep replies under six sentences and volunteer command examples when useful."
        ),
    )


def build_session() -> SQLiteSession:
    """Keep chat history in a SQLite file next to this script."""
    session_path = Path(__file__).with_suffix(".db")
    return SQLiteSession("terminal_chat", str(session_path))


def main() -> None:
    ensure_supported_python()
    require_api_key()
    agent = build_agent()
    session = build_session()

    print("Type 'exit' (or Ctrl+C) to quit. Questions? Try 'help'.\n")

    while True:  # Endless loop until the user opts out
        try:
            user_text = input("You  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = Runner.run_sync(agent, user_text, session=session)
        except Exception as exc:  # Surface SDK or API issues but keep loop alive
            print(f"AI   > Error: {exc}\n")
            continue

        assistant_text = result.final_output or "(No response)"
        print(f"AI   > {assistant_text}\n")


if __name__ == "__main__":
    main()
