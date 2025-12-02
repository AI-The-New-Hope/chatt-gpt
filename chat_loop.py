#!/usr/bin/env python3
"""
Minimal ChatGPT-like REPL using the openai-agents SDK (v0.6.1).
- Keeps conversation history in a local SQLite DB.
- Runs until the user types 'exit'/'quit' or presses Ctrl+D/Ctrl+C.
"""

import asyncio
import os
import sys
from pathlib import Path

from agents import Agent, Runner
from agents.extensions.memory import AdvancedSQLiteSession


def ensure_api_key() -> None:
    """Fail fast if OPENAI_API_KEY is not set."""
    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write("ERROR: Set OPENAI_API_KEY in your environment.\n")
        sys.exit(1)


async def main() -> None:
    ensure_api_key()

    # One lightweight agent with ChatGPT-like persona.
    agent = Agent(
        name="TerminalChatGPT",
        instructions="You are a helpful, concise assistant that replies like ChatGPT.",
    )

    # Persist conversation + usage to a small local DB.
    db_path = Path("chat_history.db")
    session = AdvancedSQLiteSession(
        session_id="terminal_chat",
        db_path=db_path,
        create_tables=True,
    )

    print("ChatGPT-style CLI. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Send the message to the agent and stream back the reply.
        result = await Runner.run(agent, user_input, session=session)
        print(f"Assistant> {result.final_output}")

        # Track usage; safe to ignore if the DB is read-only.
        await session.store_run_usage(result)


if __name__ == "__main__":
    asyncio.run(main())
