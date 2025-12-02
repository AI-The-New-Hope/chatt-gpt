#!/usr/bin/env python3
"""Minimal terminal chat loop powered by openai-agents."""

import os
import sys

from agents import Agent, OpenAIConversationsSession, Runner


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable before running this script.", file=sys.stderr)
        return 1

    agent = Agent(
        name="Terminal Assistant",
        instructions="You are a helpful AI assistant chatting through a terminal interface. Keep replies clear and helpful.",
    )
    session = OpenAIConversationsSession()

    print("Starting Terminal Assistant. Type 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            return 0

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return 0

        try:
            result = Runner.run_sync(agent, user_input, session=session)
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"Error while contacting the agent: {exc}", file=sys.stderr)
            continue

        print(f"Assistant: {result.final_output}")


if __name__ == "__main__":
    raise SystemExit(main())
