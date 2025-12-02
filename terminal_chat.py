import asyncio
import os
from agents import Agent, Runner, SQLiteSession

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant responding from the terminal. "
    "Keep answers concise unless the user asks for more detail."
)

async def main() -> None:
    agent = Agent(name="Terminal Assistant", instructions=SYSTEM_INSTRUCTION)
    session = SQLiteSession("terminal_chat")

    print("ChatGPT-like terminal session. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break

        result = await Runner.run(agent, user_message, session=session)
        print(f"Assistant: {result.final_output}\n")

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable before running.")
    asyncio.run(main())
