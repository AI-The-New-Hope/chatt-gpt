#!/usr/bin/env python3
"""Terminal-friendly OpenAI Agents SDK chat loop with optional web search."""
from __future__ import annotations

import argparse
import os
import sys

from agents import Agent, ModelSettings, Runner, SQLiteSession, WebSearchTool
from agents.exceptions import AgentsException
from openai.types.shared import Reasoning

DEFAULT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.1")
DEFAULT_SESSION_ID = "terminal-chat"
DEFAULT_DB_PATH = "chat_history.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chat with OpenAI's hosted models via the Agents SDK, optionally letting them "
            "call the built-in web search tool for fresh information."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Responses-compatible model name (default: %(default)s)",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("AGENT_SESSION_ID", DEFAULT_SESSION_ID),
        help="Conversation/session identifier for SQLite memory (default: %(default)s)",
    )
    parser.add_argument(
        "--db-path",
        default=os.getenv("AGENT_SESSION_DB", DEFAULT_DB_PATH),
        help="SQLite file used to persist chat history (default: %(default)s)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.getenv("AGENT_MAX_TURNS", "16")),
        help="Safety cap for tool/LLM turns per user message (default: %(default)s)",
    )
    parser.add_argument(
        "--require-websearch",
        action="store_true",
        help="Force the agent to invoke the web search tool each turn (tool_choice=required)",
    )
    return parser.parse_args()


def build_agent(args: argparse.Namespace, thinking_mode: bool) -> Agent:
    tools = [WebSearchTool()]
    instructions = (
        "You are ChatGPT running inside a terminal. Provide clear, self-contained "
        "answers and use the WebSearch tool for time-sensitive or factual queries."
    )
    if thinking_mode:
        instructions += (
            " Reflect deeply before replying: write out your reasoning invisibly as a "
            "scratchpad and then produce a clean final answer."
        )
    agent_kwargs: dict[str, object] = {
        "name": "TerminalChatGPT",
        "instructions": instructions,
        "model": args.model,
        "tools": tools,
    }
    ms_kwargs: dict[str, object] = {}
    if args.require_websearch:
        ms_kwargs["tool_choice"] = "required"
    if thinking_mode:
        ms_kwargs["reasoning"] = Reasoning(effort="medium")
    if ms_kwargs:
        agent_kwargs["model_settings"] = ModelSettings(**ms_kwargs)
    return Agent(**agent_kwargs)


def prompt_for_thinking_mode() -> bool:
    """Ask the operator whether the session should enable 'thinking' assists."""
    while True:
        answer = input("Enable enhanced thinking mode? [y/N]: ").strip().lower()
        if not answer:
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY before running this script.", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    thinking_mode = prompt_for_thinking_mode()
    agent = build_agent(args, thinking_mode)
    session = SQLiteSession(args.session_id, db_path=args.db_path)

    thinking_msg = "ON" if thinking_mode else "off"
    print(
        "Chatting with model",
        args.model,
        f"— web search tool available. Thinking mode {thinking_msg}. Press Ctrl+C to exit.",
    )

    while True:
        try:
            user_input = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        try:
            result = Runner.run_sync(
                agent,
                user_input,
                max_turns=args.max_turns,
                session=session,
            )
        except AgentsException as exc:
            print(f"agent ! tool or model failure: {exc}")
            continue

        response_text = result.final_output or "(no final output returned)"
        print(f"agent > {response_text}\n")


if __name__ == "__main__":
    main()
