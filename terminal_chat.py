#!/usr/bin/env python3
"""Terminal-friendly OpenAI Agents SDK chat loop with Context7 and web search tools."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from agents import Agent, HostedMCPTool, Runner, SQLiteSession, WebSearchTool
from agents.exceptions import AgentsException

DEFAULT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-4.1-mini")
DEFAULT_CONTEXT7_URL = os.getenv("CONTEXT7_URL", "https://mcp.context7.com/mcp")
DEFAULT_SESSION_ID = "terminal-chat"
DEFAULT_DB_PATH = "chat_history.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chat with OpenAI's hosted models via the Agents SDK while giving them "
            "Context7 MCP documentation tools and the built-in WebSearchTool."
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
        "--context7-label",
        default=os.getenv("CONTEXT7_LABEL", "context7"),
        help="Label shown to the model for the Context7 MCP server (default: %(default)s)",
    )
    parser.add_argument(
        "--context7-url",
        default=DEFAULT_CONTEXT7_URL,
        help="Public MCP endpoint for Context7 (default: %(default)s)",
    )
    parser.add_argument(
        "--context7-approval",
        default=os.getenv("CONTEXT7_APPROVAL_POLICY", "never"),
        choices=["never", "always"],
        help="When to require manual approval before running Context7 tools",
    )
    parser.add_argument(
        "--require-websearch",
        action="store_true",
        help="Force the agent to always reach for tools when answering (sets tool_choice)",
    )
    return parser.parse_args()


def build_context7_tool(args: argparse.Namespace) -> HostedMCPTool:
    headers = {}
    api_key = os.getenv("CONTEXT7_API_KEY")
    if api_key:
        headers["CONTEXT7_API_KEY"] = api_key

    tool_config = {
        "type": "mcp",
        "server_label": args.context7_label,
        "server_url": args.context7_url,
        "require_approval": args.context7_approval,
    }
    if headers:
        tool_config["headers"] = headers

    return HostedMCPTool(tool_config=tool_config)


def build_agent(args: argparse.Namespace) -> Agent:
    tools = [WebSearchTool(), build_context7_tool(args)]
    agent_kwargs: dict[str, object] = {
        "name": "TerminalChatGPT",
        "instructions": (
            "You are a concise AI assistant that cites live sources when appropriate. "
            "Use the web search tool for current events and defer to the Context7 MCP "
            "tools for fresh API and library docs before you improvise."
        ),
        "model": args.model,
        "tools": tools,
    }
    if args.require_websearch:
        agent_kwargs["model_settings"] = {"tool_choice": "required"}
    return Agent(**agent_kwargs)


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY before running this script.", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    agent = build_agent(args)
    session = SQLiteSession(args.session_id, db_path=args.db_path)

    print(
        "Chatting with model", args.model,
        "— Context7 + web search tools enabled. Press Ctrl+C to exit.",
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
