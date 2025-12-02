"""Simple local agent runner compatible with chat_terminal.py."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI


@dataclass
class Agent:
    """Descriptor for an OpenAI-powered assistant."""

    name: str
    instructions: str


@dataclass
class RunnerResult:
    """Container for the assistant's response."""

    final_output: str
    raw_response: object


class OpenAIConversationsSession:
    """Tracks running conversation state."""

    def __init__(self) -> None:
        self._messages: List[Dict[str, str]] = []
        self._has_system: bool = False

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._messages

    def prepare_messages(self, agent: Agent, user_message: str) -> List[Dict[str, str]]:
        if not self._has_system:
            self._messages.append({"role": "system", "content": agent.instructions})
            self._has_system = True
        self._messages.append({"role": "user", "content": user_message})
        return self._messages

    def add_assistant_response(self, response: str) -> None:
        self._messages.append({"role": "assistant", "content": response})


class Runner:
    """Minimal synchronous runner around the OpenAI chat API."""

    _client: Optional[OpenAI] = None

    @classmethod
    def _client_instance(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI()
        return cls._client

    @classmethod
    def run_sync(
        cls,
        agent: Agent,
        user_input: str,
        session: Optional[OpenAIConversationsSession] = None,
        *,
        model: Optional[str] = None,
    ) -> RunnerResult:
        if session is None:
            session = OpenAIConversationsSession()

        messages = session.prepare_messages(agent, user_input)
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        client = cls._client_instance()
        response = client.chat.completions.create(model=model_name, messages=messages)
        message_content = response.choices[0].message.content or ""

        session.add_assistant_response(message_content)
        return RunnerResult(final_output=message_content, raw_response=response)
