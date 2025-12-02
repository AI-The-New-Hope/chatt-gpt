#!/usr/bin/env python3
"""
chat_terminal_agent.py
Simple OpenAI Agents SDK loop that mimics ChatGPT in a terminal.
Optional --voice flag adds push-to-talk input (:voice command) and spoken replies.
"""
import argparse
import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple

from agents import Agent, Runner  # provided by openai-agents
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

try:
    import sounddevice as sd  # only needed when --voice is on
    import soundfile as sf
except ImportError:  # voice mode will raise if dependencies missing
    sd = sf = None


@dataclass
class TerminalAgent:
    model: str
    voice_enabled: bool
    voice_name: str
    record_seconds: int
    sample_rate: int
    transcribe_model: str = "gpt-4o-mini-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    history: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.agent = Agent(
            name="ChatGPT Terminal Twin",
            model=self.model,
            instructions=(
                "You are ChatGPT inside a Unix terminal. "
                "Answer empathetically, keep formatting plain Markdown, "
                "and ask clarifying questions when needed."
            ),
        )
        self.audio_client = AsyncOpenAI()
        self.audio_player = LocalAudioPlayer() if self.voice_enabled else None
        if self.voice_enabled and (sd is None or sf is None):
            raise RuntimeError("Install sounddevice + soundfile to use --voice mode.")

    async def loop(self) -> None:
        print("Type your question, 'exit', or ':voice' to speak.")
        while True:
            user_text = await asyncio.to_thread(input, "\nyou> ")
            user_text = user_text.strip()
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break
            if user_text == ":voice":
                if not self.voice_enabled:
                    print("Voice capture disabled. Restart with --voice.")
                    continue
                user_text = await self._capture_voice_turn()

            reply = await self._ask_agent(user_text)
            print(f"\nai> {reply.strip()}")
            if self.voice_enabled:
                await self._speak(reply)

    async def _ask_agent(self, user_text: str) -> str:
        formatted = self._format_history(user_text)
        result = await Runner.run(self.agent, input=formatted)
        response = result.final_output.strip()
        self.history.extend([("user", user_text), ("assistant", response)])
        return response

    def _format_history(self, latest_user_text: str) -> str:
        lines = [
            f"{'User' if role == 'user' else 'Assistant'}: {text}"
            for role, text in self.history
        ]
        lines.append(f"User: {latest_user_text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    async def _capture_voice_turn(self) -> str:
        print(f"(Recording {self.record_seconds}s… speak now)")
        path = await asyncio.to_thread(self._record_audio_blocking)
        try:
            async with self.audio_client.audio.transcriptions.with_streaming_response.create(
                model=self.transcribe_model,
                file=open(path, "rb"),
                response_format="text",
            ) as stream:
                transcript = ""
                async for chunk in stream:
                    transcript += chunk
        finally:
            os.remove(path)
        cleaned = transcript.strip()
        print(f"(Heard) you> {cleaned}")
        return cleaned

    def _record_audio_blocking(self) -> str:
        frames = int(self.sample_rate * self.record_seconds)
        recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        tmp = NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        sf.write(tmp.name, recording, self.sample_rate)
        return tmp.name

    async def _speak(self, text: str) -> None:
        async with self.audio_client.audio.speech.with_streaming_response.create(
            model=self.tts_model,
            voice=self.voice_name,
            input=text,
        ) as response:
            await self.audio_player.play(response)


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running.")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChatGPT-like terminal agent via OpenAI Agents SDK."
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model for the agent.")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable push-to-talk input and spoken replies.",
    )
    parser.add_argument("--voice-name", default="coral", help="Voice for spoken output.")
    parser.add_argument(
        "--record-seconds", type=int, default=8, help="Microphone capture length per turn."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Microphone sample rate."
    )
    args = parser.parse_args()

    ensure_api_key()

    agent = TerminalAgent(
        model=args.model,
        voice_enabled=args.voice,
        voice_name=args.voice_name,
        record_seconds=args.record_seconds,
        sample_rate=args.sample_rate,
    )
    await agent.loop()


if __name__ == "__main__":
    asyncio.run(main())
