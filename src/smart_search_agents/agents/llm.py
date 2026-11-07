"""Chat backends with tool calling: Ollama (local, default) and Claude (cloud, opt-in).

Each backend keeps its own conversation in its provider's native format, so provider
details (Claude's thinking blocks, Ollama's tool messages) never leak into the agent loop.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict  # JSON Schema of the arguments


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    call: ToolCall
    content: str
    is_error: bool = False


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end"  # end | tool_use | max_tokens | refusal


class ChatBackend(Protocol):
    name: str
    local: bool  # False = prompts and tool results leave the laptop

    def start(self, system: str, tools: list[ToolSpec]) -> None: ...

    def send_user(self, text: str) -> Turn: ...

    def send_tool_results(self, results: list[ToolResult], note: str | None = None) -> Turn: ...


class OllamaBackend:
    """Ollama's /api/chat with tools. Runs entirely on this machine."""

    local = True

    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434",
                 timeout: float = 300.0, transport: httpx.BaseTransport | None = None):
        self.model = model
        self.name = f"ollama:{model}"
        self.http = httpx.Client(base_url=url, timeout=timeout, transport=transport)
        self.messages: list[dict] = []
        self.tools: list[dict] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.messages = [{"role": "system", "content": system}]
        self.tools = [
            {"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
            for t in tools
        ]

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._chat()

    def send_tool_results(self, results: list[ToolResult], note: str | None = None) -> Turn:
        for r in results:
            self.messages.append({"role": "tool", "tool_name": r.call.name, "content": r.content})
        if note:
            self.messages.append({"role": "user", "content": note})
        return self._chat()

    def _chat(self) -> Turn:
        body = {"model": self.model, "messages": self.messages, "stream": False,
                "options": {"temperature": 0.2}}
        if self.tools:
            body["tools"] = self.tools
        try:
            r = self.http.post("/api/chat", json=body)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed ({exc}). Is it running? Try: ollama serve  and  "
                f"ollama pull {self.model}"
            ) from exc
        msg = r.json()["message"]
        self.messages.append(msg)
        raw = [(tc.get("function", {}).get("name", ""), tc.get("function", {}).get("arguments"))
               for tc in msg.get("tool_calls") or []]
        text = msg.get("content") or ""
        if not raw and (inline := self._inline_call(text)):
            raw, text = [inline], ""  # small models sometimes write the call as text
            msg["tool_calls"] = [{"function": {"name": inline[0], "arguments": inline[1]}}]
            msg["content"] = ""
        calls = [ToolCall(uuid.uuid4().hex[:12], name, _clean_args(args)) for name, args in raw]
        return Turn(text, calls, "tool_use" if calls else "end")

    def _inline_call(self, text: str) -> tuple[str, dict] | None:
        s = text.strip().strip("`").removeprefix("json").strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        try:
            obj = json.loads(s)
        except ValueError:
            return None
        names = {t["function"]["name"] for t in self.tools}
        if isinstance(obj, dict) and obj.get("name") in names:
            return obj["name"], obj.get("parameters") or obj.get("arguments") or {}
        return None


class ClaudeBackend:
    """Claude via the Anthropic SDK, manual tool-use loop.

    Thinking is adaptive (the model default); depth is set with effort. Server-side refusal
    fallbacks are on ("default" routing). Whole assistant content is sent back each turn so
    thinking blocks are preserved."""

    local = False
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium",
                 max_tokens: int = 16000, client=None, api_key: str | None = None):
        import anthropic

        self.client = client or anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = f"claude:{model}"
        self.effort = effort
        self.max_tokens = max_tokens
        self.system = ""
        self.tools: list[dict] = []
        self.messages: list[dict] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.schema}
                      for t in tools]
        self.messages = []

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._create()

    def send_tool_results(self, results: list[ToolResult], note: str | None = None) -> Turn:
        # All results of one assistant turn go back in a single user message.
        content: list[dict] = [
            {"type": "tool_result", "tool_use_id": r.call.id, "content": r.content,
             "is_error": r.is_error}
            for r in results
        ]
        if note:
            content.append({"type": "text", "text": note})
        self.messages.append({"role": "user", "content": content})
        return self._create()

    def _create(self) -> Turn:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            output_config={"effort": self.effort},
            betas=[self.FALLBACK_BETA],
            fallbacks="default",
        )
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "refusal":
            return Turn(text or "The model declined this request.", [], "refusal")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content
                 if b.type == "tool_use"]
        if response.stop_reason == "max_tokens":
            return Turn(text, [], "max_tokens")  # never run a possibly truncated tool call
        return Turn(text, calls, "tool_use" if calls else "end")


_EMPTY = {"null", "none", "undefined", ""}


def _clean_args(args) -> dict:
    """Parse string arguments and drop placeholder values ("null", "None", "") that small
    local models put in optional fields, which would otherwise fail schema validation."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {}
    if not isinstance(args, dict):
        return {}
    return {k: v for k, v in args.items()
            if v is not None and not (isinstance(v, str) and v.strip().lower() in _EMPTY)}


def make_backend(settings) -> ChatBackend:
    a = settings.agents
    if a.backend == "claude":
        from ..settings import get_claude_key

        return ClaudeBackend(a.claude_model, a.claude_effort, api_key=get_claude_key())
    if a.backend == "ollama":
        return OllamaBackend(a.model, settings.core.ollama_url)
    raise ValueError(f"agents.backend must be 'ollama' or 'claude', not {a.backend!r}")
