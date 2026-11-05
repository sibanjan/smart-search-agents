"""The agent loop shared by every agent: model turn -> tool calls -> results -> repeat."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from .llm import ChatBackend, ToolResult
from .tools import ToolBox

FINISH_NOTE = "Tool budget used up. Stop calling tools and give your final answer now."


@dataclass
class AgentResult:
    text: str
    rounds: int
    stop: str
    tool_log: list[dict] = field(default_factory=list)  # {"tool", "args", "is_error"}


async def run_agent(
    llm: ChatBackend,
    tools: ToolBox,
    system: str,
    task: str,
    max_rounds: int = 6,
    on_event: Callable[[str, dict], None] | None = None,
) -> AgentResult:
    emit = on_event or (lambda kind, data: None)
    llm.start(system, tools.specs)
    turn = await asyncio.to_thread(llm.send_user, task)
    log: list[dict] = []
    rounds = 0
    while turn.tool_calls:
        rounds += 1
        results: list[ToolResult] = []
        for call in turn.tool_calls:
            emit("tool", {"name": call.name, "args": call.args})
            res = await tools.call(call)
            log.append({"tool": call.name, "args": call.args, "is_error": res.is_error})
            results.append(res)
        note = FINISH_NOTE if rounds >= max_rounds else None
        turn = await asyncio.to_thread(llm.send_tool_results, results, note)
        if rounds >= max_rounds and turn.tool_calls:
            # The model ignored the note; answer with no further tool access.
            turn.tool_calls = []
            turn.stop = "max_rounds"
    emit("answer", {"text": turn.text})
    return AgentResult(turn.text, rounds, turn.stop, log)
