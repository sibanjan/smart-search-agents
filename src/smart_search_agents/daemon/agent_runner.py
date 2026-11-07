"""Run the Ask and Curator agents for the web UI, streaming what they do as server-sent events.

The agents still reach the index only through MCP: each run gets an in-process MCP server
that calls this daemon's HTTP API, exactly as `ssa mcp` would. A run happens on its own
thread with its own event loop, because the MCP tools make blocking HTTP calls back into
this server and must never run on the server's own loop.

Events: start {model, local}, tool {name, args}, text {text}, text_reset, answer {text},
done {rounds, stop}, error {message}.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import queue
import threading
from typing import Callable, Iterator

from ..agents import ask as ask_agent
from ..agents import curator as curator_agent
from ..agents.base import run_agent
from ..agents.llm import make_backend
from ..agents.tools import McpToolBox
from ..settings import Settings

CLAUDE_CONTEXT_CHARS = 40000
_GROUP = getattr(builtins, "BaseExceptionGroup", ())  # Python 3.11+


class Stopped(Exception):
    """The browser went away; stop at the next event."""


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        mcp_server_factory: Callable[[], object],
        backend_factory: Callable[[Settings], object] = make_backend,
    ):
        self.settings = settings
        self.mcp_server_factory = mcp_server_factory
        self.backend_factory = backend_factory

    def ask(self, question: str) -> Iterator[str]:
        return self._stream(ask_agent.TOOLS, ask_agent.SYSTEM, question,
                            self.settings.agents.max_rounds)

    def curate(self, focus: str | None = None) -> Iterator[str]:
        task = curator_agent.TASK + (f" Focus on: {focus}" if focus else "")
        return self._stream(curator_agent.TOOLS, curator_agent.SYSTEM, task, 12)

    def _stream(self, tools: list[str], system: str, task: str, max_rounds: int) -> Iterator[str]:
        events: queue.Queue = queue.Queue()
        stop = threading.Event()

        def emit(kind: str, data: dict) -> None:
            if stop.is_set():
                raise Stopped()
            events.put((kind, data))

        def work() -> None:
            try:
                llm = self.backend_factory(self.settings)
                emit("start", {"model": llm.name, "local": llm.local})
                limit = self.settings.agents.context_chars if llm.local else CLAUDE_CONTEXT_CHARS

                async def go():
                    server = self.mcp_server_factory()
                    async with McpToolBox(tools, max_chars=limit, server=server) as box:
                        return await run_agent(llm, box, system, task, max_rounds, emit)

                result = asyncio.run(go())
                emit("done", {"rounds": result.rounds, "stop": result.stop})
            except Stopped:
                pass
            except Exception as exc:  # shown to the user; the daemon keeps running
                events.put(("error", {"message": _message(exc)}))
            finally:
                events.put(None)

        threading.Thread(target=work, name="agent-run", daemon=True).start()
        try:
            while (item := events.get()) is not None:
                kind, data = item
                yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            stop.set()


def _message(exc: Exception) -> str:
    # anyio wraps errors from inside the MCP session in exception groups
    while _GROUP and isinstance(exc, _GROUP) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or type(exc).__name__
