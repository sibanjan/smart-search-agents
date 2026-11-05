"""The agents' only way to the index: MCP tools.

`McpToolBox` connects to this project's MCP server (started over stdio by default) and
exposes an allow-listed subset of its tools. Any MCP server with the same tool names works.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Protocol

from .llm import ToolCall, ToolResult, ToolSpec

PASS_ENV = ("SMARTSEARCH_CONFIG", "SMARTSEARCH_HOME", "SMARTSEARCH_TOKEN")


class ToolBox(Protocol):
    specs: list[ToolSpec]

    async def call(self, call: ToolCall) -> ToolResult: ...


class McpToolBox:
    def __init__(self, allow: list[str], max_chars: int = 8000, server: Any = None):
        """server: an MCPServer instance (in-process), StdioServerParameters, or a URL.
        Default: spawn `python -m smart_search_agents mcp` over stdio."""
        from mcp.client.stdio import StdioServerParameters, get_default_environment

        self.allow = allow
        self.max_chars = max_chars
        # The stdio child gets a minimal environment; pass on which config/index to use.
        env = get_default_environment()
        env.update({k: v for k, v in os.environ.items() if k in PASS_ENV})
        self.server = server or StdioServerParameters(
            command=sys.executable, args=["-m", "smart_search_agents", "mcp"], env=env
        )
        self.specs: list[ToolSpec] = []
        self._client = None

    async def __aenter__(self) -> McpToolBox:
        from mcp import Client

        self._client = Client(self.server)
        await self._client.__aenter__()
        listed = (await self._client.list_tools()).tools
        missing = set(self.allow) - {t.name for t in listed}
        if missing:
            await self.__aexit__(None, None, None)
            raise RuntimeError(f"MCP server lacks tools: {', '.join(sorted(missing))}")
        self.specs = [ToolSpec(t.name, t.description or "", t.input_schema)
                      for t in listed if t.name in self.allow]
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc)
            self._client = None

    async def call(self, call: ToolCall) -> ToolResult:
        if call.name not in self.allow:
            return ToolResult(call, f"Tool {call.name!r} is not available to this agent.", True)
        try:
            res = await self._client.call_tool(call.name, call.args)
        except Exception as exc:
            return ToolResult(call, f"Tool error: {exc}", True)
        if res.structured_content is not None:
            text = json.dumps(res.structured_content, ensure_ascii=False)
        else:
            text = "\n".join(getattr(c, "text", "") for c in res.content)
        return ToolResult(call, truncate(text, self.max_chars), bool(res.is_error))


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} characters]"
