"""MCP tools end to end: MCP client -> MCP server -> daemon HTTP API -> index (in process)."""

import json

import pytest

from smart_search_agents.agents.ask import TOOLS as ASK_TOOLS
from smart_search_agents.agents.curator import TOOLS as CURATOR_TOOLS
from smart_search_agents.agents.llm import ToolCall
from smart_search_agents.agents.tools import McpToolBox
from smart_search_agents.client import DaemonClient
from smart_search_agents.mcp_server import build_server

EXPECTED = {
    "search_files", "get_chunks", "list_image_categories", "list_images", "image_details",
    "preview_category", "index_status", "list_jobs", "query_db", "run_job", "cancel_job",
    "suggest_category",
}


@pytest.fixture
def server(client):
    dc = DaemonClient.__new__(DaemonClient)  # skip settings lookup; reuse the test client
    dc.base_url, dc.http = "http://testserver", client
    return build_server(dc)


async def test_tool_list_and_annotations(server):
    from mcp import Client

    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == EXPECTED
    assert tools["search_files"].annotations.read_only_hint is True
    assert tools["run_job"].annotations.read_only_hint is False
    assert "open_path" not in tools  # agents never open files on screen


async def test_ask_tools_through_mcp(server):
    async with McpToolBox(ASK_TOOLS, server=server) as box:
        assert {s.name for s in box.specs} == set(ASK_TOOLS)
        hits = await box.call(ToolCall("1", "search_files", {"query": "faithfulness"}))
        assert not hits.is_error and "rag_eval.md" in hits.content
        blocked = await box.call(ToolCall("2", "run_job", {"kind": "index"}))
        assert blocked.is_error  # not on the Ask agent's allow-list


async def test_curator_tools_through_mcp(server, client):
    async with McpToolBox(CURATOR_TOOLS, server=server) as box:
        cats = await box.call(ToolCall("1", "list_image_categories", {}))
        assert '"reds": 3' in cats.content
        prev = await box.call(ToolCall("2", "preview_category", {"name": "crimson", "prompt": "deep red"}))
        assert "would_join" in prev.content
        sug = await box.call(ToolCall("3", "suggest_category",
                                      {"name": "greens", "prompt": "something green"}))
        assert "proposed" in sug.content
    assert client.get("/api/suggestions").json()["suggestions"][0]["name"] == "greens"


async def test_query_db_tool_rejects_writes(server):
    async with McpToolBox(["query_db"], server=server) as box:
        bad = await box.call(ToolCall("1", "query_db", {"sql": "DELETE FROM files"}))
        assert bad.is_error and "SELECT" in bad.content


def test_http_transport_requires_token(server):
    from starlette.testclient import TestClient

    from smart_search_agents.mcp_server import http_app

    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"}}}
    headers = {"accept": "application/json, text/event-stream"}
    with TestClient(http_app(server, "s3cret"), base_url="http://127.0.0.1:8766") as http:
        assert http.post("/mcp", json=init, headers=headers).status_code == 401
        wrong = http.post("/mcp", json=init, headers={**headers, "authorization": "Bearer nope"})
        assert wrong.status_code == 401
        ok = http.post("/mcp", json=init, headers={**headers, "authorization": "Bearer s3cret"})
        assert ok.status_code == 200 and "smartsearch" in ok.text


async def test_search_results_are_easy_for_small_models(server):
    async with McpToolBox(["search_files"], server=server) as box:
        hits = json.loads((await box.call(ToolCall("1", "search_files", {"query": "faithfulness"}))).content)
        first = hits["results"][0]
        assert first["rank"] == 1 and "score" not in first  # no raw score to mistake for data
        assert len(first["modified"]) == 10 and "\r" not in first["snippet"]
        empty = json.loads((await box.call(ToolCall("2", "search_files",
                                                    {"query": "faithfulness", "ext": ["pdf"]}))).content)
        assert empty["results"] == [] and "without them" in empty["note"]


@pytest.mark.parametrize("ext", ["md", ".md", "md, txt", "['md']", ["MD"]])
async def test_search_files_accepts_ext_the_way_small_models_write_it(server, ext):
    async with McpToolBox(["search_files"], server=server) as box:
        res = await box.call(ToolCall("1", "search_files", {"query": "faithfulness", "ext": ext}))
        assert not res.is_error, res.content
        assert json.loads(res.content)["results"]
