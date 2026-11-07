"""The UI's agent routes: server-sent events from a real agent run, whose tool calls go
through an in-process MCP server back into the daemon's HTTP API."""

import json

import pytest
from fastapi.testclient import TestClient

from smart_search_agents.agents.llm import ToolCall, Turn
from smart_search_agents.client import DaemonClient
from smart_search_agents.daemon.agent_runner import AgentRunner
from smart_search_agents.daemon.curation import Curation
from smart_search_agents.daemon.service import create_app
from smart_search_agents.mcp_server import build_server
from smart_search_agents.settings import Settings

TOKEN = "test-token"  # same as conftest


class ScriptedStreamingLLM:
    name, local = "scripted:test", True
    on_text = None

    def __init__(self, turns, fail: Exception | None = None):
        self.turns, self.fail, self.results = list(turns), fail, []

    def start(self, system, tools):
        self.tool_names = [t.name for t in tools]

    def _next(self):
        if self.fail:
            raise self.fail
        turn = self.turns.pop(0)
        if turn.text and self.on_text:
            for word in turn.text.split(" "):
                self.on_text(word + " ")
        return turn

    def send_user(self, text):
        return self._next()

    def send_tool_results(self, results, note=None):
        self.results += [r.content for r in results]
        return self._next()


def events(response):
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def run(indexed, jobs):
    """Build an app whose agents use `llm`; returns (TestClient, llm)."""
    made = []

    def make(llm):
        settings = Settings(core=indexed.cfg, config_file=None)
        holder = {}
        runner = AgentRunner(settings, lambda: build_server(holder["dc"]), lambda s: llm)
        app = create_app(indexed, jobs, Curation(indexed), settings, TOKEN,
                         allowed_hosts={"testserver"}, agents=runner)
        client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
        client.__enter__()
        made.append(client)
        dc = DaemonClient.__new__(DaemonClient)
        dc.base_url, dc.http = "http://testserver", client
        holder["dc"] = dc
        return client

    yield make
    for c in made:
        c.__exit__(None, None, None)


def test_ask_streams_tool_steps_and_the_answer(run):
    llm = ScriptedStreamingLLM([
        Turn("", [ToolCall("1", "search_files", {"query": "faithfulness"})], "tool_use"),
        Turn("Use a held-out question set [1]."),
    ])
    client = run(llm)
    r = client.get("/api/agent/ask", params={"q": "how do I evaluate RAG?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    got = events(r)
    kinds = [k for k, _ in got]
    assert kinds[0] == "start" and got[0][1] == {"model": "scripted:test", "local": True}
    assert kinds[1] == "tool" and got[1][1]["name"] == "search_files"
    assert "text" in kinds and kinds[-2:] == ["answer", "done"]
    assert got[-2][1]["text"] == "Use a held-out question set [1]."
    # the tool really ran against the index, through MCP and the daemon API
    assert "rag_eval.md" in llm.results[0]
    assert llm.tool_names == ["search_files", "get_chunks"]  # the Ask agent's allow-list only


def test_ask_reports_backend_errors_as_an_event(run):
    llm = ScriptedStreamingLLM([], fail=RuntimeError("Ollama request failed. Is it running?"))
    got = events(run(llm).get("/api/agent/ask", params={"q": "anything"}))
    assert got[-1] == ("error", {"message": "Ollama request failed. Is it running?"})


def test_ask_needs_a_question_and_the_token(run):
    client = run(ScriptedStreamingLLM([]))
    assert client.get("/api/agent/ask", params={"q": "  "}).status_code == 400
    assert TestClient(client.app).get("/api/agent/ask", params={"q": "x"}).status_code == 401


def test_curate_records_a_suggestion(run):
    llm = ScriptedStreamingLLM([
        Turn("", [ToolCall("1", "list_image_categories", {})], "tool_use"),
        Turn("", [ToolCall("2", "suggest_category",
                           {"name": "greens", "prompt": "something green"})], "tool_use"),
        Turn("Suggested greens."),
    ])
    client = run(llm)
    got = events(client.get("/api/agent/curate"))
    assert [d["name"] for k, d in got if k == "tool"] == ["list_image_categories", "suggest_category"]
    assert got[-1][0] == "done"
    names = [s["name"] for s in client.get("/api/suggestions").json()["suggestions"]]
    assert names == ["greens"]
