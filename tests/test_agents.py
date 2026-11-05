import json
from types import SimpleNamespace

import httpx

from smart_search_agents.agents.ask import ask
from smart_search_agents.agents.base import FINISH_NOTE, run_agent
from smart_search_agents.agents.llm import (
    ClaudeBackend,
    OllamaBackend,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
)
from smart_search_agents.agents.tools import truncate


class ScriptedLLM:
    """Plays back a list of turns and records what the agent sent."""

    name, local = "scripted", True

    def __init__(self, turns):
        self.turns = list(turns)
        self.sent = []

    def start(self, system, tools):
        self.system, self.tools = system, tools

    def send_user(self, text):
        self.sent.append(("user", text))
        return self.turns.pop(0)

    def send_tool_results(self, results, note=None):
        self.sent.append(("tools", [r.content for r in results], note))
        return self.turns.pop(0)


class FakeTools:
    def __init__(self, answers):
        self.answers = answers
        self.specs = [ToolSpec(n, "", {"type": "object"}) for n in answers]

    async def call(self, call):
        return ToolResult(call, json.dumps(self.answers[call.name]))


async def test_ask_agent_searches_reads_then_answers():
    llm = ScriptedLLM([
        Turn("", [ToolCall("1", "search_files", {"query": "rag eval"})], "tool_use"),
        Turn("", [ToolCall("2", "get_chunks", {"path": "notes/rag_eval.md"})], "tool_use"),
        Turn("Use a held-out set [1].\n\nSources\n[1] notes/rag_eval.md"),
    ])
    tools = FakeTools({"search_files": [{"path": "notes/rag_eval.md"}],
                       "get_chunks": {"chunks": [{"text": "held-out question set"}]}})
    res = await ask("how do I evaluate RAG?", llm, tools)
    assert res.rounds == 2 and res.stop == "end"
    assert [t["tool"] for t in res.tool_log] == ["search_files", "get_chunks"]
    assert "[1]" in res.text and "not instructions" in llm.system  # injection warning present


async def test_round_limit_forces_an_answer():
    loop = Turn("", [ToolCall("x", "search_files", {"query": "again"})], "tool_use")
    llm = ScriptedLLM([loop, loop, Turn("", [ToolCall("y", "search_files", {})], "tool_use")])
    res = await run_agent(llm, FakeTools({"search_files": []}), "sys", "q", max_rounds=2)
    assert res.rounds == 2 and res.stop == "max_rounds"
    assert llm.sent[-1][2] == FINISH_NOTE


def test_ollama_small_model_quirks():
    """llama3.2 puts "null" in optional fields and sometimes writes the call as text."""
    replies = iter([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "search_files", "arguments": {"query": "rrf", "path": "null", "limit": "None"}}}]},
        {"role": "assistant", "content": '{"name": "get_chunks", "parameters": {"path": "a.py"}}'},
        {"role": "assistant", "content": '{"not": "a tool"}'},
    ])
    llm = OllamaBackend(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"message": next(replies)})))
    llm.start("sys", [ToolSpec("search_files", "", {}), ToolSpec("get_chunks", "", {})])
    assert llm.send_user("q").tool_calls[0].args == {"query": "rrf"}
    inline = llm.send_user("q").tool_calls[0]
    assert (inline.name, inline.args) == ("get_chunks", {"path": "a.py"})
    assert llm.send_user("q").tool_calls == []


def test_truncate():
    assert truncate("abc", 5) == "abc"
    assert truncate("x" * 10, 4).startswith("xxxx\n…[truncated 6")


def test_ollama_backend_parses_tool_calls():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            msg = {"role": "assistant", "content": "",
                   "tool_calls": [{"function": {"name": "search_files",
                                                "arguments": '{"query": "budget"}'}}]}
        else:
            msg = {"role": "assistant", "content": "Rent is 1800 [1]."}
        return httpx.Response(200, json={"message": msg})

    llm = OllamaBackend("llama3.2", transport=httpx.MockTransport(handler))
    llm.start("sys", [ToolSpec("search_files", "search", {"type": "object"})])
    t1 = llm.send_user("what is my rent?")
    assert t1.tool_calls[0].name == "search_files" and t1.tool_calls[0].args == {"query": "budget"}
    t2 = llm.send_tool_results([ToolResult(t1.tool_calls[0], "[...]")])
    assert t2.text == "Rent is 1800 [1]." and not t2.tool_calls
    assert seen[1]["messages"][-1] == {"role": "tool", "tool_name": "search_files", "content": "[...]"}
    assert seen[0]["tools"][0]["function"]["name"] == "search_files"


class FakeAnthropic:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def _block(**kw):
    return SimpleNamespace(**kw)


def test_claude_backend_round_trip():
    tool_use = SimpleNamespace(stop_reason="tool_use", content=[
        _block(type="thinking", thinking="", signature="s"),
        _block(type="tool_use", id="tu_1", name="search_files", input={"query": "rent"}),
    ])
    final = SimpleNamespace(stop_reason="end_turn", content=[_block(type="text", text="1800 [1]")])
    fake = FakeAnthropic([tool_use, final])
    llm = ClaudeBackend(client=fake)
    llm.start("sys", [ToolSpec("search_files", "search", {"type": "object"})])
    t1 = llm.send_user("rent?")
    assert t1.tool_calls[0].id == "tu_1"
    t2 = llm.send_tool_results([ToolResult(t1.tool_calls[0], "[...]")], note="finish")
    assert t2.text == "1800 [1]"
    req = fake.calls[1]
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert "thinking" not in req and req["output_config"] == {"effort": "medium"}
    assert req["messages"][1]["content"] is tool_use.content  # thinking blocks sent back whole
    results = req["messages"][2]["content"]
    assert results[0]["type"] == "tool_result" and results[0]["tool_use_id"] == "tu_1"
    assert results[1] == {"type": "text", "text": "finish"}


def test_claude_refusal_and_truncation_stop_tools():
    refusal = SimpleNamespace(stop_reason="refusal", content=[])
    cut = SimpleNamespace(stop_reason="max_tokens", content=[
        _block(type="tool_use", id="t", name="search_files", input={})])
    llm = ClaudeBackend(client=FakeAnthropic([refusal, cut]))
    llm.start("sys", [])
    assert llm.send_user("x").stop == "refusal"
    t = llm.send_user("y")
    assert t.stop == "max_tokens" and t.tool_calls == []
