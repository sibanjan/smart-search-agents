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
    assert res.text.startswith("No answer: the assistant reached its search limit")
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


# ---- streaming -------------------------------------------------------------
def _ndjson(*chunks):
    return httpx.Response(200, content="\n".join(json.dumps(c) for c in chunks).encode())


def test_ollama_streams_text_and_holds_back_inline_tool_calls():
    replies = iter([
        _ndjson({"message": {"content": "Rent "}}, {"message": {"content": "is 1800."}},
                {"done": True}),
        _ndjson({"message": {"content": '{"name": "get_chunks",'}},
                {"message": {"content": ' "parameters": {"path": "a.md"}}'}}, {"done": True}),
        _ndjson({"message": {"content": "", "tool_calls": [{"function": {
            "name": "search_files", "arguments": {"query": "rent"}}}]}}, {"done": True}),
    ])
    llm = OllamaBackend(transport=httpx.MockTransport(lambda req: next(replies)))
    seen = []
    llm.on_text = seen.append
    llm.start("sys", [ToolSpec("get_chunks", "", {}), ToolSpec("search_files", "", {})])
    assert llm.send_user("q").text == "Rent is 1800." and seen == ["Rent ", "is 1800."]
    inline = llm.send_user("q")
    assert inline.tool_calls[0].name == "get_chunks" and seen == ["Rent ", "is 1800."]
    assert llm.send_user("q").tool_calls[0].args == {"query": "rent"}


def test_ollama_stream_error_is_reported():
    llm = OllamaBackend(transport=httpx.MockTransport(
        lambda req: _ndjson({"error": "model 'llama3.2' not found"})))
    llm.on_text = lambda t: None
    llm.start("sys", [])
    try:
        llm.send_user("q")
    except RuntimeError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected an error")


class FakeStream:
    def __init__(self, response, texts):
        self.response, self.texts = response, texts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([SimpleNamespace(type="text", text=t) for t in self.texts])

    def get_final_message(self):
        return self.response


def test_claude_streams_and_drops_declined_blocks_after_a_fallback():
    final = SimpleNamespace(stop_reason="end_turn", content=[
        _block(type="text", text="Partial "),
        _block(type="thinking", thinking="", signature="declined-model"),
        _block(type="fallback"),
        _block(type="text", text="answer [1]"),
    ])
    calls = []

    def stream(**kw):
        calls.append(kw)
        return FakeStream(final, ["Partial ", "answer [1]"])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    llm = ClaudeBackend(client=fake)
    seen = []
    llm.on_text = seen.append
    llm.start("sys", [])
    turn = llm.send_user("q")
    assert seen == ["Partial ", "answer [1]"] and turn.text == "Partial answer [1]"
    assert calls[0]["fallbacks"] == "default" and calls[0]["max_tokens"] == 64000
    echoed = [b.type for b in llm.messages[-1]["content"]]
    assert echoed == ["text", "text"]  # the declining model's thinking is not sent back


async def test_agent_loop_emits_text_and_resets_before_tools():
    class StreamingLLM(ScriptedLLM):
        on_text = None

        def send_user(self, text):
            self.on_text("Let me search.")
            return super().send_user(text)

        def send_tool_results(self, results, note=None):
            self.on_text("Done [1].")
            return super().send_tool_results(results, note)

    llm = StreamingLLM([Turn("Let me search.", [ToolCall("1", "search_files", {})], "tool_use"),
                        Turn("Done [1].")])
    events = []
    await run_agent(llm, FakeTools({"search_files": []}), "sys", "q",
                    on_event=lambda k, d: events.append(k))
    assert events == ["text", "text_reset", "tool", "text", "answer"]


def test_ollama_tool_call_written_after_prose():
    llm = OllamaBackend(transport=httpx.MockTransport(lambda req: None))
    llm.start("sys", [ToolSpec("search_files", "", {})])
    said = ('I found nothing. I\'ll try again.\n\n{"name": "search_files", "parameters": '
            '{"query": "faithfulness", "path": "", "since": null}}')
    assert llm._inline_call(said) == ("search_files", {"query": "faithfulness", "path": "", "since": None})
    assert llm._inline_call("```json\n" + said[said.index("{"):] + "\n```")[0] == "search_files"
    assert llm._inline_call(said + " Then I will summarise.") is None  # prose after it: not a call
    assert llm._inline_call('Use {"name": "other_tool"} like this') is None
    assert llm._inline_call("Set {a} to {b}.") is None
