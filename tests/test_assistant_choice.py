"""Choosing an assistant: the options with Ollama's state, downloading a local model, and
turning the assistant off."""

import json

import httpx
import pytest

from smart_search_agents.agents.catalog import ASSISTANTS, has_model
from smart_search_agents.agents.llm import make_backend
from smart_search_agents.daemon import service
from smart_search_agents.settings import Settings


class FakeOllama:
    """Answers /api/tags and streams /api/pull like Ollama, or acts as if it is not running."""

    def __init__(self, models=("llama3.2:latest", "llama3.2:3b", "mistral:7b"), running=True):
        self.models, self.running, self.pulled = list(models), running, []

    def handler(self, request):
        if not self.running:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/api/tags":
            digest = {"llama3.2:3b": "llama3.2:latest"}  # one model under two names
            return httpx.Response(200, json={"models": [
                {"name": m, "digest": "sha-" + digest.get(m, m)} for m in self.models]})
        if request.url.path == "/api/pull":
            model = json.loads(request.content)["model"]
            self.pulled.append(model)
            lines = [{"status": "pulling manifest"},
                     {"status": "downloading", "total": 100, "completed": 40},
                     {"status": "downloading", "total": 100, "completed": 100},
                     {"status": "success"}]
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        return httpx.Response(404)


@pytest.fixture
def ollama(monkeypatch):
    fake = FakeOllama()
    client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    monkeypatch.setattr(service.httpx, "get", client.get)
    monkeypatch.setattr(service.httpx, "stream", client.stream)
    return fake


def test_options_describe_each_choice_honestly():
    by_id = {a["id"]: a for a in ASSISTANTS}
    assert set(by_id) >= {"off", "ollama:llama3.2", "claude:claude-opus-5-5"}
    for a in ASSISTANTS:
        assert a["title"] and a["tagline"] and a["details"] and a["privacy"]
        cloud = a["backend"] == "claude"
        assert ("sent to Anthropic" in a["privacy"]) == cloud
        if a["backend"] == "ollama":
            assert a["download_gb"] > 0
            assert a.get("tested") or "Not tested" in " ".join(a["details"])


def test_has_model_matches_the_latest_tag():
    assert has_model(["llama3.2:latest"], "llama3.2")
    assert not has_model(["llama3.2:latest"], "llama3.1:8b")
    assert has_model(["qwen3:4b"], "qwen3:4b")


def test_options_show_what_is_downloaded(client, ollama):
    data = client.get("/api/assistant/options").json()
    assert data["ollama"]["running"] and "mistral:7b" in data["ollama"]["models"]
    assert data["ollama"]["others"] == ["mistral:7b"]  # llama3.2:3b is Llama 3.2 again
    got = {o["id"]: o.get("downloaded") for o in data["options"] if o["backend"] == "ollama"}
    assert got["ollama:llama3.2"] is True and got["ollama:llama3.1:8b"] is False
    assert data["current"]["backend"] == "ollama"


def test_options_when_ollama_is_missing(client, ollama):
    ollama.running = False
    data = client.get("/api/assistant/options").json()
    assert data["ollama"] == {"running": False, "models": [], "others": [],
                              "download_url": "https://ollama.com/download"}


def test_download_streams_progress_and_only_listed_models(client, ollama):
    r = client.post("/api/assistant/pull", json={"model": "llama3.1:8b"})
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert lines[-1] == {"status": "success"} and ollama.pulled == ["llama3.1:8b"]
    assert client.post("/api/assistant/pull", json={"model": "anything/else"}).status_code == 400
    ollama.running = False
    err = client.post("/api/assistant/pull", json={"model": "qwen3:4b"}).text
    assert "Could not reach Ollama" in json.loads(err)["error"]


def test_choosing_saves_and_off_stops_the_agents(client, indexed):
    r = client.put("/api/settings", json={"agents.backend": "claude",
                                          "agents.claude_model": "claude-sonnet-5-5"})
    assert r.status_code == 200
    data = client.get("/api/assistant/options").json()["current"]
    assert data["backend"] == "claude" and data["claude_model"] == "claude-sonnet-5-5"
    client.put("/api/settings", json={"agents.backend": "off"})
    with pytest.raises(RuntimeError, match="turned off"):
        make_backend(Settings(core=indexed.cfg, config_file=None, agents=type(
            "A", (), {"backend": "off"})()))


def test_setup_flag_is_saved(client):
    assert client.get("/api/settings").json()["values"]["daemon.setup_complete"] is False
    client.put("/api/settings", json={"daemon.setup_complete": True})
    assert client.get("/api/settings").json()["values"]["daemon.setup_complete"] is True
