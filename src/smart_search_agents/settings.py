"""Settings for the daemon, MCP server and agents.

They live in the same config.yaml SmartSearch already reads, under three extra keys
(SmartSearch ignores keys it does not know):

    daemon:   {host: 127.0.0.1, port: 8765}
    schedule: {index: "0 2 * * *", image_scan: "0 3 * * SUN"}
    agents:   {backend: ollama, model: llama3.2, claude_model: claude-opus-5-5}
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from smartsearch.config import DEFAULT_HOME, Config, config_path_candidates, load_config


@dataclass
class DaemonSettings:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class AgentSettings:
    backend: str = "ollama"  # "ollama" (local) | "claude" (cloud: excerpts leave the laptop)
    model: str = "llama3.2"  # Ollama model
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"  # low | medium | high | xhigh | max
    max_rounds: int = 6  # tool-calling rounds per question
    context_chars: int = 6000  # passage budget for the local model


@dataclass
class Settings:
    core: Config
    config_file: Path | None
    daemon: DaemonSettings = field(default_factory=DaemonSettings)
    schedule: dict[str, str] = field(
        default_factory=lambda: {"index": "0 2 * * *", "image_scan": "0 3 * * SUN"}
    )
    agents: AgentSettings = field(default_factory=AgentSettings)

    @property
    def home(self) -> Path:
        return Path(self.core.index_dir).expanduser()

    @property
    def token_path(self) -> Path:
        return self.home / "token"

    @property
    def pid_path(self) -> Path:
        return self.home / "daemon.pid"

    @property
    def base_url(self) -> str:
        return f"http://{self.daemon.host}:{self.daemon.port}"


def active_config_file(path: str | Path | None = None) -> Path | None:
    candidates = [Path(path).expanduser()] if path else config_path_candidates()
    return next((p for p in candidates if p.is_file()), None)


def load_settings(path: str | Path | None = None) -> Settings:
    cfg_file = active_config_file(path)
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file else None
    raw = raw or {}
    s = Settings(core=load_config(cfg_file) if cfg_file else Config(), config_file=cfg_file)
    s.daemon = _merge(DaemonSettings(), raw.get("daemon"))
    s.agents = _merge(AgentSettings(), raw.get("agents"))
    if isinstance(raw.get("schedule"), dict):
        s.schedule = {k: str(v) for k, v in raw["schedule"].items() if v}
    return s


def _merge(obj, data):
    for k, v in (data or {}).items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj


def read_or_create_token(path: Path | None = None) -> str:
    """Shared secret between the daemon and its local clients (MCP server, CLI)."""
    path = path or (DEFAULT_HOME / "token")
    if env := os.environ.get("SMARTSEARCH_TOKEN"):
        return env
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    return token
