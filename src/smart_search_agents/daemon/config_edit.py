"""Read and change settings from the UI.

Each setting has a check and an effect: applied at once, or queued work (a re-index, an image
scan, a re-classify), or saved for the next start. A save is all-or-nothing: every value is
checked before anything changes. Only the changed keys are written, so anything else in
config.yaml stays, though YAML comments do not survive (the first save keeps a .bak copy).

The daemon's host and index_dir are deliberately not editable: the first keeps the service on
this machine, the second would need the database moved.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import yaml
from apscheduler.triggers.cron import CronTrigger

from ..settings import Settings

INDEX, SCAN, CLASSIFY = "index", "image_scan", "image_classify"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
SCHEDULED = (INDEX, SCAN)


class InvalidSettings(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


# ---- checks: return the cleaned value or raise ValueError --------------------
def _text(v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError("must not be empty")
    return v.strip()


def _texts(v: Any) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise ValueError("must be a list of text values")
    return list(dict.fromkeys(s.strip() for s in v if s.strip()))


def _folders(v: Any) -> list[str]:
    folders = _texts(v)
    if not folders:
        raise ValueError("add at least one folder")
    return folders


def _extensions(v: Any) -> list[str]:
    exts = [e.lower() if e.startswith(".") else "." + e.lower() for e in _texts(v)]
    if not exts:
        raise ValueError("add at least one file type")
    return list(dict.fromkeys(exts))


def _flag(v: Any) -> bool:
    if not isinstance(v, bool):
        raise ValueError("must be true or false")
    return v


def _number(low: float, high: float, kind: type = float) -> Callable[[Any], Any]:
    def check(v: Any):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and v != int(v)):
            raise ValueError(f"must be a{' whole' if kind is int else ''} number")
        if not low <= v <= high:
            raise ValueError(f"must be between {low} and {high}")
        return kind(v)

    return check


def _choice(*options: str) -> Callable[[Any], str]:
    def check(v: Any) -> str:
        if v not in options:
            raise ValueError(f"must be one of: {', '.join(options)}")
        return v

    return check


def _cron(v: Any) -> str:
    """Five-field cron text, or empty for off."""
    if v is None or v == "":
        return ""
    try:
        CronTrigger.from_crontab(_text(v))
    except ValueError as exc:
        raise ValueError(f"not a valid schedule: {exc}") from exc
    return v.strip()


def _url(v: Any) -> str:
    url = urlparse(_text(v))
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError("must look like http://localhost:11434")
    return v.strip().rstrip("/")


def _categories(v: Any) -> dict[str, str]:
    if not isinstance(v, dict) or not v:
        raise ValueError("keep at least one category")
    out = {}
    for name, prompt in v.items():
        if not isinstance(prompt, str) or not str(name).strip() or not prompt.strip():
            raise ValueError("every category needs a name and a description")
        out[str(name).strip()] = prompt.strip()
    return out


@dataclass(frozen=True)
class Field:
    check: Callable[[Any], Any]
    jobs: tuple[str, ...] = ()  # work to queue when the value changes
    restart: bool = False  # takes effect on the next start
    warning: str = ""


MODEL_WARNING = "Changing this model clears that index, which is rebuilt after the restart."
CHUNK_WARNING = "This applies to new and changed files. Rebuild the index to apply it to all files."

FIELDS: dict[str, Field] = {
    # files
    "roots": Field(_folders, (INDEX, SCAN)),
    "extensions": Field(_extensions, (INDEX,)),
    "exclude_dirs": Field(_texts, (INDEX, SCAN)),
    "skip_hidden": Field(_flag, (INDEX, SCAN)),
    "max_file_size_mb": Field(_number(0.1, 2048), (INDEX,)),
    "chunk_size": Field(_number(100, 20000, int), warning=CHUNK_WARNING),
    "chunk_overlap": Field(_number(0, 5000, int), warning=CHUNK_WARNING),
    # search
    "default_limit": Field(_number(1, 100, int)),
    "semantic_weight": Field(_number(0, 10)),
    "keyword_weight": Field(_number(0, 10)),
    "filename_weight": Field(_number(0, 10)),
    # text model
    "embedder": Field(_choice("auto", "sentence-transformers", "hash"), restart=True,
                      warning=MODEL_WARNING),
    "model_name": Field(_text, restart=True, warning=MODEL_WARNING),
    "batch_size": Field(_number(1, 512, int), restart=True),
    # images
    "image_roots": Field(_texts, (SCAN,)),
    "image_extensions": Field(_extensions, (SCAN,)),
    "image_min_size_kb": Field(_number(0, 102400), (SCAN,)),
    "image_model": Field(_text, restart=True, warning=MODEL_WARNING),
    "image_categories": Field(_categories, (CLASSIFY,)),
    "image_min_confidence": Field(_number(0, 1), (CLASSIFY,)),
    "image_label_threshold": Field(_number(0, 1), (CLASSIFY,)),
    "image_clusters": Field(_number(0, 100, int), (CLASSIFY,)),
    "image_min_cluster_size": Field(_number(2, 1000, int), (CLASSIFY,)),
    # AI assistant
    "ollama_url": Field(_url),
    "ollama_model": Field(_text),
    "agents.backend": Field(_choice("ollama", "claude", "off")),
    "agents.model": Field(_text),
    "agents.claude_model": Field(_text),
    "agents.claude_effort": Field(_choice("low", "medium", "high", "xhigh", "max")),
    "agents.max_rounds": Field(_number(1, 30, int)),
    "agents.context_chars": Field(_number(500, 200000, int)),
    # schedule and service
    "schedule.index": Field(_cron),
    "schedule.image_scan": Field(_cron),
    "daemon.port": Field(_number(1024, 65535, int), restart=True),
    "daemon.setup_complete": Field(_flag),
}


class SettingsEditor:
    def __init__(self, settings: Settings, engine, jobs, curation, scheduler=None, watcher=None):
        self.settings = settings
        self.cfg = engine.cfg
        self.jobs = jobs
        self.curation = curation
        self.scheduler = scheduler
        self.watcher = watcher
        self.pending: dict[str, Any] = {}  # saved, waiting for a restart
        self._lock = threading.Lock()

    # ---- reading ---------------------------------------------------------
    def current(self, key: str) -> Any:
        if key in self.pending:
            return self.pending[key]
        section, _, name = key.rpartition(".")
        if section == "agents":
            return getattr(self.settings.agents, name)
        if section == "daemon":
            return getattr(self.settings.daemon, name)
        if section == "schedule":
            return self.settings.schedule.get(name, "")
        return getattr(self.cfg, name)

    def read(self) -> dict:
        return {
            "values": {k: self.current(k) for k in FIELDS},
            "config_file": str(self.settings.config_file or self.target_file),
            "restart_required": sorted(self.pending),
        }

    @property
    def target_file(self) -> Path:
        return self.settings.config_file or self.settings.home / "config.yaml"

    # ---- changing --------------------------------------------------------
    def update(self, changes: dict[str, Any]) -> dict:
        with self._lock:
            clean = self._validate(changes)
            changed = {k: v for k, v in clean.items() if v != self.current(k)}
            warnings = list(dict.fromkeys(FIELDS[k].warning for k in changed if FIELDS[k].warning))
            warnings += self._privacy_warnings(changed)
            if changed:
                self._save(changed)
                self._apply(changed)
            return {
                "changed": sorted(changed),
                "jobs": self._queue(changed),
                "restart_required": sorted(self.pending),
                "warnings": warnings,
            }

    def _validate(self, changes: dict[str, Any]) -> dict[str, Any]:
        clean, errors = {}, {}
        for key, value in changes.items():
            field = FIELDS.get(key)
            if field is None:
                errors[key] = "this setting cannot be changed here"
                continue
            try:
                clean[key] = field.check(value)
            except ValueError as exc:
                errors[key] = str(exc)
        if "chunk_overlap" not in errors and "chunk_size" not in errors:
            size = clean.get("chunk_size", self.current("chunk_size"))
            overlap = clean.get("chunk_overlap", self.current("chunk_overlap"))
            if overlap >= size:
                errors["chunk_overlap"] = "must be smaller than the chunk size"
        if errors:
            raise InvalidSettings(errors)
        return clean

    def _privacy_warnings(self, changed: dict[str, Any]) -> list[str]:
        host = urlparse(changed.get("ollama_url", "")).hostname
        if host and host not in LOCAL_HOSTS:
            return [f"Ollama is now on another machine ({host}), so file excerpts will be sent there."]
        return []

    def _save(self, changed: dict[str, Any]) -> None:
        path = self.target_file
        raw = {}
        if path.is_file():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            backup = path.with_name(path.name + ".bak")
            if not backup.exists():
                shutil.copy2(path, backup)
        for key, value in changed.items():
            section, _, name = key.rpartition(".")
            if section == "schedule":  # stored whole: a partial block would drop the other job
                raw["schedule"] = {k: self.current(f"schedule.{k}") for k in SCHEDULED}
                raw["schedule"][name] = value
            elif section:
                raw.setdefault(section, {})[name] = value
            else:
                raw[name] = value
        if "schedule.index" in changed and "schedule.image_scan" in changed:
            raw["schedule"] = {k: changed[f"schedule.{k}"] for k in SCHEDULED}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        os.replace(tmp, path)
        self.settings.config_file = path

    def _apply(self, changed: dict[str, Any]) -> None:
        removed = set()
        if "image_categories" in changed:
            removed = set(self.cfg.image_categories) - set(changed["image_categories"])
        new_schedule = False
        for key, value in changed.items():
            section, _, name = key.rpartition(".")
            if FIELDS[key].restart:
                self.pending[key] = value
            elif section in ("agents", "daemon"):
                setattr(getattr(self.settings, section), name, value)
            elif section == "schedule":
                new_schedule = True
                if value:
                    self.settings.schedule[name] = value
                else:
                    self.settings.schedule.pop(name, None)
            else:
                setattr(self.cfg, name, value)
        if removed:
            self.curation.retire(removed)  # or an accepted suggestion returns on the next start
        if new_schedule and self.scheduler:
            self.scheduler.reschedule(self.settings.schedule)
        if "roots" in changed and self.watcher:
            self.watcher.set_roots(self.cfg.root_paths)

    def _queue(self, changed: dict[str, Any]) -> list[dict]:
        kinds = {j for k in changed for j in FIELDS[k].jobs}
        if "roots" in changed and self.cfg.image_roots:
            kinds.discard(SCAN)  # images have their own folders
        if SCAN in kinds:
            kinds.discard(CLASSIFY)  # a scan re-classifies when it finishes
        return [self.jobs.submit(kind, {}, "api") for kind in sorted(kinds)]
