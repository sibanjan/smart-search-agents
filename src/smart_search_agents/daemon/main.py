"""Start the daemon: engine, job queue, scheduler, watcher and HTTP API in one process."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..settings import Settings, load_settings, read_or_create_token
from .curation import Curation
from .db import make_engine
from .jobs import JobQueue
from .scheduler import Scheduler
from .service import create_app
from .watch import Watcher

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self, settings: Settings, watch: bool = True, schedule: bool = True):
        self.settings = settings
        self.engine = make_engine(settings.core)
        self.jobs = JobQueue(self.engine)
        self.curation = Curation(self.engine)
        self.scheduler = Scheduler(self.jobs, settings.schedule) if schedule else None
        self.watcher = Watcher(self.jobs, self.engine.indexer) if watch else None
        self.token = read_or_create_token(settings.token_path)
        self.app = create_app(
            self.engine, self.jobs, self.curation, settings, self.token, self.scheduler,
            watcher=self.watcher,
        )

    def start_background(self) -> None:
        log.info("config: %s", self.settings.config_file or "(defaults)")
        _ = self.engine.embedder  # text model loads once; CLIP loads on the first image job
        self.jobs.start()
        if self.scheduler:
            self.scheduler.start()
            queued = self.scheduler.catch_up()
            if queued:
                log.info("catch-up queued: %s", ", ".join(queued))
        if self.watcher:
            self.watcher.start()

    def stop(self) -> None:
        if self.watcher:
            self.watcher.stop()
        if self.scheduler:
            self.scheduler.stop()
        self.jobs.stop()


def run(config: str | None = None, watch: bool = True, schedule: bool = True) -> None:
    import uvicorn

    settings = load_settings(config)
    pid = _acquire_pid(settings.pid_path)
    daemon = Daemon(settings, watch=watch, schedule=schedule)
    try:
        daemon.start_background()
        log.info("SmartSearch daemon on %s (pid %s)", settings.base_url, pid)
        uvicorn.run(
            daemon.app, host=settings.daemon.host, port=settings.daemon.port, log_level="warning"
        )
    finally:
        daemon.stop()
        settings.pid_path.unlink(missing_ok=True)


def _acquire_pid(path: Path) -> int:
    """One daemon per index: refuse to start if another live process holds the pid file."""
    if path.is_file():
        try:
            other = int(path.read_text().strip())
        except ValueError:
            other = 0
        if other and other != os.getpid() and _alive(other):
            raise SystemExit(f"Daemon already running (pid {other}). Stop it with: ssa daemon stop")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))
    return os.getpid()


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
