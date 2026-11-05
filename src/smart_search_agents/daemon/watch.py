"""File watcher that queues `index_paths` jobs instead of writing to the index itself."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from smartsearch.indexer import Indexer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .jobs import JobQueue

log = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    """Collects changed paths and flushes them as one job every `delay` seconds,
    so an editor's burst of saves becomes a single re-index."""

    def __init__(self, jobs: JobQueue, indexer: Indexer, delay: float = 2.0):
        self.jobs, self.indexer, self.delay = jobs, indexer, delay
        self.pending: set[str] = set()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, name="watch-flush", daemon=True).start()

    def _add(self, path: str) -> None:
        if not self.indexer.is_candidate(Path(path)):  # judged by name: works for deletes too
            return
        with self.lock:
            self.pending.add(path)

    def on_created(self, e: FileSystemEvent) -> None:
        if not e.is_directory:
            self._add(e.src_path)

    on_modified = on_created
    on_deleted = on_created

    def on_moved(self, e: FileSystemEvent) -> None:
        if not e.is_directory:
            self._add(e.src_path)
            self._add(e.dest_path)

    def flush(self) -> None:
        with self.lock:
            batch, self.pending = sorted(self.pending), set()
        if batch:
            self.jobs.submit("index_paths", {"paths": batch}, trigger="watcher")

    def _loop(self) -> None:
        while not self._stop.wait(self.delay):
            try:
                self.flush()
            except Exception as exc:
                log.warning("watch: could not queue changes: %s", exc)

    def stop(self) -> None:
        self._stop.set()


class Watcher:
    def __init__(self, jobs: JobQueue, indexer: Indexer):
        self.handler = _Handler(jobs, indexer)
        self.observer = Observer()
        self.observer.daemon = True
        for root in indexer.cfg.root_paths:
            if root.is_dir():
                self.observer.schedule(self.handler, str(root), recursive=True)
                log.info("watching %s", root)

    def start(self) -> None:
        self.observer.start()

    def stop(self) -> None:
        self.handler.stop()
        self.observer.stop()
        self.observer.join(timeout=5)
