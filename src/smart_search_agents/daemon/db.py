"""Thread-safe access to index.db for the daemon.

SmartSearch's Store shares one sqlite3 connection. That is fine for the CLI, but in the
daemon the job worker writes while API threads read, and Python's sqlite3 objects must not
be used by two threads at once. Here every thread gets its own connection to the same WAL
database: readers never block on the writer, and the job queue keeps writes one at a time.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Callable

from smartsearch.config import Config
from smartsearch.engine import Engine
from smartsearch.images import ImageCatalog
from smartsearch.store import Store


class ThreadLocalStore(Store):
    def __init__(self, db_path: Path):
        self._local = threading.local()
        super().__init__(db_path)  # its `self.conn = ...` lands in the setter below

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = self._connect()
        return c

    @conn.setter
    def conn(self, value: sqlite3.Connection) -> None:
        self._local.conn = value

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c


class ThreadSafeImageCatalog(ImageCatalog):
    """ImageCatalog keeps `self.conn` from construction; resolve it per thread instead."""

    @property
    def conn(self) -> sqlite3.Connection:
        return self.store.conn

    @conn.setter
    def conn(self, value) -> None:
        pass


def make_engine(cfg: Config, image_encoder_factory: Callable | None = None) -> Engine:
    eng = Engine.__new__(Engine)
    eng.cfg = cfg
    eng.store = ThreadLocalStore(cfg.db_path)
    # cached_property slot: replace the image catalog before anything builds the default one
    eng.__dict__["images"] = ThreadSafeImageCatalog(cfg, eng.store, image_encoder_factory)
    return eng
