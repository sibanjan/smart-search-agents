"""Job queue: the daemon's single writer.

Every change to index.db (file index, image scan, classification) is a job. One worker
thread runs them strictly one at a time, so SQLite never sees two writers. Jobs and their
history live in the `jobs` table of index.db.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from smartsearch.engine import Engine

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    args TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,          -- queued | running | done | failed | cancelled
    progress TEXT,
    report TEXT,
    error TEXT,
    trigger TEXT NOT NULL,         -- schedule | watcher | api | mcp | cli | startup
    created REAL NOT NULL,
    started REAL,
    finished REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, id);
"""

KINDS = ("index", "index_paths", "image_scan", "image_classify", "rebuild")
ACTIVE = ("queued", "running")


class Cancelled(Exception):
    pass


@dataclass
class JobContext:
    job_id: int
    cancel: threading.Event
    report_progress: Callable[[str], None]

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()


class JobQueue:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.store = engine.store
        self._wake: queue.Queue[int] = queue.Queue()
        self._cancel: dict[int, threading.Event] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.handlers: dict[str, Callable[[dict, JobContext], dict]] = {
            "index": self._run_index,
            "index_paths": self._run_index_paths,
            "image_scan": self._run_image_scan,
            "image_classify": self._run_image_classify,
            "rebuild": self._run_rebuild,
        }
        with self.store.lock, self.conn:
            self.conn.executescript(SCHEMA)
            # Anything left running belongs to a previous daemon that died.
            self.conn.execute(
                "UPDATE jobs SET status='failed', error='daemon restarted', finished=? "
                "WHERE status='running'",
                (time.time(),),
            )

    @property
    def conn(self):
        return self.store.conn  # per-thread connection in the daemon

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self._thread.start()
        for row in self.conn.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY id"):
            self._wake.put(row["id"])

    def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        for ev in self._cancel.values():
            ev.set()
        self._wake.put(-1)
        if self._thread:
            self._thread.join(timeout)

    # ---- API -------------------------------------------------------------
    def submit(self, kind: str, args: dict | None = None, trigger: str = "api") -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}; expected one of {', '.join(KINDS)}")
        args = args or {}
        with self.store.lock, self.conn:
            if kind == "index_paths":
                # Merge into a queued index_paths job instead of piling up many.
                row = self.conn.execute(
                    "SELECT id, args FROM jobs WHERE kind='index_paths' AND status='queued' "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    merged = json.loads(row["args"])
                    paths = dict.fromkeys(merged.get("paths", []) + args.get("paths", []))
                    merged["paths"] = list(paths)
                    self.conn.execute(
                        "UPDATE jobs SET args=? WHERE id=?", (json.dumps(merged), row["id"])
                    )
                    return self.get(row["id"])
            else:
                # Coalesce: the same job already waiting is not queued twice.
                row = self.conn.execute(
                    "SELECT id FROM jobs WHERE kind=? AND args=? AND status='queued'",
                    (kind, json.dumps(args, sort_keys=True)),
                ).fetchone()
                if row:
                    return self.get(row["id"])
            cur = self.conn.execute(
                "INSERT INTO jobs(kind,args,status,trigger,created) VALUES(?,?,?,?,?)",
                (kind, json.dumps(args, sort_keys=True), "queued", trigger, time.time()),
            )
            job_id = cur.lastrowid
        self._wake.put(job_id)
        return self.get(job_id)

    def cancel(self, job_id: int) -> dict | None:
        job = self.get(job_id)
        if job is None or job["status"] not in ACTIVE:
            return job
        if job["status"] == "queued":
            self._finish(job_id, "cancelled")
        elif job_id in self._cancel:
            self._cancel[job_id].set()  # the worker stops between files
        return self.get(job_id)

    def get(self, job_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row(row) if row else None

    def list(self, status: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict]:
        sql, params = "SELECT * FROM jobs WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [_row(r) for r in self.conn.execute(sql, params)]

    def last_done(self, kind: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE kind=? AND status='done' ORDER BY finished DESC LIMIT 1",
            (kind,),
        ).fetchone()
        return _row(row) if row else None

    def wait(self, job_id: int, timeout: float = 30) -> dict | None:
        """Block until a job leaves queued/running (used by tests and the CLI)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get(job_id)
            if job and job["status"] not in ACTIVE:
                return job
            time.sleep(0.05)
        return self.get(job_id)

    # ---- worker ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._wake.get()
            if job_id < 0:
                break
            job = self.get(job_id)
            if job is None or job["status"] != "queued":
                continue
            self._run(job)

    def _run(self, job: dict) -> None:
        job_id = job["id"]
        cancel = self._cancel[job_id] = threading.Event()
        last = [0.0]

        def progress(text: str) -> None:
            if time.time() - last[0] >= 1.0:  # throttle DB writes to ~1/s
                last[0] = time.time()
                with self.store.lock, self.conn:
                    self.conn.execute("UPDATE jobs SET progress=? WHERE id=?", (text, job_id))

        with self.store.lock, self.conn:
            claimed = self.conn.execute(
                "UPDATE jobs SET status='running', started=? WHERE id=? AND status='queued'",
                (time.time(), job_id),
            ).rowcount
        if not claimed:  # cancelled in the meantime
            self._cancel.pop(job_id, None)
            return
        job = self.get(job_id)  # re-read: index_paths may have merged more paths
        log.info("job %s %s started", job_id, job["kind"])
        try:
            report = self.handlers[job["kind"]](job["args"], JobContext(job_id, cancel, progress))
            self._finish(job_id, "done", report=report)
        except Cancelled:
            self._finish(job_id, "cancelled")
        except Exception as exc:  # a failed job never takes the worker down
            log.exception("job %s failed", job_id)
            self._finish(job_id, "failed", error=str(exc))
        finally:
            self._cancel.pop(job_id, None)
        log.info("job %s %s %s", job_id, job["kind"], self.get(job_id)["status"])

    def _finish(self, job_id: int, status: str, report: dict | None = None, error: str | None = None):
        with self.store.lock, self.conn:
            self.conn.execute(
                "UPDATE jobs SET status=?, report=?, error=?, finished=? WHERE id=?",
                (status, json.dumps(report) if report is not None else None, error, time.time(), job_id),
            )

    # ---- job kinds -------------------------------------------------------
    def _run_index(self, args: dict, ctx: JobContext) -> dict:
        def progress(path: Path, i: int) -> None:
            ctx.check()
            ctx.report_progress(f"[{i + 1}] {path}")

        return self.engine.indexer.run(force=bool(args.get("force")), progress=progress).__dict__

    def _run_index_paths(self, args: dict, ctx: JobContext) -> dict:
        indexer, counts = self.engine.indexer, {"indexed": 0, "removed": 0, "skipped": 0}
        for path in map(Path, args.get("paths", [])):
            ctx.check()
            if not path.exists():
                indexer.remove_file(path)
                counts["removed"] += 1
            elif indexer.is_candidate(path):
                status = indexer.index_file(path)
                counts["indexed" if status in ("added", "updated") else "skipped"] += 1
            else:
                counts["skipped"] += 1
        return counts

    def _run_image_scan(self, args: dict, ctx: JobContext) -> dict:
        def progress(path: Path, i: int) -> None:
            ctx.check()
            ctx.report_progress(f"[{i + 1}] {path}")

        roots = [Path(r) for r in args["roots"]] if args.get("roots") else None
        dropped = 0 if roots else self._drop_images_outside_roots()
        rep = self.engine.images.scan(roots, progress=progress)  # also re-classifies
        return {**rep.__dict__, "dropped": dropped, "categories": self.engine.images.summary()}

    def _drop_images_outside_roots(self) -> int:
        """The core scan only forgets images under the folders it scanned, so images from a
        folder the user removed in Settings would stay forever. Their labels go by cascade."""
        roots = self.engine.images.roots
        rows = self.conn.execute("SELECT id, path FROM images").fetchall()
        gone = [(r["id"],) for r in rows
                if not any(Path(r["path"]).is_relative_to(root) for root in roots)]
        with self.store.lock, self.conn:
            self.conn.executemany("DELETE FROM images WHERE id=?", gone)
        return len(gone)

    def _run_image_classify(self, args: dict, ctx: JobContext) -> dict:
        return {"categories": self.engine.images.classify()}

    def _run_rebuild(self, args: dict, ctx: JobContext) -> dict:
        target = args.get("target", "files")
        if target == "images":
            with self.store.lock, self.conn:
                self.conn.execute("DELETE FROM images")
            return self._run_image_scan({}, ctx)
        self.store.clear()
        return self._run_index({"force": True}, ctx)


def _row(r) -> dict:
    d = dict(r)
    d["args"] = json.loads(d["args"] or "{}")
    d["report"] = json.loads(d["report"]) if d.get("report") else None
    return d
