"""The daemon's HTTP API: SmartSearch's existing API + web UI, plus jobs, chunks, image
details, read-only SQL, config and category suggestions.

All writes go through the job queue. Requests must come to 127.0.0.1/localhost and carry
the token (Authorization: Bearer <token>); the web UI gets it as a same-site cookie when
the page loads.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from smartsearch.engine import Engine
from smartsearch.web.app import create_app as create_core_app

from ..settings import Settings
from .curation import Curation
from .jobs import JobQueue

QUERY_ROW_LIMIT = 500
QUERY_TIMEOUT_S = 5.0
COOKIE = "ssa_session"


class JobRequest(BaseModel):
    kind: str
    args: dict = {}
    trigger: str = "api"


class CategoryRequest(BaseModel):
    name: str
    prompt: str
    evidence: list[str] = []
    rationale: str = ""


class SqlRequest(BaseModel):
    sql: str


def create_app(
    engine: Engine,
    jobs: JobQueue,
    curation: Curation,
    settings: Settings,
    token: str,
    scheduler=None,
    allowed_hosts: set[str] | None = None,
) -> FastAPI:
    app = create_core_app(engine, watch=False, index_on_start=False)
    # The core app re-indexes in a thread of its own; route it through the queue instead.
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") != "/api/reindex"]
    hosts = allowed_hosts or {"127.0.0.1", "localhost"}
    started = time.time()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in hosts:  # blocks DNS-rebinding pages in a browser
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {token}" and request.cookies.get(COOKIE) != token:
                return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
        response = await call_next(request)
        if request.url.path == "/":
            response.set_cookie(COOKIE, token, httponly=True, samesite="strict")
        return response

    # ---- jobs ------------------------------------------------------------
    @app.get("/api/jobs")
    def list_jobs(status: str | None = None, kind: str | None = None, limit: int = Query(50, le=500)):
        return {"jobs": jobs.list(status, kind, limit)}

    @app.post("/api/jobs")
    def submit_job(req: JobRequest):
        try:
            return jobs.submit(req.kind, req.args, req.trigger)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/reindex")
    def reindex():
        return {"ok": True, "job": jobs.submit("index", {}, "api")}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int):
        return jobs.get(job_id) or _404("job")

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: int):
        return jobs.cancel(job_id) or _404("job")

    # ---- reading ---------------------------------------------------------
    @app.get("/api/chunks")
    def chunks(path: str, query: str | None = None, limit: int = Query(5, le=50)):
        """Full chunk texts of one indexed file, optionally ranked by a query."""
        f = engine.store.get_file(path)
        if f is None:
            raise HTTPException(404, "File is not in the index")
        rows = engine.store.conn.execute(
            "SELECT ord, text, vec FROM chunks WHERE file_id=? ORDER BY ord", (f.id,)
        ).fetchall()
        scores = [None] * len(rows)
        if query and rows and all(r["vec"] is not None for r in rows):
            q = engine.embedder.encode([query], query=True)[0]
            mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
            if mat.shape[1] == q.shape[0]:
                scores = (mat @ q).tolist()
        items = [
            {"ord": r["ord"], "text": r["text"], "score": None if s is None else round(s, 4)}
            for r, s in zip(rows, scores)
        ]
        if query:
            items.sort(key=lambda c: -(c["score"] or 0))
        return {"path": path, "chunks": items[:limit], "total_chunks": len(rows)}

    @app.get("/api/images/detail")
    def image_detail(path: str):
        row = engine.store.conn.execute(
            "SELECT id, path, size, mtime, width, height FROM images WHERE path=?", (path,)
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Image is not in the index")
        labels = engine.images._labels_for([row["id"]]).get(row["id"], [])
        return {**{k: row[k] for k in ("path", "size", "mtime", "width", "height")}, "labels": labels}

    @app.post("/api/query")
    def query(req: SqlRequest):
        return run_readonly_sql(engine.store.path, req.sql)

    @app.get("/api/config")
    def config():
        c = engine.cfg
        return {
            "config_file": str(settings.config_file) if settings.config_file else None,
            "roots": c.roots,
            "image_roots": c.image_roots or c.roots,
            "extensions": c.extensions,
            "text_model": engine.store.get_meta("embedder") or c.model_name,
            "image_model": c.image_model,
            "image_categories": c.image_categories,
            "image_min_confidence": c.image_min_confidence,
            "image_label_threshold": c.image_label_threshold,
            "schedule": settings.schedule,
            "agents": {"backend": settings.agents.backend, "model": settings.agents.model,
                       "claude_model": settings.agents.claude_model},
        }

    @app.get("/api/daemon/status")
    def daemon_status():
        last = {k: _brief(jobs.last_done(k)) for k in ("index", "image_scan", "image_classify")}
        return {
            "uptime_s": round(time.time() - started),
            "index": engine.store.stats(),
            "images": engine.images.count(),
            "running": jobs.list("running", limit=5),
            "queued": len(jobs.list("queued", limit=500)),
            "last_done": last,
            "next_runs": scheduler.next_runs() if scheduler else {},
            "config_file": str(settings.config_file) if settings.config_file else None,
        }

    # ---- categories ------------------------------------------------------
    @app.post("/api/images/preview_category")
    def preview_category(req: CategoryRequest):
        return curation.preview(req.name, req.prompt)

    @app.get("/api/suggestions")
    def suggestions(status: str | None = None):
        return {"suggestions": curation.list(status)}

    @app.post("/api/suggestions")
    def suggest(req: CategoryRequest):
        try:
            return curation.suggest(req.name, req.prompt, req.evidence, req.rationale)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/suggestions/{sid}/{decision}")
    def decide(sid: int, decision: str):
        if decision not in ("accept", "reject"):
            raise HTTPException(400, "decision must be accept or reject")
        try:
            s = curation.decide(sid, decision == "accept")
        except KeyError:
            _404("suggestion")
        job = jobs.submit("image_classify", {}, "api") if decision == "accept" else None
        return {"suggestion": s, "job": job}

    @app.get("/api/suggestions/export")
    def export():
        return {"yaml": curation.export_yaml()}

    return app


def run_readonly_sql(db_path: Path, sql: str) -> dict:
    """One SELECT/WITH statement on a read-only connection, capped in rows and time."""
    stmt = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", stmt) or ";" in stmt:
        raise HTTPException(400, "Only a single SELECT or WITH statement is allowed")
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    deadline = time.time() + QUERY_TIMEOUT_S
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(stmt)
        rows = cur.fetchmany(QUERY_ROW_LIMIT + 1)
        cols = [d[0] for d in cur.description or []]
    except sqlite3.Error as exc:
        raise HTTPException(400, f"SQL error: {exc}") from exc
    finally:
        conn.close()
    return {
        "columns": cols,
        "rows": [[_jsonable(v) for v in r] for r in rows[:QUERY_ROW_LIMIT]],
        "truncated": len(rows) > QUERY_ROW_LIMIT,
    }


def _jsonable(v):
    return f"<{len(v)} bytes>" if isinstance(v, (bytes, memoryview)) else v


def _brief(job: dict | None) -> dict | None:
    return {k: job[k] for k in ("id", "finished", "report")} if job else None


def _404(what: str):
    raise HTTPException(404, f"{what} not found")
