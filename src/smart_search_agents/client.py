"""HTTP client for the daemon, shared by the MCP server and the CLI."""

from __future__ import annotations

from typing import Any

import httpx

from .settings import Settings, load_settings, read_or_create_token


class DaemonUnavailable(RuntimeError):
    pass


class DaemonClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ):
        settings = settings or load_settings()
        self.base_url = base_url or settings.base_url
        token = token or read_or_create_token(settings.token_path)
        self.http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            transport=transport,
        )

    def _call(self, method: str, path: str, **kw) -> Any:
        try:
            r = self.http.request(method, path, **kw)
        except httpx.ConnectError as exc:
            raise DaemonUnavailable(
                f"SmartSearch daemon is not running at {self.base_url}. Start it with: ssa daemon run"
            ) from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise RuntimeError(f"{r.status_code}: {detail}")
        return r.json()

    # endpoint is positional-only: `path` is also a search parameter
    def get(self, endpoint: str, /, **params) -> Any:
        return self._call("GET", endpoint, params={k: v for k, v in params.items() if v is not None})

    def post(self, endpoint: str, /, body: dict | None = None) -> Any:
        return self._call("POST", endpoint, json=body or {})

    def delete(self, endpoint: str, /) -> Any:
        return self._call("DELETE", endpoint)

    # ---- typed helpers ---------------------------------------------------
    def search(self, query: str, ext: list[str] | None = None, path: str | None = None,
               since: str | None = None, mode: str = "hybrid", limit: int = 10) -> list[dict]:
        params = {"q": query, "mode": mode, "limit": limit, "path": path, "since": since}
        if ext:
            params["ext"] = [e if e.startswith(".") else "." + e for e in ext]
        return self.get("/api/search", **params)["results"]

    def chunks(self, path: str, query: str | None = None, limit: int = 5) -> dict:
        return self.get("/api/chunks", path=path, query=query, limit=limit)

    def submit(self, kind: str, args: dict | None = None, trigger: str = "api") -> dict:
        return self.post("/api/jobs", {"kind": kind, "args": args or {}, "trigger": trigger})

    def status(self) -> dict:
        return self.get("/api/daemon/status")
