"""Command line:  ssa --help"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .settings import load_settings

app = typer.Typer(help="SmartSearch daemon, MCP server and agents.", no_args_is_help=True)
daemon_app = typer.Typer(help="Run and manage the background daemon.", no_args_is_help=True)
jobs_app = typer.Typer(help="Index and scan jobs.", no_args_is_help=True)
sugg_app = typer.Typer(help="Image category suggestions from the Curator.", no_args_is_help=True)
app.add_typer(daemon_app, name="daemon")
app.add_typer(jobs_app, name="jobs")
app.add_typer(sugg_app, name="suggestions")

for _stream in (sys.stdout, sys.stderr):  # cp1252 consoles: never crash on ✓ or ●
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

console = Console()
err = Console(stderr=True)  # stdout must stay clean for the MCP stdio transport
ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config.yaml")
TASK_NAME = "SmartSearchDaemon"


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=err, show_path=False, show_time=False)],
    )
    for noisy in ("httpx", "sentence_transformers", "urllib3", "watchdog", "apscheduler", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _client(config: Optional[Path] = None):
    from .client import DaemonClient

    return DaemonClient(settings=load_settings(config))


def _fail(exc: Exception) -> None:
    console.print(f"[red]{escape(str(exc))}[/]")
    raise typer.Exit(1)


# ---- daemon ----------------------------------------------------------------
@daemon_app.command("run")
def daemon_run(
    config: Optional[Path] = ConfigOpt,
    no_watch: bool = typer.Option(False, "--no-watch", help="Don't watch folders for changes"),
    no_schedule: bool = typer.Option(False, "--no-schedule", help="Don't run scheduled jobs"),
) -> None:
    """Run the daemon in the foreground (Ctrl+C to stop)."""
    from .daemon.main import run

    run(str(config) if config else None, watch=not no_watch, schedule=not no_schedule)


@daemon_app.command("status")
def daemon_status(config: Optional[Path] = ConfigOpt) -> None:
    """Show whether the daemon is up, index size, jobs and schedule."""
    try:
        s = _client(config).status()
    except Exception as exc:
        _fail(exc)
    idx = s["index"]
    console.print(
        f"[green]● running[/] {s['uptime_s']}s · files {idx['files']} · chunks {idx['chunks']} · "
        f"images {s['images']} · queued {s['queued']}\n  config: {s['config_file'] or '(defaults)'}"
    )
    for kind, when in (s.get("next_runs") or {}).items():
        console.print(f"  next {kind}: {when}")
    for j in s["running"]:
        console.print(f"  running #{j['id']} {j['kind']}: {escape(j.get('progress') or '')}")


@daemon_app.command("stop")
def daemon_stop(config: Optional[Path] = ConfigOpt) -> None:
    """Stop the daemon started from this index's pid file."""
    pid_path = load_settings(config).pid_path
    if not pid_path.is_file():
        console.print("[yellow]No daemon pid file; nothing to stop.[/]")
        return
    pid = int(pid_path.read_text().strip())
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, 15)
    pid_path.unlink(missing_ok=True)
    console.print(f"[green]✓ Stopped daemon (pid {pid}).[/]")


@daemon_app.command("install")
def daemon_install(config: Optional[Path] = ConfigOpt) -> None:
    """Start the daemon automatically at log-on (Windows Task Scheduler)."""
    if os.name != "nt":
        console.print("Use a launchd agent (macOS) or a systemd user service (Linux) running "
                      "`ssa daemon run`.")
        raise typer.Exit(1)
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")  # no console window
    cmd = f'"{pyw if pyw.exists() else exe}" -m smart_search_agents daemon run'
    if config:
        cmd += f' --config "{config.resolve()}"'
    r = subprocess.run(
        ["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/RL", "LIMITED", "/TN", TASK_NAME, "/TR", cmd],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _fail(RuntimeError(r.stderr.strip() or r.stdout.strip()))
    console.print(f"[green]✓ Registered task {TASK_NAME}[/] — runs at log-on: {escape(cmd)}\n"
                  f"Start it now with: schtasks /Run /TN {TASK_NAME}")


@daemon_app.command("uninstall")
def daemon_uninstall() -> None:
    """Remove the log-on task."""
    r = subprocess.run(["schtasks", "/Delete", "/F", "/TN", TASK_NAME], capture_output=True, text=True)
    console.print(f"[green]✓ Removed {TASK_NAME}[/]" if r.returncode == 0 else
                  f"[yellow]{escape(r.stderr.strip())}[/]")


# ---- MCP -------------------------------------------------------------------
@app.command("mcp")
def mcp_cmd(
    http: bool = typer.Option(False, "--http", help="Streamable HTTP instead of stdio"),
    port: int = typer.Option(8766, "--port"),
) -> None:
    """Run the MCP server (stdio by default, for `claude mcp add smartsearch -- ssa mcp`)."""
    from .mcp_server import main as run_mcp

    run_mcp(http=http, port=port)


# ---- jobs ------------------------------------------------------------------
@jobs_app.command("list")
def jobs_list(status: Optional[str] = None, limit: int = 20, config: Optional[Path] = ConfigOpt):
    """Recent jobs."""
    try:
        jobs = _client(config).get("/api/jobs", status=status, limit=limit)["jobs"]
    except Exception as exc:
        _fail(exc)
    t = Table("#", "Kind", "Status", "Trigger", "Created", "Took", "Report / progress")
    for j in jobs:
        took = f"{j['finished'] - j['started']:.1f}s" if j.get("finished") and j.get("started") else ""
        info = j.get("error") or (json.dumps(j["report"])[:80] if j.get("report") else j.get("progress") or "")
        t.add_row(str(j["id"]), j["kind"], j["status"], j["trigger"],
                  time.strftime("%m-%d %H:%M", time.localtime(j["created"])), took, escape(info))
    console.print(t)


@jobs_app.command("run")
def jobs_run(
    kind: str = typer.Argument(..., help="index | image_scan | image_classify | rebuild"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for it to finish"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Queue a job in the daemon."""
    try:
        c = _client(config)
        job = c.submit(kind, trigger="cli")
        console.print(f"Queued #{job['id']} {job['kind']} ({job['status']})")
        while wait and job["status"] in ("queued", "running"):
            time.sleep(1)
            job = c.get(f"/api/jobs/{job['id']}")
            console.print(f"  {job['status']} {escape(job.get('progress') or '')}", end="\r")
        if wait:
            console.print(f"\n#{job['id']} {job['status']}: {escape(json.dumps(job.get('report')))}")
    except Exception as exc:
        _fail(exc)


@jobs_app.command("cancel")
def jobs_cancel(job_id: int, config: Optional[Path] = ConfigOpt) -> None:
    """Cancel a queued or running job."""
    try:
        job = _client(config).delete(f"/api/jobs/{job_id}")
    except Exception as exc:
        _fail(exc)
    console.print(f"#{job['id']} {job['status']}")


@app.command("query")
def query_cmd(sql: str, config: Optional[Path] = ConfigOpt) -> None:
    """Read-only SQL on the index, through the daemon."""
    try:
        res = _client(config).post("/api/query", {"sql": sql})
    except Exception as exc:
        _fail(exc)
    t = Table(*res["columns"])
    for row in res["rows"]:
        t.add_row(*[escape(str(v)) for v in row])
    console.print(t)
    if res["truncated"]:
        console.print("[dim](first 500 rows)[/]")


# ---- agents ----------------------------------------------------------------
def _backend(config: Optional[Path], backend: Optional[str], model: Optional[str]):
    from .agents.llm import make_backend

    s = load_settings(config)
    if backend:
        s.agents.backend = backend
    if model:
        if s.agents.backend == "claude":
            s.agents.claude_model = model
        else:
            s.agents.model = model
    llm = make_backend(s)
    if not llm.local:
        err.print(f"[yellow]Using {llm.name}: file excerpts are sent to Anthropic.[/]")
    return s, llm


def _show_event(kind: str, data: dict) -> None:
    if kind == "tool":
        err.print(f"[dim]→ {data['name']}({escape(json.dumps(data['args'], ensure_ascii=False))[:120]})[/]")


@app.command("ask")
def ask_cmd(
    question: str,
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="ollama | claude"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Answer a question from your files (Ask agent, via MCP tools)."""
    from .agents.ask import TOOLS, ask
    from .agents.tools import McpToolBox

    s, llm = _backend(config, backend, model)
    limit = s.agents.context_chars if llm.local else 40000

    async def go():
        async with McpToolBox(TOOLS, max_chars=limit) as tools:
            return await ask(question, llm, tools, s.agents.max_rounds, _show_event)

    try:
        res = asyncio.run(go())
    except Exception as exc:
        _fail(exc)
    console.print(Panel(escape(res.text or "(no answer)"), title=f"Answer · {llm.name}",
                        subtitle=f"{res.rounds} tool rounds", border_style="green"))


@app.command("curate")
def curate_cmd(
    focus: Optional[str] = typer.Option(None, "--focus", help="e.g. 'screenshots'"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Review image categories and record suggestions (Curator agent)."""
    from .agents.curator import TOOLS, curate
    from .agents.tools import McpToolBox

    s, llm = _backend(config, backend, model)

    async def go():
        async with McpToolBox(TOOLS, max_chars=s.agents.context_chars if llm.local else 40000) as tools:
            return await curate(llm, tools, focus, max(12, s.agents.max_rounds), _show_event)

    try:
        res = asyncio.run(go())
    except Exception as exc:
        _fail(exc)
    console.print(Panel(escape(res.text or "(no summary)"), title=f"Curator · {llm.name}",
                        border_style="cyan"))
    console.print("Review with: [bold]ssa suggestions list[/]")


# ---- suggestions -----------------------------------------------------------
@sugg_app.command("list")
def sugg_list(status: Optional[str] = "proposed", config: Optional[Path] = ConfigOpt) -> None:
    """Category suggestions (default: those waiting for a decision)."""
    try:
        items = _client(config).get("/api/suggestions", status=status)["suggestions"]
    except Exception as exc:
        _fail(exc)
    t = Table("#", "Name", "Prompt", "Would join", "From", "Status", show_lines=True)
    for s in items:
        p = s.get("preview") or {}
        t.add_row(str(s["id"]), escape(s["name"]), escape(s["prompt"]), str(p.get("would_join", "")),
                  escape(", ".join(f"{k} {v}" for k, v in (p.get("from") or {}).items())), s["status"])
    console.print(t)


@sugg_app.command("accept")
def sugg_accept(sid: int, config: Optional[Path] = ConfigOpt) -> None:
    """Accept a suggestion: adds the category and re-links images (no re-scan)."""
    try:
        r = _client(config).post(f"/api/suggestions/{sid}/accept")
    except Exception as exc:
        _fail(exc)
    console.print(f"[green]✓ Accepted {escape(r['suggestion']['name'])}[/]; re-classifying "
                  f"(job #{r['job']['id']}). Keep it permanently: [bold]ssa suggestions export[/]")


@sugg_app.command("reject")
def sugg_reject(sid: int, config: Optional[Path] = ConfigOpt) -> None:
    """Reject a suggestion."""
    try:
        r = _client(config).post(f"/api/suggestions/{sid}/reject")
    except Exception as exc:
        _fail(exc)
    console.print(f"Rejected {escape(r['suggestion']['name'])}")


@sugg_app.command("export")
def sugg_export(config: Optional[Path] = ConfigOpt) -> None:
    """Print image_categories (including accepted suggestions) as YAML for config.yaml."""
    try:
        print(_client(config).get("/api/suggestions/export")["yaml"])
    except Exception as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
