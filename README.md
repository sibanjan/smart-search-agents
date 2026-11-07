# SmartSearch Agents

A background daemon, an MCP server and two LLM agents on top of
[SmartSearch](https://github.com/sibanjan/SmartSearch) (local semantic + keyword search over your files, and CLIP image
categories). Implements the *SmartSearch Multi-Agent Architecture Design*.

```
 Web UI / CLI ──HTTP──┐
 MCP clients ─┐       │
 Ask agent ───┼─MCP─▶ MCP server ──HTTP──▶ smartsearchd daemon ──▶ smartsearch core ──▶ index.db
 Curator ─────┘       (no models,          (only writer: job queue,
                       no DB access)         scheduler, watcher, models)
```

- **Daemon** (`ssa daemon run`): loads the models once, serves the web UI and API on
  `127.0.0.1:8765`, and runs every write as a queued job, one at a time. A watcher queues
  re-index jobs for changed files; a cron schedule runs a nightly index and a weekly image scan,
  and catches up on start if a run was missed.
- **MCP server** (`ssa mcp`): SmartSearch as tools for any MCP client. It forwards to the daemon.
- **Ask agent** (`ssa ask`): multi-step search-and-read with `[n]` citations.
- **Curator agent** (`ssa curate`): proposes image categories; you accept or reject them.

SmartSearch itself is used as the core library, unchanged.

## Setup (Windows, Python 3.10+)

```powershell
cd smart-search-agents
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e "..\smart-search[ai]" -e ".[claude,dev]"
```

Configuration is the same `config.yaml` SmartSearch reads (`$SMARTSEARCH_CONFIG`, then
`./config.yaml`, then `~/.smartsearch/config.yaml`). Add the `daemon`, `schedule` and `agents`
blocks from [config.example.yaml](config.example.yaml) to it.

## Run

```powershell
ssa daemon run                 # foreground; Ctrl+C to stop
ssa daemon install             # or: start at log-on (Task Scheduler, no console window)
ssa daemon status
ssa jobs run index --wait      # queue a job (index | image_scan | image_classify | rebuild)
ssa jobs list
ssa query "SELECT category, COUNT(*) FROM image_category_view GROUP BY category"
```

Open http://127.0.0.1:8765 for the web UI: **Files**, **Images** and **Settings**. The CLI and MCP
server authenticate with the token in `~/.smartsearch/token` (created on first run); the browser
gets it as a same-site cookie.

## First run and choosing an assistant

The first time the page opens it shows **Setup**: the folders to search and the assistant. The
same choice is in **Settings → AI assistant**. The options and their descriptions live in
[agents/catalog.py](src/smart_search_agents/agents/catalog.py):

| Option | Where it runs | Notes |
|---|---|---|
| No assistant | — | Search and images only; Ask AI is hidden |
| Qwen3 (4B) | This computer (Ollama) | 2.5 GB; best local answers (10/10 in our test) but 2-12 min per question |
| Llama 3.2 (3B) | This computer (Ollama) | 2.0 GB; under a minute per question; 8/10 in our test |
| Llama 3.1 (8B) | This computer (Ollama) | 4.9 GB, 16 GB RAM recommended; 5/10 in our test |
| Claude Opus 5.5 / Sonnet 5.5 / Haiku 5.5 | Anthropic | Needs an API key; question and excerpts are sent to Anthropic; cost estimates shown |

Local models can be downloaded from the page (through Ollama, with progress). Setup will not
finish with a model that is not downloaded or a Claude choice without a key.

The local numbers come from a small benchmark (five questions with known answers, twice each,
on an Intel Core Ultra 7 laptop on battery). No local model could suggest image categories;
that needs Claude. Claude was not part of the benchmark.

## In the web UI

- **Ask AI** runs the Ask agent: you see each search and file it reads, the answer streams in, and
  the footer says which model answered and whether excerpts left the computer. **Stop** ends it.
- **Images → Category suggestions** runs the Curator and lists its proposals with a preview (how
  many images would move, from where, with examples). **Accept** re-sorts the images; nothing
  changes before that. Small local models often stop before suggesting anything; Claude does
  this task much better.
- **Settings → Recent activity** lists index and scan jobs, with **Cancel** for running ones.

The agents still reach the index only through MCP: the daemon gives each run an in-process MCP
server that calls its own HTTP API, and streams the run to the page as server-sent events
(`/api/agent/ask?q=`, `/api/agent/curate`).

## Settings

The Settings page edits the active `config.yaml` (the first save keeps a `.bak` copy, because YAML
comments are not preserved). Each change does what it needs:

| Change | Takes effect |
|---|---|
| Folders, file types, size limits | At once, and queues a file check and/or image scan |
| Image categories and thresholds | At once, and re-sorts images without a re-scan |
| Search weights, schedule, watched folders, assistant | At once |
| Text or image model, embedder, port | After a restart; a model change clears that index |
| Chunk size and overlap | New and changed files; rebuild the index to apply to all |

The Claude API key goes to Windows Credential Manager, never to `config.yaml`. The daemon's host
and `index_dir` are not editable. Remove a folder and its files leave the index on the next check.

## Agents

```powershell
ssa ask "how did I evaluate the RAG pipeline?"            # local llama3.2 via Ollama
ssa ask --backend claude "summarise my thesis feedback"    # Claude: excerpts leave the laptop
ssa curate                                                 # propose image categories
ssa suggestions list                                       # review them
ssa suggestions accept 3                                   # adds it and re-links images
ssa suggestions export                                     # YAML to paste into config.yaml
```

Claude uses `claude-opus-5-5` (adaptive thinking, effort `medium`, server-side refusal
fallbacks on) and needs `ANTHROPIC_API_KEY` or an `ant auth login` profile.

Accepted suggestions take effect in the daemon at once and survive restarts (they live in the
database); `export` writes them into your config permanently.

## Use the tools from Claude Code or Claude Desktop

This folder has a project-level `.mcp.json`, so Claude Code opened here sees the `smartsearch`
server. Elsewhere:

```powershell
claude mcp add smartsearch -- "C:\path\to\smart-search-agents\.venv\Scripts\python.exe" -m smart_search_agents mcp
```

Or run it over HTTP for several clients: `ssa mcp --http` (serves `127.0.0.1:8766/mcp`).
Every HTTP request must carry the daemon token, otherwise it gets a 401:

```powershell
$token = Get-Content "$HOME\.smartsearch\token"
claude mcp add --transport http smartsearch http://127.0.0.1:8766/mcp --header "Authorization: Bearer $token"
```

| Tool | Kind | Does |
|---|---|---|
| `search_files` | read | Hybrid search with ext / path / since filters |
| `get_chunks` | read | Full text chunks of a file, ranked by a query |
| `list_image_categories`, `list_images`, `image_details` | read | Image categories and links |
| `preview_category` | read | Dry run of a new category prompt |
| `index_status`, `list_jobs` | read | Daemon, index and job state |
| `query_db` | read | One SELECT/WITH, read-only connection, 500 rows, 5 s |
| `run_job`, `cancel_job` | action | Queue or cancel index / scan jobs |
| `suggest_category` | action | Store a category proposal for review |

Resources: `smartsearch://config`, `smartsearch://stats`, `smartsearch://categories`.
Prompts: `ask_my_files`, `review_image_categories`. No tool opens, moves or deletes files.

## Project layout

```
src/smart_search_agents/
  settings.py        daemon / schedule / agents settings from config.yaml, token
  client.py          HTTP client for the daemon (MCP server, CLI)
  daemon/
    db.py            per-thread SQLite connections (worker writes while API threads read)
    jobs.py          job queue: single writer, coalescing, cancel, history in `jobs`
    scheduler.py     APScheduler cron + catch-up
    watch.py         watchdog -> index_paths jobs (debounced 2 s)
    curation.py      preview_category, category_suggestions
    config_edit.py   Settings page: checks each setting, saves it, applies it
    agent_runner.py  runs the agents for the UI and streams their events
    service.py       FastAPI: SmartSearch API + UI, jobs, chunks, SQL, config, suggestions
    main.py          wires it together; pid file
  mcp_server.py      MCPServer (mcp 2.x) tools, resources, prompts
  agents/
    llm.py           Ollama and Claude backends with tool calling
    tools.py         MCP client tool box (allow-listed per agent)
    base.py          agent loop with a round limit
    ask.py, curator.py
  web/index.html     the UI: Files, Images, Settings
  cli.py             `ssa` commands
tests/               offline: hash embedder, fake CLIP, scripted LLM
```

## Tests

```powershell
python -m pytest -q
```
