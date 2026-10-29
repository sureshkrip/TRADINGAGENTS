"""FastAPI wrapper that runs the trading pipeline as a web service.

A single ``TradingAgentsGraph`` is built lazily and reused across requests
(matching the CLI, whose memory/reflection is designed to carry across runs).
Because a full run takes minutes, ``POST /analyze`` submits a background job and
returns immediately with a ``job_id``; poll ``GET /analyze/{job_id}`` for the
result. Configuration (LLM provider, models, API keys) comes from the same
``TRADINGAGENTS_*`` / provider env vars the CLI honors — nothing app-specific.

Run locally:  ``uvicorn tradingagents.web.server:app --port 8000``
Or:           ``tradingagents-web``  (console script, see pyproject.toml)
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# NOTE: the heavy agent stack (langchain, langgraph, ...) is imported lazily in
# _get_graph(), not at module load. This keeps app startup and /health fast and
# ensures a misconfigured pipeline fails an individual job rather than the whole
# web process.

# --- Job store -------------------------------------------------------------
# In-process job registry. Analysis is long-running, so requests are handled
# asynchronously. This is deliberately simple (single process, in-memory): jobs
# do not survive a restart. Bounded by TRADINGAGENTS_WEB_MAX_JOBS so completed
# jobs do not accumulate without bound.

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = max(10, int(os.getenv("TRADINGAGENTS_WEB_MAX_JOBS", "200")))

# Run analyses off the event loop. Default to one at a time: the graph writes to
# shared on-disk dirs (results/cache/memory) and reuses one graph instance, so
# serial execution avoids cross-run interference. Override with the env var.
_MAX_WORKERS = max(1, int(os.getenv("TRADINGAGENTS_WEB_WORKERS", "1")))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="ta-analyze")

_GRAPH: Any = None
_GRAPH_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prune_jobs_unlocked() -> None:
    """Evict oldest completed/errored jobs when _JOBS exceeds _MAX_JOBS."""
    if len(_JOBS) <= _MAX_JOBS:
        return
    terminal = [
        (jid, j.get("created_at", ""))
        for jid, j in _JOBS.items()
        if j.get("status") in ("done", "error")
    ]
    terminal.sort(key=lambda item: item[1])
    excess = len(_JOBS) - _MAX_JOBS
    for jid, _ in terminal[:excess]:
        _JOBS.pop(jid, None)


def _get_graph() -> Any:
    """Build the graph once and reuse it (thread-safe, lazy).

    The agent stack is imported here (not at module load) so the web app can
    start and serve /health without the full pipeline's dependencies resolving.
    """
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                from tradingagents.default_config import DEFAULT_CONFIG
                from tradingagents.graph.trading_graph import TradingAgentsGraph

                _GRAPH = TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
    return _GRAPH


def _extract_reports(final_state: dict[str, Any] | None) -> dict[str, str]:
    """Extract all individual agent and debate reports from a completed graph state."""
    if not isinstance(final_state, dict):
        return {}
    inv_debate = final_state.get("investment_debate_state") or {}
    risk_debate = final_state.get("risk_debate_state") or {}
    return {
        "market_report": final_state.get("market_report") or "",
        "sentiment_report": final_state.get("sentiment_report") or "",
        "news_report": final_state.get("news_report") or "",
        "fundamentals_report": final_state.get("fundamentals_report") or "",
        "investment_plan": final_state.get("investment_plan") or "",
        "trader_investment_plan": final_state.get("trader_investment_plan") or "",
        "final_trade_decision": final_state.get("final_trade_decision") or "",
        "bull_history": inv_debate.get("bull_history") or "",
        "bear_history": inv_debate.get("bear_history") or "",
        "debate_history": inv_debate.get("history") or "",
        "risk_history": risk_debate.get("history") or "",
    }


def _run_job(job_id: str, ticker: str, trade_date: str, asset_type: str) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(status="running", started_at=_now())
    try:
        graph = _get_graph()
        final_state, decision = graph.propagate(ticker, trade_date, asset_type=asset_type)
        state_dict = final_state if isinstance(final_state, dict) else {}
        result = {
            "decision": decision,
            "final_trade_decision": state_dict.get("final_trade_decision"),
            "reports": _extract_reports(state_dict),
        }
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="done", finished_at=_now(), result=result, error=None)
    except Exception as exc:  # surface the failure to the poller rather than 500-ing silently
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status="error",
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )


def _run_funnel_job(job_id: str, req: dict) -> None:
    """Run the full screening funnel (universe -> screen -> triage -> analyze -> report).

    Long-running like ``_run_job`` (it invokes the multi-agent graph per pick),
    so it runs on the same executor and reports via the same job store. The
    result keeps the funnel light — stage counts, the ranked picks, and the
    on-disk report paths — rather than inlining the full write-ups.
    """
    with _JOBS_LOCK:
        _JOBS[job_id].update(status="running", started_at=_now())
    try:
        # Lazy import: keep the heavy funnel/agent stack off the app's import path
        # (same rationale as _get_graph) so /health stays fast and a misconfigured
        # pipeline fails this job, not the whole web process.
        from tradingagents.funnel.pipeline import run_funnel, run_theme_funnel
        from tradingagents.funnel.report import (
            classify_decision,
            screened_to_dicts,
            triaged_to_dicts,
        )

        if req.get("theme"):
            out = run_theme_funnel(
                req["theme"],
                req["date"],
                max_deep=req.get("max_deep") if req.get("max_deep") is not None else 2,
                top_triage=req.get("top_triage"),
                write=True,
            )
        else:
            out = run_funnel(
                req["date"],
                req.get("asset_type", "stock"),
                top_screen=req.get("top_screen"),
                top_triage=req.get("top_triage"),
                max_deep=req.get("max_deep"),
                write=True,
            )
        picks = [
            {
                "ticker": a.ticker,
                "bucket": classify_decision(a.decision) if a.error is None else "FAILED",
                "decision": a.decision,
                "triage_score": a.triage_score,
                "screen_score": a.screen_score,
                "thesis": a.thesis,
                "red_flag": a.red_flag,
                "report": a.report,
                "error": a.error,
            }
            for a in out.analyzed
        ]
        result = {
            "universe_size": out.universe_size,
            "screened": len(out.screened),
            "triaged": len(out.triaged),
            "analyzed": len(out.analyzed),
            "report_path": out.report_path,
            "csv_path": out.csv_path,
            "picks": picks,
            "screened_detail": screened_to_dicts(out.screened),
            "triaged_detail": triaged_to_dicts(out.triaged),
        }
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="done", finished_at=_now(), result=result, error=None)
    except Exception as exc:
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status="error",
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )


# --- API models ------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, examples=["NVDA"], description="Ticker symbol to analyze.")
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2024-05-10"],
        description="Trade date (YYYY-MM-DD).",
    )
    asset_type: Literal["stock", "crypto"] = "stock"


class AnalyzeAccepted(BaseModel):
    job_id: str
    status: str


class FunnelRequest(BaseModel):
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2024-05-10"],
        description="As-of trade date (YYYY-MM-DD) used for every analysis.",
    )
    asset_type: Literal["stock", "crypto"] = "stock"
    theme: str | None = Field(
        None, min_length=2, max_length=80,
        description="Run a theme funnel: tickers are discovered from this name (no list needed).",
    )
    # Per-stage cutoffs; None falls back to each stage's TRADINGAGENTS_* default.
    top_screen: int | None = Field(None, ge=1, description="Stage 0 survivors to keep.")
    top_triage: int | None = Field(None, ge=1, description="Stage 1 shortlist size.")
    max_deep: int | None = Field(None, ge=1, description="Hard cap on Stage 2 deep runs.")


# --- App -------------------------------------------------------------------

app = FastAPI(
    title="TradingAgents",
    description="Multi-agent LLM financial trading framework, exposed as a web service.",
    version="0.3.1",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Coolify / load balancers."""
    return {"status": "ok"}


_TICKER_PARAM_RE = re.compile(r"^[A-Za-z0-9._\-\^=:]{1,24}$")


@app.get("/quote/{ticker}")
def get_quote(ticker: str) -> dict[str, Any]:
    """Return Google Finance quote snapshot, resolved identity, and external links."""
    cleaned = ticker.strip().upper()
    if not _TICKER_PARAM_RE.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    from tradingagents.agents.utils.agent_utils import resolve_instrument_identity
    from tradingagents.dataflows.google_finance import (
        fetch_google_finance_quote,
        resolve_google_finance_url,
    )
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    canonical = normalize_symbol(cleaned)
    identity = resolve_instrument_identity(canonical)
    exchange = identity.get("exchange")
    gf_snap = fetch_google_finance_quote(canonical, exchange=exchange, timeout=6.0)
    gf_url = gf_snap.get("url") or resolve_google_finance_url(canonical, exchange=exchange)

    return {
        "ticker": cleaned,
        "canonical": canonical,
        "company_name": gf_snap.get("name") or identity.get("company_name") or cleaned,
        "sector": identity.get("sector"),
        "industry": identity.get("industry"),
        "exchange": gf_snap.get("exchange") or exchange,
        "price": gf_snap.get("price"),
        "stats": gf_snap.get("stats") or {},
        "about": gf_snap.get("about"),
        "google_finance_url": gf_url,
        "yahoo_finance_url": f"https://finance.yahoo.com/quote/{quote_plus(canonical)}",
    }


@app.post("/analyze", response_model=AnalyzeAccepted, status_code=202)
def analyze(req: AnalyzeRequest) -> AnalyzeAccepted:
    """Queue an analysis run. Poll ``GET /analyze/{job_id}`` for the result."""
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _prune_jobs_unlocked()
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": "analyze",
            "status": "queued",
            "request": req.model_dump(),
            "created_at": _now(),
            "result": None,
            "error": None,
        }
    _EXECUTOR.submit(_run_job, job_id, req.ticker, req.date, req.asset_type)
    return AnalyzeAccepted(job_id=job_id, status="queued")


@app.post("/funnel", response_model=AnalyzeAccepted, status_code=202)
def funnel(req: FunnelRequest) -> AnalyzeAccepted:
    """Queue a full screening-funnel run. Poll ``GET /funnel/{job_id}`` for the result.

    Sources the universe, ranks it with the Stage 0 quant screen, triages the
    survivors with the cheap LLM, runs the full multi-agent analysis on the
    shortlist, and writes a ranked Markdown+CSV report. Long-running — the deep
    stage invokes the agent graph per pick.
    """
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _prune_jobs_unlocked()
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": "funnel",
            "status": "queued",
            "request": req.model_dump(),
            "created_at": _now(),
            "result": None,
            "error": None,
        }
    _EXECUTOR.submit(_run_funnel_job, job_id, req.model_dump())
    return AnalyzeAccepted(job_id=job_id, status="queued")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _funnel_runs_dir() -> str:
    from tradingagents.default_config import DEFAULT_CONFIG

    return os.path.join(DEFAULT_CONFIG["results_dir"], "funnel")


# NOTE: these literal routes are declared BEFORE /funnel/{job_id} so "runs" is
# matched here rather than captured as a job_id.
@app.get("/funnel/runs")
def list_funnel_runs() -> dict[str, Any]:
    """List persisted funnel runs (dates, newest first) for the history browser.

    Reads the on-disk archive (``<results_dir>/funnel/<date>/run.json``), so it
    survives restarts — unlike the in-memory job store.
    """
    base = _funnel_runs_dir()
    runs: list[dict[str, Any]] = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            record = os.path.join(base, name, "run.json")
            if not os.path.isfile(record):
                continue
            entry = {"date": name}
            try:
                with open(record, encoding="utf-8") as fh:
                    data = json.load(fh)
                entry["analyzed"] = data.get("analyzed")
                entry["generated_at"] = data.get("generated_at")
                entry["buys"] = sum(1 for p in data.get("picks", []) if p.get("bucket") == "BUY")
            except (OSError, json.JSONDecodeError):
                pass  # a malformed run.json shouldn't hide the rest of the history
            runs.append(entry)
    runs.sort(key=lambda r: r["date"], reverse=True)
    return {"count": len(runs), "runs": runs}


@app.get("/funnel/runs/{date}")
def get_funnel_run(date: str) -> dict[str, Any]:
    """Return one persisted run's full record (counts + picks + write-ups)."""
    if not _DATE_RE.match(date):  # also blocks path traversal via the date param
        raise HTTPException(status_code=400, detail="Invalid date (expected YYYY-MM-DD)")
    record = os.path.join(_funnel_runs_dir(), date, "run.json")
    if not os.path.isfile(record):
        raise HTTPException(status_code=404, detail="No run for that date")
    with open(record, encoding="utf-8") as fh:
        return json.load(fh)


_SLUG_RE = re.compile(r"^[a-z0-9-]{1,80}$")


def _themes_dir() -> str:
    return os.path.join(_funnel_runs_dir(), "themes")


@app.get("/funnel/themes")
def list_funnel_themes() -> dict[str, Any]:
    """List themes that have runs, with their latest run summary (for the UI)."""
    base = _themes_dir()
    themes: list[dict[str, Any]] = []
    if os.path.isdir(base):
        for slug in os.listdir(base):
            tdir = os.path.join(base, slug)
            if not os.path.isdir(tdir):
                continue
            dates = sorted(
                (d for d in os.listdir(tdir) if os.path.isfile(os.path.join(tdir, d, "run.json"))),
                reverse=True,
            )
            if not dates:
                continue
            entry = {"slug": slug, "latest": dates[0], "run_count": len(dates)}
            try:
                with open(os.path.join(tdir, dates[0], "run.json"), encoding="utf-8") as fh:
                    data = json.load(fh)
                entry["label"] = data.get("label") or slug
                entry["buys"] = sum(1 for p in data.get("picks", []) if p.get("bucket") == "BUY")
            except (OSError, json.JSONDecodeError):
                entry["label"] = slug
            themes.append(entry)
    themes.sort(key=lambda t: t["latest"], reverse=True)
    return {"count": len(themes), "themes": themes}


@app.get("/funnel/themes/{slug}")
def get_theme_runs(slug: str) -> dict[str, Any]:
    """List a theme's run dates (newest first)."""
    if not _SLUG_RE.match(slug):
        raise HTTPException(status_code=400, detail="Invalid theme slug")
    tdir = os.path.join(_themes_dir(), slug)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail="Unknown theme")
    dates = sorted(
        (d for d in os.listdir(tdir) if os.path.isfile(os.path.join(tdir, d, "run.json"))),
        reverse=True,
    )
    return {"slug": slug, "runs": dates}


@app.get("/funnel/themes/{slug}/{date}")
def get_theme_run(slug: str, date: str) -> dict[str, Any]:
    """Return one theme run's full record (counts + picks + write-ups)."""
    if not _SLUG_RE.match(slug):
        raise HTTPException(status_code=400, detail="Invalid theme slug")
    if not _DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="Invalid date (expected YYYY-MM-DD)")
    record = os.path.join(_themes_dir(), slug, date, "run.json")
    if not os.path.isfile(record):
        raise HTTPException(status_code=404, detail="No run for that theme/date")
    with open(record, encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/funnel/{job_id}")
def get_funnel_job(job_id: str) -> dict[str, Any]:
    """Return a funnel job's status and, once done, its ranked picks + report paths."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return dict(job)


_SUMMARY_FIELDS = (
    "job_id", "kind", "status", "request", "created_at", "started_at", "finished_at", "error",
)


@app.get("/jobs")
def list_jobs(status: str | None = None) -> dict[str, Any]:
    """List all jobs (compact summaries, newest first).

    Pass ``?status=done`` (or queued/running/error) to filter. The full result
    body is omitted here to keep the list light — fetch a specific job via
    ``GET /analyze/{job_id}`` for its ``result``. ``done`` and ``error`` are the
    two terminal ("finished") states.
    """
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    if status is not None:
        jobs = [j for j in jobs if j["status"] == status]
    summaries = []
    for j in jobs:
        row = {k: j[k] for k in _SUMMARY_FIELDS if k in j}
        if j.get("status") == "done" and isinstance(j.get("result"), dict):
            row["decision"] = j["result"].get("decision")
        summaries.append(row)
    summaries.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"count": len(summaries), "jobs": summaries}


@app.get("/analyze/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Return a job's status and, once ``status == "done"``, its result."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return dict(job)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Modern browser UI for single-ticker multi-agent analysis + Google Finance quote card."""
    return _INDEX_HTML


@app.get("/funnel", response_class=HTMLResponse)
def funnel_page() -> str:
    """Browser UI for the screening funnel: submit a run, watch it, see the picks.

    Served on GET; the JSON API lives on POST /funnel and GET /funnel/{job_id},
    which this page calls. (Same path, different methods.)
    """
    return _FUNNEL_HTML


_SHARED_CSS = """
  :root {
    --bg: #0b0f17;
    --bg-elev: #111827;
    --bg-card: #161f30;
    --bg-input: #0f172a;
    --border: rgba(148, 163, 184, 0.16);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --buy-bg: rgba(16, 185, 129, 0.16);
    --buy-text: #34d399;
    --buy-border: rgba(16, 185, 129, 0.35);
    --hold-bg: rgba(148, 163, 184, 0.16);
    --hold-text: #cbd5e1;
    --hold-border: rgba(148, 163, 184, 0.35);
    --sell-bg: rgba(239, 68, 68, 0.16);
    --sell-text: #f87171;
    --sell-border: rgba(239, 68, 68, 0.35);
    --warn-bg: rgba(245, 158, 11, 0.16);
    --warn-text: #fbbf24;
    --gf-blue: #60a5fa;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f8fafc;
      --bg-elev: #ffffff;
      --bg-card: #ffffff;
      --bg-input: #f1f5f9;
      --border: rgba(15, 23, 42, 0.12);
      --text: #0f172a;
      --muted: #64748b;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --buy-bg: rgba(16, 185, 129, 0.12);
      --buy-text: #047857;
      --buy-border: rgba(16, 185, 129, 0.3);
      --hold-bg: rgba(100, 116, 139, 0.12);
      --hold-text: #334155;
      --hold-border: rgba(100, 116, 139, 0.3);
      --sell-bg: rgba(239, 68, 68, 0.12);
      --sell-text: #b91c1c;
      --sell-border: rgba(239, 68, 68, 0.3);
      --warn-bg: rgba(245, 158, 11, 0.12);
      --warn-text: #b45309;
      --gf-blue: #1a73e8;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }
  a { color: var(--gf-blue); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .topbar {
    border-bottom: 1px solid var(--border);
    background: var(--bg-elev);
    position: sticky;
    top: 0;
    z-index: 20;
    backdrop-filter: blur(10px);
  }
  .topbar-inner {
    max-width: 78rem;
    margin: 0 auto;
    padding: .85rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: .65rem;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .brand-badge {
    font-size: .72rem;
    font-weight: 600;
    padding: .15rem .5rem;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.15);
    color: var(--gf-blue);
    border: 1px solid rgba(59, 130, 246, 0.3);
  }
  .nav-pills { display: flex; gap: .4rem; align-items: center; }
  .nav-pill {
    padding: .42rem .85rem;
    border-radius: .5rem;
    font-size: .88rem;
    font-weight: 600;
    color: var(--muted);
    border: 1px solid transparent;
    transition: all .15s ease;
  }
  .nav-pill:hover { color: var(--text); background: var(--bg-input); text-decoration: none; }
  .nav-pill.active {
    color: var(--text);
    background: var(--bg-card);
    border-color: var(--border);
    box-shadow: 0 1px 2px rgba(0,0,0,.08);
  }
  .container {
    max-width: 78rem;
    margin: 1.5rem auto 3rem;
    padding: 0 1.25rem;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: .75rem;
    padding: 1.25rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  }
  .muted { color: var(--muted); font-size: .88rem; }
  label {
    display: block;
    margin: .5rem 0 .25rem;
    font-weight: 600;
    font-size: .82rem;
    text-transform: uppercase;
    letter-spacing: .04em;
    color: var(--muted);
  }
  input, select, button {
    font: inherit;
    font-size: .92rem;
    padding: .55rem .75rem;
    border-radius: .5rem;
    border: 1px solid var(--border);
    background: var(--bg-input);
    color: var(--text);
  }
  input:focus, select:focus {
    outline: 2px solid var(--accent);
    border-color: transparent;
  }
  input, select { width: 100%; }
  button.primary {
    background: var(--accent);
    color: #fff;
    border-color: transparent;
    font-weight: 600;
    cursor: pointer;
    transition: background .15s ease;
  }
  button.primary:hover { background: var(--accent-hover); }
  button.secondary {
    background: var(--bg-input);
    color: var(--text);
    font-weight: 600;
    cursor: pointer;
  }
  button.secondary:hover { border-color: var(--gf-blue); }
  .chip-row { display: flex; gap: .4rem; flex-wrap: wrap; margin-top: .55rem; }
  .chip {
    font-size: .78rem;
    padding: .22rem .6rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-input);
    color: var(--muted);
    cursor: pointer;
    user-select: none;
  }
  .chip:hover, .chip.active { color: var(--text); border-color: var(--gf-blue); }
  .gf-btn {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .35rem .7rem;
    border-radius: .45rem;
    font-size: .8rem;
    font-weight: 600;
    border: 1px solid rgba(59, 130, 246, 0.35);
    background: rgba(59, 130, 246, 0.1);
    color: var(--gf-blue);
    white-space: nowrap;
  }
  .gf-btn:hover { background: rgba(59, 130, 246, 0.18); text-decoration: none; }
  .tag {
    display: inline-block;
    padding: .18rem .6rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .02em;
  }
  .BUY, .OVERWEIGHT { background: var(--buy-bg); color: var(--buy-text); border: 1px solid var(--buy-border); }
  .HOLD, .NEUTRAL { background: var(--hold-bg); color: var(--hold-text); border: 1px solid var(--hold-border); }
  .SELL, .UNDERWEIGHT { background: var(--sell-bg); color: var(--sell-text); border: 1px solid var(--sell-border); }
  .UNKNOWN, .FAILED { background: var(--warn-bg); color: var(--warn-text); border: 1px solid rgba(245,158,11,.35); }
  .md-body { font-size: .93rem; line-height: 1.65; overflow-wrap: anywhere; }
  .md-body h1, .md-body h2, .md-body h3, .md-body h4 { margin: 1.1rem 0 .45rem; line-height: 1.3; }
  .md-body h2 { font-size: 1.15rem; border-bottom: 1px solid var(--border); padding-bottom: .3rem; }
  .md-body h3 { font-size: 1rem; }
  .md-body p { margin: .55rem 0; }
  .md-body ul, .md-body ol { margin: .5rem 0; padding-left: 1.4rem; }
  .md-body li { margin: .25rem 0; }
  .md-body code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .85em;
    background: rgba(148,163,184,.15);
    padding: .12rem .35rem;
    border-radius: .25rem;
  }
  .md-body pre {
    white-space: pre-wrap;
    background: var(--bg-input);
    border: 1px solid var(--border);
    padding: .85rem;
    border-radius: .5rem;
    overflow-x: auto;
  }
  .md-body table {
    border-collapse: collapse;
    width: 100%;
    margin: .75rem 0;
    font-size: .86rem;
  }
  .md-body th, .md-body td {
    border: 1px solid var(--border);
    padding: .45rem .65rem;
    text-align: left;
  }
  .md-body th { background: var(--bg-input); font-weight: 600; }
"""


_SHARED_MD_JS = """
function esc(s){
  return (s==null?'':String(s)).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function gfLink(ticker){
  const sym = encodeURIComponent(String(ticker||'').trim().toUpperCase());
  return 'https://www.google.com/finance?q=' + sym;
}
function inlineMd(text){
  let s = esc(text);
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
  s = s.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
  s = s.replace(/(https?:\\/\\/[^\\s<)]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  return s;
}
function renderMarkdown(md){
  if (!md) return '<p class="muted">No report content recorded for this section.</p>';
  const lines = String(md).replace(/\\r\\n/g, '\\n').split('\\n');
  let html = '';
  let inList = false;
  let inTable = false;
  let inCode = false;
  for (let i = 0; i < lines.length; i++){
    const raw = lines[i];
    const line = raw.trim();
    if (line.startsWith('```')){
      if (inList){ html += '</ul>'; inList = false; }
      if (inTable){ html += '</tbody></table>'; inTable = false; }
      html += inCode ? '</code></pre>' : '<pre><code>';
      inCode = !inCode;
      continue;
    }
    if (inCode){ html += esc(raw) + '\\n'; continue; }
    if (!line){
      if (inList){ html += '</ul>'; inList = false; }
      if (inTable){ html += '</tbody></table>'; inTable = false; }
      continue;
    }
    if (line.startsWith('|') && line.endsWith('|')){
      const cells = line.slice(1, -1).split('|').map(c => c.trim());
      if (cells.every(c => /^[:\\-]+$/.test(c))) continue;
      if (!inTable){
        if (inList){ html += '</ul>'; inList = false; }
        html += '<table><thead><tr>' + cells.map(c => '<th>' + inlineMd(c) + '</th>').join('') + '</tr></thead><tbody>';
        inTable = true;
      } else {
        html += '<tr>' + cells.map(c => '<td>' + inlineMd(c) + '</td>').join('') + '</tr>';
      }
      continue;
    }
    if (inTable){ html += '</tbody></table>'; inTable = false; }
    if (/^#{1,4}\\s+/.test(line)){
      if (inList){ html += '</ul>'; inList = false; }
      const lvl = Math.min(4, line.match(/^#+/)[0].length);
      html += '<h' + lvl + '>' + inlineMd(line.replace(/^#{1,4}\\s+/, '')) + '</h' + lvl + '>';
      continue;
    }
    if (/^[-*]\\s+/.test(line)){
      if (!inList){ html += '<ul>'; inList = true; }
      html += '<li>' + inlineMd(line.replace(/^[-*]\\s+/, '')) + '</li>';
      continue;
    }
    if (inList){ html += '</ul>'; inList = false; }
    html += '<p>' + inlineMd(line) + '</p>';
  }
  if (inList) html += '</ul>';
  if (inTable) html += '</tbody></table>';
  if (inCode) html += '</code></pre>';
  return '<div class="md-body">' + html + '</div>';
}
"""


_INDEX_HTML = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradingAgents · Multi-Agent Financial Terminal</title>
<style>"""
    + _SHARED_CSS
    + """
  .grid-main {
    display: grid;
    grid-template-columns: 22rem 1fr;
    gap: 1.25rem;
    align-items: start;
  }
  @media (max-width: 920px) { .grid-main { grid-template-columns: 1fr; } }
  .quote-banner {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
  }
  .quote-title { font-size: 1.25rem; font-weight: 700; margin: 0; }
  .quote-price { font-size: 1.5rem; font-weight: 800; font-variant-numeric: tabular-nums; }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
    gap: .6rem;
    margin-bottom: 1rem;
  }
  .stat-cell {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: .5rem;
    padding: .5rem .7rem;
  }
  .stat-cell .k { font-size: .73rem; color: var(--muted); text-transform: uppercase; }
  .stat-cell .v { font-size: .92rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: .1rem; }
  .verdict-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: .85rem 1rem;
    border-radius: .6rem;
    background: var(--bg-input);
    border: 1px solid var(--border);
    margin-bottom: 1rem;
  }
  .tabs {
    display: flex;
    gap: .35rem;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    padding-bottom: .6rem;
    margin-bottom: 1rem;
  }
  .tab-btn {
    padding: .4rem .75rem;
    border-radius: .45rem;
    font-size: .83rem;
    font-weight: 600;
    background: transparent;
    border: 1px solid transparent;
    color: var(--muted);
    cursor: pointer;
  }
  .tab-btn:hover { color: var(--text); background: var(--bg-input); }
  .tab-btn.active {
    color: var(--text);
    background: var(--bg-input);
    border-color: var(--gf-blue);
  }
  .job-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: .5rem .65rem;
    border-radius: .45rem;
    border: 1px solid var(--border);
    background: var(--bg-input);
    margin-top: .45rem;
    font-size: .83rem;
    cursor: pointer;
  }
  .job-item:hover { border-color: var(--gf-blue); }
</style></head><body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <span>⚡ TradingAgents</span>
      <span class="brand-badge">Multi-Agent Terminal</span>
    </div>
    <nav class="nav-pills">
      <a class="nav-pill active" href="/">Single-Ticker Analysis</a>
      <a class="nav-pill" href="/funnel">Screen a whole universe &rarr;</a>
    </nav>
  </div>
</header>

<main class="container">
  <div class="grid-main">
    <aside style="display:flex;flex-direction:column;gap:1.25rem">
      <section class="card">
        <h2 style="margin:0 0 .25rem;font-size:1.05rem">Run Deep Analysis</h2>
        <p class="muted" style="margin:0 0 .75rem">Deploys 12 specialized LLM analysts, researchers, trader &amp; risk managers.</p>

        <label for="ticker">Ticker Symbol</label>
        <input id="ticker" value="NVDA" placeholder="e.g. NVDA, AAPL, 0700.HK, BTC-USD">
        <div class="chip-row" id="presets">
          <span class="chip" data-t="NVDA" data-a="stock">NVDA</span>
          <span class="chip" data-t="AAPL" data-a="stock">AAPL</span>
          <span class="chip" data-t="MSFT" data-a="stock">MSFT</span>
          <span class="chip" data-t="TSLA" data-a="stock">TSLA</span>
          <span class="chip" data-t="0700.HK" data-a="stock">0700.HK</span>
          <span class="chip" data-t="BTC-USD" data-a="crypto">BTC-USD</span>
        </div>

        <label for="date">Analysis Date (YYYY-MM-DD)</label>
        <input id="date" type="date" value="2024-05-10">

        <label for="asset">Asset Class</label>
        <select id="asset">
          <option value="stock">Equity / ETF (stock)</option>
          <option value="crypto">Cryptocurrency (crypto)</option>
        </select>

        <div style="display:flex;gap:.5rem;margin-top:1rem">
          <button id="go" class="primary" style="flex:1">Analyze</button>
          <button id="quoteBtn" class="secondary" type="button" title="Refresh Google Finance Quote">Quote</button>
        </div>
      </section>

      <section class="card">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <h3 style="margin:0;font-size:.95rem">Recent Analysis Jobs</h3>
          <button id="refreshJobs" class="secondary" style="padding:.2rem .5rem;font-size:.75rem">Refresh</button>
        </div>
        <div id="jobsList" class="muted" style="margin-top:.5rem">No jobs in this session yet.</div>
      </section>
    </aside>

    <section class="card">
      <div id="quoteBox" class="quote-banner">
        <div>
          <div class="muted" id="qMeta">Instrument Snapshot</div>
          <h1 class="quote-title" id="qName">NVDA</h1>
        </div>
        <div style="display:flex;align-items:center;gap:.75rem;flex-wrap:wrap">
          <div class="quote-price" id="qPrice"></div>
          <a id="gfLinkBtn" class="gf-btn" href="https://www.google.com/finance?q=NVDA" target="_blank" rel="noopener">
            Google Finance &#8599;
          </a>
          <a id="yfLinkBtn" class="gf-btn" href="https://finance.yahoo.com/quote/NVDA" target="_blank" rel="noopener">
            Yahoo Finance &#8599;
          </a>
        </div>
      </div>
      <div id="quoteStats" class="stats-grid" style="display:none"></div>

      <div id="verdictBar" class="verdict-bar" style="display:none">
        <div>
          <span class="muted" style="font-size:.78rem;text-transform:uppercase;font-weight:600">Portfolio Manager Verdict</span>
          <div id="verdictMeta" style="font-weight:700;font-size:1.05rem;margin-top:.15rem"></div>
        </div>
        <span id="verdictBadge" class="tag HOLD">READY</span>
      </div>

      <div id="tabsBar" class="tabs" style="display:none">
        <button class="tab-btn active" data-tab="final_trade_decision">PM Final Decision</button>
        <button class="tab-btn" data-tab="market_report">Market &amp; Technicals</button>
        <button class="tab-btn" data-tab="sentiment_report">Sentiment</button>
        <button class="tab-btn" data-tab="news_report">News &amp; Macro</button>
        <button class="tab-btn" data-tab="fundamentals_report">Fundamentals</button>
        <button class="tab-btn" data-tab="investment_plan">Research Plan &amp; Debate</button>
        <button class="tab-btn" data-tab="trader_investment_plan">Trader &amp; Risk Debate</button>
      </div>

      <div id="out"><p class="muted">Ready. Select a ticker and click <strong>Analyze</strong> to run the multi-agent pipeline, or click <strong>Quote</strong> to inspect live Google Finance metrics.</p></div>
    </section>
  </div>
</main>

<script>"""
    + _SHARED_MD_JS
    + """
const $ = id => document.getElementById(id);
const out = $('out');
let currentReports = {};
let activeTab = 'final_trade_decision';

// Set default date to today in local YYYY-MM-DD if user opens fresh in browser
(function initDate(){
  const params = new URLSearchParams(window.location.search);
  if (params.get('ticker')) $('ticker').value = params.get('ticker').toUpperCase();
  if (params.get('date')) $('date').value = params.get('date');
  else {
    const d = new Date();
    $('date').value = d.toISOString().slice(0, 10);
  }
})();

function ratingClass(decision){
  const u = String(decision||'').toUpperCase();
  if (u.includes('BUY') || u.includes('OVERWEIGHT')) return 'BUY';
  if (u.includes('SELL') || u.includes('UNDERWEIGHT')) return 'SELL';
  if (u.includes('HOLD') || u.includes('NEUTRAL')) return 'HOLD';
  return 'UNKNOWN';
}

async function loadQuote(ticker){
  const sym = (ticker || $('ticker').value).trim().toUpperCase();
  if (!sym) return;
  $('qName').textContent = sym;
  $('gfLinkBtn').href = gfLink(sym);
  $('yfLinkBtn').href = 'https://finance.yahoo.com/quote/' + encodeURIComponent(sym);
  try {
    const r = await fetch('/quote/' + encodeURIComponent(sym));
    if (!r.ok) return;
    const q = await r.json();
    $('qName').textContent = (q.company_name && q.company_name !== q.ticker) ? `${q.company_name} (${q.ticker})` : q.ticker;
    const metaParts = [q.exchange, q.sector, q.industry].filter(Boolean);
    $('qMeta').textContent = metaParts.length ? metaParts.join(' · ') : 'Instrument Snapshot';
    $('qPrice').textContent = q.price || '';
    if (q.google_finance_url) $('gfLinkBtn').href = q.google_finance_url;
    if (q.yahoo_finance_url) $('yfLinkBtn').href = q.yahoo_finance_url;

    const stats = q.stats || {};
    const keys = ['Market cap', 'P/E ratio', 'Day range', 'Year range', 'Previous close', 'Avg Volume', 'Dividend yield'];
    const shown = keys.filter(k => stats[k]);
    if (shown.length){
      $('quoteStats').style.display = 'grid';
      $('quoteStats').innerHTML = shown.map(k =>
        `<div class="stat-cell"><div class="k">${esc(k)}</div><div class="v">${esc(stats[k])}</div></div>`
      ).join('');
    } else {
      $('quoteStats').style.display = 'none';
    }
  } catch (e) {}
}

function renderActiveTab(){
  let md = currentReports[activeTab] || '';
  if (activeTab === 'investment_plan' && currentReports.debate_history){
    md += '\\n\\n## Bull vs. Bear Research Debate\\n\\n' + currentReports.debate_history;
  }
  if (activeTab === 'trader_investment_plan' && currentReports.risk_history){
    md += '\\n\\n## Risk Management Team Debate\\n\\n' + currentReports.risk_history;
  }
  out.innerHTML = renderMarkdown(md);
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeTab = btn.getAttribute('data-tab');
    renderActiveTab();
  };
});

document.querySelectorAll('#presets .chip').forEach(chip => {
  chip.onclick = () => {
    $('ticker').value = chip.getAttribute('data-t');
    $('asset').value = chip.getAttribute('data-a') || 'stock';
    loadQuote($('ticker').value);
  };
});

$('quoteBtn').onclick = () => loadQuote($('ticker').value);
$('ticker').addEventListener('change', () => loadQuote($('ticker').value));

function displayJobResult(j, elapsedText){
  const res = j.result || {};
  const req = j.request || {};
  const decision = res.decision || 'COMPLETE';
  const rpt = res.reports || {};
  currentReports = {
    final_trade_decision: rpt.final_trade_decision || res.final_trade_decision || '',
    market_report: rpt.market_report || '',
    sentiment_report: rpt.sentiment_report || '',
    news_report: rpt.news_report || '',
    fundamentals_report: rpt.fundamentals_report || '',
    investment_plan: rpt.investment_plan || '',
    trader_investment_plan: rpt.trader_investment_plan || '',
    debate_history: rpt.debate_history || '',
    risk_history: rpt.risk_history || '',
  };
  $('verdictBar').style.display = 'flex';
  $('verdictMeta').textContent = `${req.ticker || $('ticker').value} · ${req.date || $('date').value}` + (elapsedText ? ` (${elapsedText})` : '');
  const badge = $('verdictBadge');
  badge.className = 'tag ' + ratingClass(decision);
  badge.textContent = decision;
  $('tabsBar').style.display = 'flex';
  renderActiveTab();
}

async function loadJobs(){
  try {
    const r = await fetch('/jobs');
    if (!r.ok) return;
    const {jobs} = await r.json();
    const an = (jobs || []).filter(j => (j.kind || 'analyze') === 'analyze').slice(0, 8);
    if (!an.length){ $('jobsList').textContent = 'No jobs in this session yet.'; return; }
    $('jobsList').innerHTML = an.map(j => {
      const req = j.request || {};
      const badge = j.status === 'done'
        ? `<span class="tag ${ratingClass(j.decision)}">${esc(j.decision || 'DONE')}</span>`
        : `<span class="muted">${esc(j.status)}</span>`;
      return `<div class="job-item" data-jid="${esc(j.job_id)}">`
        + `<div><strong>${esc(req.ticker || '?')}</strong> <span class="muted">${esc(req.date || '')}</span></div>`
        + badge + `</div>`;
    }).join('');
    $('jobsList').querySelectorAll('.job-item').forEach(el => {
      el.onclick = async () => {
        const jid = el.getAttribute('data-jid');
        const j = await (await fetch('/analyze/' + jid)).json();
        if (j.request && j.request.ticker){
          $('ticker').value = j.request.ticker;
          loadQuote(j.request.ticker);
        }
        if (j.status === 'done') displayJobResult(j);
        else if (j.status === 'error') out.textContent = 'Failed: ' + j.error;
      };
    });
  } catch (e) {}
}

$('refreshJobs').onclick = loadJobs;

$('go').onclick = async () => {
  const body = {
    ticker: $('ticker').value.trim().toUpperCase(),
    date: $('date').value.trim(),
    asset_type: $('asset').value,
  };
  loadQuote(body.ticker);
  $('verdictBar').style.display = 'none';
  $('tabsBar').style.display = 'none';
  out.textContent = 'Submitting...';
  const r = await fetch('/analyze', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
  if (!r.ok) { out.textContent = 'Error: ' + r.status + ' ' + await r.text(); return; }
  const {job_id} = await r.json();
  loadJobs();
  const t0 = Date.now();
  out.textContent = 'Running (job ' + job_id + ')...';
  while (true) {
    await new Promise(res => setTimeout(res, 4000));
    const j = await (await fetch('/analyze/' + job_id)).json();
    const mins = Math.floor((Date.now()-t0)/60000), secs = Math.floor((Date.now()-t0)/1000)%60;
    if (j.status === 'done') {
      displayJobResult(j, `${mins}m${secs}s`);
      loadJobs();
      break;
    }
    if (j.status === 'error') { out.textContent = 'Failed: ' + j.error; loadJobs(); break; }
    out.textContent = `Status: ${j.status} (${mins}m${secs}s) — running Analysts → Bull/Bear Debate → Trader → Risk Debate → Portfolio Manager...`;
  }
};

loadQuote($('ticker').value);
loadJobs();
</script>
</body></html>"""
)


_FUNNEL_HTML = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradingAgents · Screening funnel</title>
<style>"""
    + _SHARED_CSS
    + """
  h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
  h2 { font-size: 1.08rem; margin: 1.5rem 0 .6rem; }
  h3 { font-size: .98rem; margin: 1.25rem 0 .5rem; }
  .row { display: flex; gap: .85rem; flex-wrap: wrap; align-items: flex-end; }
  .row > div { flex: 1; min-width: 8rem; }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
    gap: .85rem;
    margin: 1rem 0;
  }
  .kpi-card {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: .65rem;
    padding: .85rem 1rem;
  }
  .kpi-card .stage { font-size: .74rem; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  .kpi-card .val { font-size: 1.55rem; font-weight: 800; font-variant-numeric: tabular-nums; margin-top: .15rem; }
  .kpi-card .sub { font-size: .78rem; color: var(--muted); margin-top: .1rem; }
  #status { margin: 1rem 0; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; margin-top: .5rem; font-size: .88rem; }
  th, td { text-align: left; padding: .55rem .6rem; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { font-size: .76rem; text-transform: uppercase; color: var(--muted); letter-spacing: .03em; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .tick { font-weight: 700; }
  .flag { color: var(--sell-text); font-size: .84rem; }
  .score-pill {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }
  .score-bar {
    width: 2.8rem;
    height: .38rem;
    background: rgba(148,163,184,.2);
    border-radius: 999px;
    overflow: hidden;
    display: inline-block;
  }
  .score-bar > span { display: block; height: 100%; background: var(--accent); }
  .paths { margin-top: 1.25rem; font-size: .84rem; }
  details { margin: .45rem 0; border: 1px solid var(--border); border-radius: .5rem; padding: .55rem .8rem; background: var(--bg-input); }
  summary { cursor: pointer; font-weight: 600; }
  .archive-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin-top: 1.5rem;
  }
  @media (max-width: 780px) { .archive-grid { grid-template-columns: 1fr; } }
  .archive-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: .5rem .7rem;
    border-radius: .45rem;
    border: 1px solid var(--border);
    background: var(--bg-input);
    margin-top: .45rem;
    font-size: .86rem;
  }
</style></head><body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <span>⚡ TradingAgents</span>
      <span class="brand-badge">Screening funnel</span>
    </div>
    <nav class="nav-pills">
      <a class="nav-pill" href="/">Single ticker &rarr;</a>
      <a class="nav-pill active" href="/funnel">Universe &amp; Theme Funnel</a>
    </nav>
  </div>
</header>

<main class="container">
  <section class="card">
    <h1>TradingAgents · Screening funnel</h1>
    <p class="muted" style="margin:0 0 1rem">Sources a universe, ranks it (quant screen), triages with a cheap LLM, then runs the full multi-agent analysis on the shortlist. The deep stage is slow &mdash; a run can take many minutes. &middot; <a href="/">Single ticker &rarr;</a></p>

    <div class="row">
      <div><label for="date">As-of date</label><input id="date" type="date" value="2024-05-10" placeholder="YYYY-MM-DD"></div>
      <div><label for="top_screen">Top screen (Stage 0)</label><input id="top_screen" type="number" min="1" placeholder="60"></div>
      <div><label for="top_triage">Top triage (Stage 1)</label><input id="top_triage" type="number" min="1" placeholder="8"></div>
      <div><label for="max_deep">Max deep (Stage 2)</label><input id="max_deep" type="number" min="1" placeholder="5"></div>
      <div style="flex:0 0 auto"><button id="go" class="primary">Run funnel</button></div>
    </div>

    <div style="margin-top:1.25rem;padding-top:1rem;border-top:1px solid var(--border)">
      <label for="theme" style="margin-top:0">Or run a theme — tickers auto-discovered from the name (no list needed)</label>
      <div class="row">
        <div style="flex:3"><input id="theme" placeholder="e.g. data center, cybersecurity, nuclear SMR"></div>
        <div style="flex:0 0 auto"><button id="goTheme" class="primary">Run theme</button></div>
      </div>
      <div class="chip-row" id="themePresets">
        <span class="chip" data-th="AI data center infrastructure">AI Data Center</span>
        <span class="chip" data-th="cybersecurity leaders">Cybersecurity</span>
        <span class="chip" data-th="nuclear SMR & uranium">Nuclear SMR</span>
        <span class="chip" data-th="semiconductor equipment">Semiconductors</span>
        <span class="chip" data-th="robotics & industrial automation">Robotics</span>
        <span class="chip" data-th="GLP-1 & metabolic biotech">GLP-1 Biotech</span>
      </div>
    </div>
  </section>

  <div id="status"></div>
  <div id="results"></div>

  <div class="archive-grid">
    <section class="card">
      <h2 style="margin-top:0">Themes</h2>
      <div id="themes" class="muted">Loading…</div>
    </section>
    <section class="card">
      <h2 style="margin-top:0">Past universe runs</h2>
      <div id="history" class="muted">Loading…</div>
    </section>
  </div>
</main>

<script>"""
    + _SHARED_MD_JS
    + """
const $ = id => document.getElementById(id);
const status = $('status'), results = $('results'), history = $('history'), themes = $('themes');
const BUCKETS = ['BUY','HOLD','SELL','UNKNOWN','FAILED'];
const TITLES = {BUY:'✅ Buy / Overweight — the picks', HOLD:'⏸️ Hold / Neutral', SELL:'❌ Sell / Avoid', UNKNOWN:'❔ Unclassified', FAILED:'⚠️ Failed to analyze'};

(function initDate(){
  const d = new Date();
  $('date').value = d.toISOString().slice(0, 10);
})();

function numOrBlank(v){ return (v==null||isNaN(v)) ? '' : Math.round(v); }
function scoreBar(v){
  if (v==null||isNaN(v)) return '';
  const n = Math.max(0, Math.min(100, Math.round(v)));
  return `<span class="score-pill"><span>${n}</span><span class="score-bar"><span style="width:${n}%"></span></span></span>`;
}
function tickerLinks(ticker, date){
  const sym = esc(ticker);
  const d = encodeURIComponent(date || $('date').value || '');
  return `<span class="tick">${sym}</span> `
    + `<a class="gf-btn" style="padding:.12rem .42rem;font-size:.72rem" href="${gfLink(ticker)}" target="_blank" rel="noopener" title="Open ${sym} on Google Finance">GF &#8599;</a> `
    + `<a class="gf-btn" style="padding:.12rem .42rem;font-size:.72rem" href="/?ticker=${encodeURIComponent(ticker)}&date=${d}" title="Deep Analyze ${sym}">Analyze &#8599;</a>`;
}

function render(result, heading){
  const picks = result.picks || [];
  const runDate = result.trade_date || $('date').value;
  const by = Object.fromEntries(BUCKETS.map(b => [b, []]));
  for (const p of picks) (by[p.bucket] || by.UNKNOWN).push(p);

  let html = '<section class="card" style="margin-top:1.25rem">';
  if (heading) html += `<h2 style="margin-top:0">${esc(heading)}</h2>`;

  html += `<div class="kpi-grid">`
    + `<div class="kpi-card"><div class="stage">Universe</div><div class="val">${result.universe_size ?? 0}</div><div class="sub">Candidate instruments</div></div>`
    + `<div class="kpi-card"><div class="stage">Stage 0 · Quant Screen</div><div class="val">${result.screened ?? 0}</div><div class="sub">Momentum + trend + liquidity</div></div>`
    + `<div class="kpi-card"><div class="stage">Stage 1 · LLM Triage</div><div class="val">${result.triaged ?? 0}</div><div class="sub">Conviction + thesis shortlist</div></div>`
    + `<div class="kpi-card"><div class="stage">Stage 2 · Deep Analyzed</div><div class="val">${result.analyzed ?? 0}</div><div class="sub">Full multi-agent write-ups</div></div>`
    + `</div>`;

  for (const b of BUCKETS){
    const rows = by[b]; if (!rows.length) continue;
    html += `<h3>${TITLES[b]}</h3><table><thead><tr>`;
    html += b==='FAILED'
      ? '<th>Ticker</th><th>Error</th></tr></thead><tbody>'
      : '<th>#</th><th>Ticker</th><th>Decision</th><th class="num">Triage</th><th class="num">Screen</th><th>Thesis</th><th>Red flag</th></tr></thead><tbody>';
    rows.forEach((p,i) => {
      html += b==='FAILED'
        ? `<tr><td>${tickerLinks(p.ticker, runDate)}</td><td>${esc(p.error)}</td></tr>`
        : `<tr><td class="num">${i+1}</td><td>${tickerLinks(p.ticker, runDate)}</td>`
          + `<td><span class="tag ${esc(p.bucket)}">${esc(p.decision||'')}</span></td>`
          + `<td class="num">${scoreBar(p.triage_score)}</td><td class="num">${scoreBar(p.screen_score)}</td>`
          + `<td>${esc(p.thesis)}</td><td class="flag">${esc(p.red_flag||'')}</td></tr>`;
    });
    html += '</tbody></table>';
  }

  const withReports = picks.filter(p => p.report);
  if (withReports.length){
    html += '<h3>Full Multi-Agent Write-Ups</h3>';
    for (const p of withReports)
      html += `<details><summary>${esc(p.ticker)} — <span class="tag ${esc(p.bucket)}">${esc(p.decision||'')}</span></summary><div style="margin-top:.65rem">${renderMarkdown(p.report)}</div></details>`;
  }

  // Stage 1 — triaged shortlist
  const tri = result.triaged_detail || [];
  if (tri.length){
    html += `<h3>Stage 1 — triaged shortlist (${tri.length})</h3><table><thead><tr>`
      + '<th>#</th><th>Ticker</th><th class="num">Triage</th><th class="num">Screen</th><th>Thesis</th><th>Red flag</th>'
      + '</tr></thead><tbody>';
    tri.forEach((t,i) => {
      html += `<tr><td class="num">${i+1}</td><td>${tickerLinks(t.ticker, runDate)}</td>`
        + `<td class="num">${scoreBar(t.triage_score)}</td><td class="num">${scoreBar(t.screen_score)}</td>`
        + `<td>${esc(t.thesis)}</td><td class="flag">${esc(t.red_flag||'')}</td></tr>`;
    });
    html += '</tbody></table>';
  }

  // Stage 0 — screened universe (quant ranking)
  const scr = result.screened_detail || [];
  if (scr.length){
    let t = '<table><thead><tr><th>#</th><th>Ticker</th><th class="num">Screen</th>'
      + '<th class="num">3m</th><th class="num">6m</th><th class="num">vs 200sma</th>'
      + '<th class="num">trend</th><th class="num">$vol(M)</th></tr></thead><tbody>';
    scr.forEach((s,i) => {
      const pct = v => (v==null||isNaN(v)) ? '' : (v*100).toFixed(0)+'%';
      t += `<tr><td class="num">${i+1}</td><td>${tickerLinks(s.ticker, runDate)}</td>`
        + `<td class="num">${numOrBlank(s.score)}</td><td class="num">${pct(s.ret_3m)}</td>`
        + `<td class="num">${pct(s.ret_6m)}</td><td class="num">${pct(s.above_200sma)}</td>`
        + `<td class="num">${s.trend_struct ?? ''}</td>`
        + `<td class="num">${s.dollar_vol==null?'':Math.round(s.dollar_vol/1e6)}</td></tr>`;
    });
    t += '</tbody></table>';
    html += `<h3>Stage 0 — screened universe (${scr.length})</h3>`
      + `<details><summary>Show ${scr.length} screened stocks (quant-ranked)</summary>${t}</details>`;
  }

  if (result.report_path) html += `<p class="paths muted">Saved to <code>${esc(result.report_path)}</code> (+ .csv, run.json)</p>`;
  html += '</section>';
  results.innerHTML = html;
  window.scrollTo({top: 0, behavior: 'smooth'});
}

async function loadHistory(){
  try {
    const {runs} = await (await fetch('/funnel/runs')).json();
    if (!runs.length){ history.textContent = 'No past runs yet.'; return; }
    history.innerHTML = runs.map(r =>
      `<div class="archive-item"><a href="#" data-date="${esc(r.date)}"><strong>${esc(r.date)}</strong></a>`
      + `<span class="muted">${r.analyzed ?? '?'} analyzed · <span class="tag BUY">${r.buys ?? 0} buys</span></span></div>`).join('');
    history.querySelectorAll('a[data-date]').forEach(a => a.onclick = async (e) => {
      e.preventDefault();
      const d = a.getAttribute('data-date');
      status.textContent = 'Loading ' + d + ' …';
      const rec = await (await fetch('/funnel/runs/' + d)).json();
      status.textContent = 'Showing run for ' + d;
      render(rec, 'Funnel run — ' + d);
    });
  } catch(err){ history.textContent = 'Could not load history.'; }
}

async function loadThemes(){
  try {
    const {themes:list} = await (await fetch('/funnel/themes')).json();
    if (!list.length){ themes.textContent = 'No theme runs yet — enter a theme above.'; return; }
    themes.innerHTML = list.map(t =>
      `<div class="archive-item"><a href="#" data-theme="${esc(t.slug)}"><strong>${esc(t.label||t.slug)}</strong></a>`
      + `<span class="muted">${esc(t.latest)} · <span class="tag BUY">${t.buys ?? 0} buys</span> (${t.run_count})</span></div>`).join('');
    themes.querySelectorAll('a[data-theme]').forEach(a => a.onclick = async (e) => {
      e.preventDefault();
      const slug = a.getAttribute('data-theme');
      const {runs} = await (await fetch('/funnel/themes/' + slug)).json();
      if (!runs.length) return;
      const rec = await (await fetch('/funnel/themes/' + slug + '/' + runs[0]).then(r=>r.json()));
      status.textContent = 'Showing ' + slug + ' — ' + runs[0];
      render(rec, (rec.label||slug) + ' — ' + runs[0]);
    });
  } catch(err){ themes.textContent = 'Could not load themes.'; }
}

document.querySelectorAll('#themePresets .chip').forEach(chip => {
  chip.onclick = () => { $('theme').value = chip.getAttribute('data-th'); };
});

async function pollJob(job_id, refresh){
  const t0 = Date.now();
  while (true) {
    await new Promise(res => setTimeout(res, 4000));
    const j = await (await fetch('/funnel/' + job_id)).json();
    const mins = Math.floor((Date.now()-t0)/60000), secs = Math.floor((Date.now()-t0)/1000)%60;
    if (j.status === 'done') { status.textContent = `Done in ${mins}m${secs}s`; render(j.result); refresh(); break; }
    if (j.status === 'error') { status.textContent = 'Failed: ' + j.error; break; }
    status.textContent = `Status: ${j.status} … (${mins}m${secs}s) — screening, triaging, then ~minutes per deep analysis`;
  }
}

async function submitRun(body, refresh){
  results.innerHTML = ''; status.textContent = 'Submitting…';
  const r = await fetch('/funnel', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
  if (!r.ok) { status.textContent = 'Error: ' + r.status + ' ' + await r.text(); return; }
  const {job_id} = await r.json();
  await pollJob(job_id, refresh);
}

$('go').onclick = () => {
  const body = { date: $('date').value.trim() };
  for (const k of ['top_screen','top_triage','max_deep']){ const v = $(k).value.trim(); if (v) body[k] = parseInt(v,10); }
  submitRun(body, loadHistory);
};

$('goTheme').onclick = () => {
  const theme = $('theme').value.trim();
  if (!theme){ status.textContent = 'Enter a theme name first.'; return; }
  const body = { date: $('date').value.trim(), theme };
  const md = $('max_deep').value.trim(); if (md) body.max_deep = parseInt(md,10);
  submitRun(body, loadThemes);
};

loadHistory();
loadThemes();
</script>
</body></html>"""
)


def main() -> None:
    """Console-script entry point: ``tradingagents-web``."""
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("TRADINGAGENTS_WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("TRADINGAGENTS_WEB_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
