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
# do not survive a restart. For durable/multi-replica needs, back this with the
# Redis instance the project already depends on.

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# Run analyses off the event loop. Default to one at a time: the graph writes to
# shared on-disk dirs (results/cache/memory) and reuses one graph instance, so
# serial execution avoids cross-run interference. Override with the env var.
_MAX_WORKERS = max(1, int(os.getenv("TRADINGAGENTS_WEB_WORKERS", "1")))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="ta-analyze")

_GRAPH: Any = None
_GRAPH_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _run_job(job_id: str, ticker: str, trade_date: str, asset_type: str) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(status="running", started_at=_now())
    try:
        graph = _get_graph()
        final_state, decision = graph.propagate(ticker, trade_date, asset_type=asset_type)
        result = {
            "decision": decision,
            "final_trade_decision": final_state.get("final_trade_decision"),
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
        from tradingagents.funnel.report import classify_decision

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


@app.post("/analyze", response_model=AnalyzeAccepted, status_code=202)
def analyze(req: AnalyzeRequest) -> AnalyzeAccepted:
    """Queue an analysis run. Poll ``GET /analyze/{job_id}`` for the result."""
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
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
    "job_id", "status", "request", "created_at", "started_at", "finished_at", "error",
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
    summaries = [{k: j[k] for k in _SUMMARY_FIELDS if k in j} for j in jobs]
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
    """Minimal browser UI: submit a ticker/date and poll for the decision."""
    return _INDEX_HTML


@app.get("/funnel", response_class=HTMLResponse)
def funnel_page() -> str:
    """Browser UI for the screening funnel: submit a run, watch it, see the picks.

    Served on GET; the JSON API lives on POST /funnel and GET /funnel/{job_id},
    which this page calls. (Same path, different methods.)
    """
    return _FUNNEL_HTML


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradingAgents</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 46rem; margin: 3rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; }
  label { display: block; margin: .75rem 0 .25rem; font-weight: 600; }
  input, select, button { font: inherit; padding: .5rem .6rem; }
  input, select { width: 100%; box-sizing: border-box; }
  button { margin-top: 1rem; cursor: pointer; }
  pre { white-space: pre-wrap; background: rgba(127,127,127,.12); padding: 1rem; border-radius: .5rem; margin-top: 1.5rem; }
  .muted { opacity: .7; font-size: .9rem; }
</style></head><body>
<h1>TradingAgents</h1>
<p class="muted">Runs the multi-agent pipeline for a ticker on a historical date. A run can take several minutes. &middot; <a href="/funnel">Screen a whole universe &rarr;</a></p>
<label for="ticker">Ticker</label>
<input id="ticker" value="NVDA">
<label for="date">Date (YYYY-MM-DD)</label>
<input id="date" value="2024-05-10">
<label for="asset">Asset type</label>
<select id="asset"><option value="stock">stock</option><option value="crypto">crypto</option></select>
<button id="go">Analyze</button>
<pre id="out">Ready.</pre>
<script>
const out = document.getElementById('out');
document.getElementById('go').onclick = async () => {
  const body = {
    ticker: document.getElementById('ticker').value.trim(),
    date: document.getElementById('date').value.trim(),
    asset_type: document.getElementById('asset').value,
  };
  out.textContent = 'Submitting...';
  const r = await fetch('/analyze', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
  if (!r.ok) { out.textContent = 'Error: ' + r.status + ' ' + await r.text(); return; }
  const {job_id} = await r.json();
  out.textContent = 'Running (job ' + job_id + ')...';
  while (true) {
    await new Promise(res => setTimeout(res, 4000));
    const j = await (await fetch('/analyze/' + job_id)).json();
    if (j.status === 'done') { out.textContent = 'Decision: ' + j.result.decision + '\\n\\n' + (j.result.final_trade_decision || ''); break; }
    if (j.status === 'error') { out.textContent = 'Failed: ' + j.error; break; }
    out.textContent = 'Status: ' + j.status + ' ...';
  }
};
</script>
</body></html>"""


_FUNNEL_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradingAgents · Funnel</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 3rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 1.75rem; }
  label { display: block; margin: .6rem 0 .2rem; font-weight: 600; font-size: .9rem; }
  input, button { font: inherit; padding: .5rem .6rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; }
  .row > div { flex: 1; min-width: 8rem; }
  input { width: 100%; box-sizing: border-box; }
  button { margin-top: 1rem; cursor: pointer; }
  .muted { opacity: .7; font-size: .9rem; }
  #status { margin: 1.25rem 0; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; margin-top: .5rem; font-size: .9rem; }
  th, td { text-align: left; padding: .4rem .5rem; border-bottom: 1px solid rgba(127,127,127,.25); vertical-align: top; }
  th { font-size: .8rem; text-transform: uppercase; opacity: .7; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .tick { font-weight: 700; }
  .flag { color: #c0392b; }
  .tag { display: inline-block; padding: .1rem .5rem; border-radius: .5rem; font-size: .8rem; font-weight: 600; }
  .BUY { background: rgba(39,174,96,.18); } .HOLD { background: rgba(127,127,127,.18); }
  .SELL { background: rgba(192,57,43,.18); } .UNKNOWN, .FAILED { background: rgba(241,196,15,.18); }
  .paths { margin-top: 1.5rem; font-size: .85rem; }
  h3 { font-size: 1rem; margin-top: 1.25rem; }
  details { margin: .4rem 0; }
  summary { cursor: pointer; font-weight: 600; }
  details pre { white-space: pre-wrap; background: rgba(127,127,127,.12); padding: .75rem; border-radius: .4rem; font-size: .85rem; margin-top: .4rem; }
  #history div { margin: .25rem 0; }
</style></head><body>
<h1>TradingAgents · Screening funnel</h1>
<p class="muted">Sources a universe, ranks it (quant screen), triages with a cheap LLM, then runs the full multi-agent analysis on the shortlist. The deep stage is slow &mdash; a run can take many minutes. &middot; <a href="/">Single ticker &rarr;</a></p>
<div class="row">
  <div><label for="date">As-of date</label><input id="date" value="2024-05-10" placeholder="YYYY-MM-DD"></div>
  <div><label for="top_screen">Top screen</label><input id="top_screen" type="number" min="1" placeholder="60"></div>
  <div><label for="top_triage">Top triage</label><input id="top_triage" type="number" min="1" placeholder="8"></div>
  <div><label for="max_deep">Max deep</label><input id="max_deep" type="number" min="1" placeholder="5"></div>
</div>
<button id="go">Run funnel</button>

<label for="theme" style="margin-top:1.5rem">Or run a theme — tickers auto-discovered from the name (no list needed)</label>
<div class="row">
  <div style="flex:3"><input id="theme" placeholder="e.g. data center, cybersecurity, nuclear SMR"></div>
  <div><button id="goTheme" style="margin-top:0">Run theme</button></div>
</div>

<div id="status"></div>
<div id="results"></div>

<h2>Themes</h2>
<div id="themes" class="muted">Loading…</div>

<h2>Past universe runs</h2>
<div id="history" class="muted">Loading…</div>

<script>
const $ = id => document.getElementById(id);
const status = $('status'), results = $('results'), history = $('history'), themes = $('themes');
const BUCKETS = ['BUY','HOLD','SELL','UNKNOWN','FAILED'];
const TITLES = {BUY:'✅ Buy / Overweight — the picks', HOLD:'⏸️ Hold / Neutral', SELL:'❌ Sell / Avoid', UNKNOWN:'❔ Unclassified', FAILED:'⚠️ Failed to analyze'};

function esc(s){ return (s==null?'':String(s)).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function numOrBlank(v){ return (v==null||isNaN(v)) ? '' : Math.round(v); }

function render(result, heading){
  const picks = result.picks || [];
  const by = Object.fromEntries(BUCKETS.map(b => [b, []]));
  for (const p of picks) (by[p.bucket] || by.UNKNOWN).push(p);
  let html = heading ? `<h2>${esc(heading)}</h2>` : '';
  html += `<p class="muted">universe ${result.universe_size} → screened ${result.screened} → triaged ${result.triaged} → analyzed ${result.analyzed}</p>`;
  for (const b of BUCKETS){
    const rows = by[b]; if (!rows.length) continue;
    html += `<h3>${TITLES[b]}</h3><table><thead><tr>`;
    html += b==='FAILED'
      ? '<th>Ticker</th><th>Error</th></tr></thead><tbody>'
      : '<th>#</th><th>Ticker</th><th>Decision</th><th class="num">Triage</th><th class="num">Screen</th><th>Thesis</th><th>Red flag</th></tr></thead><tbody>';
    rows.forEach((p,i) => {
      html += b==='FAILED'
        ? `<tr><td class="tick">${esc(p.ticker)}</td><td>${esc(p.error)}</td></tr>`
        : `<tr><td class="num">${i+1}</td><td class="tick">${esc(p.ticker)}</td>`
          + `<td><span class="tag ${esc(p.bucket)}">${esc(p.decision||'')}</span></td>`
          + `<td class="num">${numOrBlank(p.triage_score)}</td><td class="num">${numOrBlank(p.screen_score)}</td>`
          + `<td>${esc(p.thesis)}</td><td class="flag">${esc(p.red_flag||'')}</td></tr>`;
    });
    html += '</tbody></table>';
  }
  const withReports = picks.filter(p => p.report);
  if (withReports.length){
    html += '<h3>Full write-ups</h3>';
    for (const p of withReports)
      html += `<details><summary>${esc(p.ticker)} — ${esc(p.decision||'')}</summary><pre>${esc(p.report)}</pre></details>`;
  }
  if (result.report_path) html += `<p class="paths muted">Saved to <code>${esc(result.report_path)}</code> (+ .csv, run.json)</p>`;
  results.innerHTML = html;
  window.scrollTo(0,0);
}

async function loadHistory(){
  try {
    const {runs} = await (await fetch('/funnel/runs')).json();
    if (!runs.length){ history.textContent = 'No past runs yet.'; return; }
    history.innerHTML = runs.map(r =>
      `<div><a href="#" data-date="${esc(r.date)}">${esc(r.date)}</a>`
      + ` <span class="muted">— ${r.analyzed ?? '?'} analyzed, ${r.buys ?? 0} buys</span></div>`).join('');
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
      `<div><a href="#" data-theme="${esc(t.slug)}">${esc(t.label||t.slug)}</a>`
      + ` <span class="muted">— latest ${esc(t.latest)}, ${t.buys ?? 0} buys, ${t.run_count} run(s)</span></div>`).join('');
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
