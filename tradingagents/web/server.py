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

import os
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
<p class="muted">Runs the multi-agent pipeline for a ticker on a historical date. A run can take several minutes.</p>
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
