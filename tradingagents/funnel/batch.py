"""Stage 2 — full multi-agent analysis over the triaged shortlist.

Runs the real ``TradingAgentsGraph`` (the ~25-min, many-LLM-call pipeline) on
each name Stage 1 handed up, and collects the decision plus the full write-up.
This is the only stage that spends serious tokens/time, which is exactly why the
funnel narrows so hard before reaching it.

Deliberate choices:
  * **One graph instance, reused across tickers** — matches the web service and
    the CLI, so agent memory/reflection carries across the batch.
  * **Serial by default** — the graph writes to shared on-disk dirs and reuses
    one instance, so serial execution avoids cross-run interference (same reason
    the web service defaults to one worker).
  * **Hard cap on deep runs** (``TRADINGAGENTS_BATCH_MAX_DEEP``) so cost can
    never run away regardless of how many names clear triage.
  * **Per-ticker isolation** — one failed analysis is recorded and the batch
    continues; it never sinks the whole run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from tradingagents.funnel.triage import TriageResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """A deep-analysis outcome for one ticker, plus its funnel provenance.

    ``decision`` is the graph's final signal (e.g. BUY / SELL / Overweight);
    ``report`` is the full final_trade_decision write-up. On failure both are
    None and ``error`` carries the reason, so the report stage can show which
    names couldn't be analyzed rather than silently dropping them.
    """

    ticker: str
    decision: str | None
    report: str | None
    error: str | None
    # Funnel provenance carried through for the final ranked report.
    triage_score: float
    screen_score: float
    thesis: str
    red_flag: str | None


def _build_graph():
    """Construct one TradingAgentsGraph from the ambient config (lazy import)."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())


def run_deep_analysis(
    candidates: list[TriageResult],
    trade_date: str,
    asset_type: str = "stock",
    *,
    max_deep: int | None = None,
    graph=None,
) -> list[AnalysisResult]:
    """Run the full multi-agent graph on each candidate; collect the decisions.

    Args:
        candidates: TriageResult shortlist from Stage 1 (already ranked).
        trade_date: as-of date ``YYYY-MM-DD`` passed to every analysis.
        asset_type: ``"stock"`` (default) or ``"crypto"``.
        max_deep: hard cap on analyses this run; defaults to
            ``TRADINGAGENTS_BATCH_MAX_DEEP`` (10). Extra candidates are skipped
            and logged, never silently dropped.
        graph: optional pre-built graph (injected in tests); built once from
            config when omitted.

    Results preserve the input order (i.e. triage rank).
    """
    if not candidates:
        return []

    if max_deep is None:
        raw = os.getenv("TRADINGAGENTS_BATCH_MAX_DEEP", "10")
        try:
            max_deep = int(raw)
        except ValueError:
            raise ValueError(
                f"TRADINGAGENTS_BATCH_MAX_DEEP must be an integer, got {raw!r}."
            ) from None

    to_run = candidates[:max_deep] if max_deep > 0 else candidates
    if len(candidates) > len(to_run):
        logger.info(
            "Deep-analysis cap: running %d of %d candidates (max_deep=%d); "
            "skipping %s",
            len(to_run), len(candidates), max_deep,
            [c.ticker for c in candidates[len(to_run):]],
        )

    if graph is None:
        graph = _build_graph()

    results: list[AnalysisResult] = []
    for i, cand in enumerate(to_run, start=1):
        logger.info("Deep analysis %d/%d: %s", i, len(to_run), cand.ticker)
        decision: str | None = None
        report: str | None = None
        error: str | None = None
        try:
            final_state, decision = graph.propagate(
                cand.ticker, trade_date, asset_type=asset_type
            )
            report = (final_state or {}).get("final_trade_decision")
        except Exception as exc:  # isolate one bad run from the rest of the batch
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("Deep analysis failed for %s: %s", cand.ticker, error)

        results.append(
            AnalysisResult(
                ticker=cand.ticker,
                decision=decision,
                report=report,
                error=error,
                triage_score=cand.triage_score,
                screen_score=cand.screen_score,
                thesis=cand.thesis,
                red_flag=cand.red_flag,
            )
        )
    return results
