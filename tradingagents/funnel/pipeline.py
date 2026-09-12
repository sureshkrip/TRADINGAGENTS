"""End-to-end funnel orchestrator: universe → screen → triage → analyze → report.

``run_funnel()`` chains all four stages and, by default, writes the ranked
Markdown + CSV report to disk. It's the single entry point a batch job or a
scheduled task calls; each stage's own knobs (env vars) still apply, and the
per-stage ``top_*`` counts can be overridden per call.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tradingagents.dataflows.screen import ScreenResult, screen_universe
from tradingagents.dataflows.universe import get_universe
from tradingagents.funnel.batch import AnalysisResult, run_deep_analysis
from tradingagents.funnel.report import build_report, results_to_picks
from tradingagents.funnel.triage import TriageResult, triage_candidates

logger = logging.getLogger(__name__)


@dataclass
class FunnelOutput:
    """Everything the funnel produced, for the caller to persist or inspect."""

    trade_date: str
    universe_size: int
    screened: list[ScreenResult] = field(default_factory=list)
    triaged: list[TriageResult] = field(default_factory=list)
    analyzed: list[AnalysisResult] = field(default_factory=list)
    report_md: str = ""
    report_csv: str = ""
    report_path: str | None = None  # Markdown file, if written
    csv_path: str | None = None
    run_json_path: str | None = None  # structured record for the history browser


def _default_out_dir(trade_date: str) -> str:
    """``<results_dir>/funnel/<trade_date>`` — sits alongside per-run agent logs."""
    from tradingagents.default_config import DEFAULT_CONFIG

    return os.path.join(DEFAULT_CONFIG["results_dir"], "funnel", trade_date)


def run_funnel(
    trade_date: str,
    asset_type: str = "stock",
    *,
    top_screen: int | None = None,
    top_triage: int | None = None,
    max_deep: int | None = None,
    write: bool = True,
    out_dir: str | None = None,
) -> FunnelOutput:
    """Run the full funnel for ``trade_date`` and return (and optionally write) the report.

    Args:
        trade_date: as-of date ``YYYY-MM-DD`` for the analysis.
        asset_type: ``"stock"`` (default) or ``"crypto"``.
        top_screen / top_triage / max_deep: per-stage cutoffs; each falls back to
            its stage's env-var default when None.
        write: write the Markdown + CSV report to ``out_dir`` (default on).
        out_dir: destination dir; defaults to ``<results_dir>/funnel/<date>``.

    Short-circuits cleanly: if a stage yields nothing (empty universe, nothing
    clears the screen/triage), later stages get an empty list and the report
    still renders — you get a report that says "0 picks" rather than an error.
    """
    universe = get_universe()
    logger.info("Funnel: universe=%d tickers", len(universe))

    screened = screen_universe(universe, top_n=top_screen)
    logger.info("Funnel: screened -> %d survivors", len(screened))

    triaged = triage_candidates(screened, top_n=top_triage)
    logger.info("Funnel: triaged -> %d shortlisted", len(triaged))

    analyzed = run_deep_analysis(triaged, trade_date, asset_type, max_deep=max_deep)
    logger.info("Funnel: analyzed -> %d deep reports", len(analyzed))

    report_md, report_csv = build_report(analyzed, trade_date)

    out = FunnelOutput(
        trade_date=trade_date,
        universe_size=len(universe),
        screened=screened,
        triaged=triaged,
        analyzed=analyzed,
        report_md=report_md,
        report_csv=report_csv,
    )

    if write:
        target = out_dir or _default_out_dir(trade_date)
        os.makedirs(target, exist_ok=True)
        out.report_path = os.path.join(target, "report.md")
        out.csv_path = os.path.join(target, "report.csv")
        out.run_json_path = os.path.join(target, "run.json")
        with open(out.report_path, "w", encoding="utf-8") as fh:
            fh.write(report_md)
        with open(out.csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(report_csv)
        # Structured record powers the web history browser (view any past day's
        # picks + write-ups). Persisted on the volume, so it survives restarts —
        # unlike the in-memory job store.
        record = {
            "trade_date": trade_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "universe_size": len(universe),
            "screened": len(screened),
            "triaged": len(triaged),
            "analyzed": len(analyzed),
            "picks": results_to_picks(analyzed, include_reports=True),
        }
        with open(out.run_json_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        logger.info("Funnel: report written to %s", out.report_path)

    return out


def main() -> None:
    """Console entry point (``tradingagents-funnel``) — run the funnel for a date.

    Defaults the date to today (UTC) so a scheduled/cron invocation needs no
    args; override with ``--date`` and the per-stage cutoffs. Prints the report
    path and a one-line summary.
    """
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Run the TradingAgents screening funnel.")
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        help="As-of trade date YYYY-MM-DD (default: today UTC).")
    parser.add_argument("--asset-type", default="stock", choices=["stock", "crypto"])
    parser.add_argument("--top-screen", type=int, default=None)
    parser.add_argument("--top-triage", type=int, default=None)
    parser.add_argument("--max-deep", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run_funnel(
        args.date,
        args.asset_type,
        top_screen=args.top_screen,
        top_triage=args.top_triage,
        max_deep=args.max_deep,
        write=True,
    )
    print(
        f"Funnel {args.date}: universe={out.universe_size} screened={len(out.screened)} "
        f"triaged={len(out.triaged)} analyzed={len(out.analyzed)}\nreport: {out.report_path}"
    )


if __name__ == "__main__":
    main()
