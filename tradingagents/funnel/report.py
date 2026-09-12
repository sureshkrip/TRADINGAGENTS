"""Stage 3 — turn the deep-analysis results into a ranked, readable report.

Produces two artifacts from a list of ``AnalysisResult``:
  * **Markdown** — tiered by signal (Buys first), ranked within each tier, with
    the thesis, red flag, and (optionally) the full write-up per name. This is
    the "which stock to pick" answer a human reads.
  * **CSV** — one flat row per name for spreadsheets / downstream tooling.

Ranking is decision-bucket first (actionable Buys on top), then triage
conviction, then the Stage 0 quant score as the tiebreaker — so quant and
LLM judgment both feed the final order.
"""

from __future__ import annotations

import csv
import io

from tradingagents.funnel.batch import AnalysisResult

# Signal buckets and the order they appear in the report. The graph's decision
# text is free-form (BUY / Overweight / "Conviction Buy" / HOLD / SELL / ...),
# so we classify by keyword rather than exact match.
_BUCKET_ORDER = ["BUY", "HOLD", "SELL", "UNKNOWN"]

_BUY_WORDS = ("buy", "overweight", "long", "accumulate", "bullish", "add")
_SELL_WORDS = ("sell", "underweight", "short", "reduce", "bearish", "avoid", "exit")
_HOLD_WORDS = ("hold", "neutral", "watch", "market perform", "market-weight", "equal")


def results_to_picks(results: list[AnalysisResult], *, include_reports: bool = False) -> list[dict]:
    """Flatten analysis results into JSON-friendly pick dicts.

    Shared by the web job result and the on-disk run record so both expose the
    same shape. ``include_reports`` adds the full write-up per pick (used for the
    persisted archive, omitted from the light live-poll payload).
    """
    picks: list[dict] = []
    for r in results:
        pick = {
            "ticker": r.ticker,
            "bucket": classify_decision(r.decision) if r.error is None else "FAILED",
            "decision": r.decision,
            "triage_score": r.triage_score,
            "screen_score": r.screen_score,
            "thesis": r.thesis,
            "red_flag": r.red_flag,
            "error": r.error,
        }
        if include_reports:
            pick["report"] = r.report
        picks.append(pick)
    return picks


def screened_to_dicts(screened: list) -> list[dict]:
    """Stage 0 survivors as JSON rows (ranked): score + the key quant metrics."""
    rows: list[dict] = []
    for r in screened:
        m = getattr(r, "metrics", {}) or {}
        rows.append({
            "ticker": r.ticker,
            "score": r.score,
            "ret_3m": m.get("ret_3m"),
            "ret_6m": m.get("ret_6m"),
            "above_200sma": m.get("above_200sma"),
            "trend_struct": m.get("trend_struct"),
            "dollar_vol": m.get("dollar_vol"),
        })
    return rows


def triaged_to_dicts(triaged: list) -> list[dict]:
    """Stage 1 shortlist as JSON rows: LLM conviction + thesis + red flag."""
    return [
        {
            "ticker": t.ticker,
            "triage_score": t.triage_score,
            "screen_score": t.screen_score,
            "thesis": t.thesis,
            "red_flag": t.red_flag,
        }
        for t in triaged
    ]


def classify_decision(decision: str | None) -> str:
    """Map a free-text decision to one of BUY / SELL / HOLD / UNKNOWN.

    Checks SELL before BUY so "reduce overweight"-style phrasing isn't misread
    as a buy; falls back to UNKNOWN when nothing matches (e.g. an empty or
    unexpected decision), which keeps such names visible rather than dropped.
    """
    if not decision:
        return "UNKNOWN"
    low = decision.lower()
    if any(w in low for w in _SELL_WORDS):
        return "SELL"
    if any(w in low for w in _BUY_WORDS):
        return "BUY"
    if any(w in low for w in _HOLD_WORDS):
        return "HOLD"
    return "UNKNOWN"


def _rank_key(r: AnalysisResult):
    """Sort within a bucket: higher triage conviction, then higher quant score."""
    return (r.triage_score, r.screen_score)


def _md_escape(text: str | None) -> str:
    """Make a cell safe for a Markdown table (escape pipes, flatten newlines)."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def build_report(
    results: list[AnalysisResult],
    trade_date: str,
    *,
    include_writeups: bool = True,
) -> tuple[str, str]:
    """Return ``(markdown, csv)`` for a batch of deep-analysis results."""
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    buckets: dict[str, list[AnalysisResult]] = {b: [] for b in _BUCKET_ORDER}
    for r in ok:
        buckets[classify_decision(r.decision)].append(r)
    for b in buckets:
        buckets[b].sort(key=_rank_key, reverse=True)

    md = _build_markdown(buckets, failed, trade_date, include_writeups)
    csv_text = _build_csv(results)
    return md, csv_text


def _build_markdown(buckets, failed, trade_date, include_writeups) -> str:
    n_ok = sum(len(v) for v in buckets.values())
    lines: list[str] = [
        f"# TradingAgents funnel report — {trade_date}",
        "",
        f"Analyzed **{n_ok}** stocks · "
        f"**{len(buckets['BUY'])}** buy · **{len(buckets['HOLD'])}** hold · "
        f"**{len(buckets['SELL'])}** sell"
        + (f" · **{len(failed)}** failed" if failed else ""),
        "",
    ]

    heading = {
        "BUY": "## ✅ Buy / Overweight — the picks",
        "HOLD": "## ⏸️ Hold / Neutral",
        "SELL": "## ❌ Sell / Avoid",
        "UNKNOWN": "## ❔ Unclassified",
    }
    for bucket in _BUCKET_ORDER:
        rows = buckets[bucket]
        if not rows:
            continue
        lines += [heading[bucket], ""]
        lines += [
            "| # | Ticker | Decision | Triage | Screen | Thesis | Red flag |",
            "|---|--------|----------|-------:|-------:|--------|----------|",
        ]
        for i, r in enumerate(rows, start=1):
            lines.append(
                f"| {i} | **{r.ticker}** | {_md_escape(r.decision)} | "
                f"{r.triage_score:.0f} | {r.screen_score:.0f} | "
                f"{_md_escape(r.thesis)} | {_md_escape(r.red_flag)} |"
            )
        lines.append("")

    if failed:
        lines += ["## ⚠️ Failed to analyze", ""]
        for r in failed:
            lines.append(f"- **{r.ticker}** — {_md_escape(r.error)}")
        lines.append("")

    if include_writeups:
        detailed = buckets["BUY"] + buckets["HOLD"] + buckets["SELL"] + buckets["UNKNOWN"]
        detailed = [r for r in detailed if r.report]
        if detailed:
            lines += ["---", "", "# Full write-ups", ""]
            for r in detailed:
                lines += [
                    f"## {r.ticker} — {r.decision or 'N/A'}",
                    "",
                    r.report.strip(),
                    "",
                ]

    return "\n".join(lines)


def _build_csv(results: list[AnalysisResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["ticker", "bucket", "decision", "triage_score", "screen_score",
         "thesis", "red_flag", "error"]
    )
    for r in results:
        writer.writerow([
            r.ticker,
            classify_decision(r.decision) if r.error is None else "FAILED",
            r.decision or "",
            r.triage_score,
            r.screen_score,
            r.thesis or "",
            r.red_flag or "",
            r.error or "",
        ])
    return buf.getvalue()
