"""Stage 1 — batched cheap-LLM triage.

Takes the ~60 survivors from the Stage 0 quant screen and, in **one LLM call per
batch** (not one per ticker — that batching is the whole point), asks the cheap
``quick_think_llm`` to assign each a conviction score, a one-line thesis, and a
red flag. Keeps the top N by conviction to hand to the expensive multi-agent
graph.

Why this stage exists: Stage 0 ranks on price/volume alone. A cheap LLM pass
adds qualitative judgment — what the company actually is, obvious risks, whether
the move looks durable — for a tiny fraction of a full analysis's cost, so the
~$$$ Stage 2 graph only ever runs on names that clear both a quant *and* a
judgment bar.

The model/provider/endpoint come from the same ``TRADINGAGENTS_*`` config the
rest of the app uses (``llm_provider`` + ``quick_think_llm``), so triage tracks
whatever quick model you've configured — no separate wiring.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from tradingagents.dataflows.screen import ScreenResult

logger = logging.getLogger(__name__)

# Candidates per LLM call. ~50 pre-screened names fit comfortably in one prompt
# with room for the JSON reply; larger universes just cost one extra call.
_BATCH_SIZE = 50


@dataclass
class TriageResult:
    """One triaged candidate: the LLM's conviction plus the Stage 0 evidence."""

    ticker: str
    triage_score: float  # 0..100 conviction from the LLM
    thesis: str
    red_flag: str | None
    screen_score: float  # carried through from Stage 0 for the final report


_SYSTEM_PROMPT = (
    "You are a buy-side analyst doing rapid triage on stocks that have ALREADY "
    "passed a quantitative momentum/trend screen. For each ticker, using the "
    "provided metrics and your own knowledge of the company, judge its "
    "attractiveness as a 1-3 month long position.\n\n"
    "Return ONLY a JSON array (no prose, no markdown fences). Each element:\n"
    '  {"ticker": "SYM", "score": 0-100, "thesis": "<=20 words", '
    '"red_flag": "<=12 words or null"}\n\n'
    "score = your conviction (100 = strongest). Base it on the metrics shown and "
    "what you know; do NOT invent numbers. Include EVERY ticker given, exactly "
    "once, using the same symbol. red_flag is the single biggest risk, or null."
)


def _format_candidates(batch: list[ScreenResult]) -> str:
    """Compact one-line-per-ticker table of the Stage 0 evidence."""
    lines = [
        "ticker | screen | ret_3m | ret_6m | vs_200sma | trend | $vol(M)",
    ]
    for r in batch:
        m = r.metrics
        lines.append(
            f"{r.ticker} | {r.score:.0f} | "
            f"{m.get('ret_3m', 0) * 100:+.0f}% | {m.get('ret_6m', 0) * 100:+.0f}% | "
            f"{m.get('above_200sma', 0) * 100:+.0f}% | {m.get('trend_struct', 0)} | "
            f"{m.get('dollar_vol', 0) / 1e6:.0f}"
        )
    return "\n".join(lines)


def _parse_json_array(content: str) -> list[dict]:
    """Extract a JSON array from a model reply, tolerating fences/prose.

    Returns [] if nothing parseable is found rather than raising — a single
    malformed batch shouldn't sink the whole run.
    """
    if not content:
        return []
    # Prefer a fenced block if present, else the outermost [...] span.
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    snippet = fence.group(1) if fence else None
    if snippet is None:
        start, end = content.find("["), content.rfind("]")
        if start == -1 or end <= start:
            logger.warning("Triage reply had no JSON array: %.120s", content)
            return []
        snippet = content[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:
        logger.warning("Triage JSON parse failed (%s): %.120s", exc, snippet)
        return []
    return data if isinstance(data, list) else []


def _build_quick_llm():
    """Construct the configured cheap LLM (same plumbing as the graph)."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients import create_llm_client

    cfg = DEFAULT_CONFIG
    client = create_llm_client(
        provider=cfg["llm_provider"],
        model=cfg["quick_think_llm"],
        base_url=cfg.get("backend_url"),
    )
    return client.get_llm()


def _triage_batch(llm, batch: list[ScreenResult]) -> dict[str, dict]:
    """Run one batch through the LLM; return {ticker: {score, thesis, red_flag}}.

    Hallucinated tickers (not in the batch) are dropped, so a stray symbol from
    the model can never smuggle itself into Stage 2.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    allowed = {r.ticker for r in batch}
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content="Triage these candidates:\n\n" + _format_candidates(batch)),
    ]
    response = llm.invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content parts
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)

    out: dict[str, dict] = {}
    for item in _parse_json_array(str(content)):
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        if ticker not in allowed or ticker in out:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        red = item.get("red_flag")
        red = None if red in (None, "", "null", "None") else str(red).strip()
        out[ticker] = {
            "score": max(0.0, min(100.0, score)),
            "thesis": str(item.get("thesis", "")).strip(),
            "red_flag": red,
        }
    return out


def triage_candidates(
    candidates: list[ScreenResult],
    *,
    top_n: int | None = None,
    llm=None,
) -> list[TriageResult]:
    """Score Stage 0 survivors with the cheap LLM and return the top ``top_n``.

    Args:
        candidates: ScreenResult list from ``screen_universe`` (Stage 0).
        top_n: shortlist size; defaults to ``TRADINGAGENTS_TRIAGE_TOP_N`` (8).
        llm: optional pre-built chat model (injected in tests); built from
            config when omitted.

    Tickers the model fails to score are dropped (they didn't earn a Stage 2
    slot). Ties break on the Stage 0 score, so quant conviction is the
    tiebreaker.
    """
    if not candidates:
        return []

    if top_n is None:
        raw = os.getenv("TRADINGAGENTS_TRIAGE_TOP_N", "8")
        try:
            top_n = int(raw)
        except ValueError:
            raise ValueError(f"TRADINGAGENTS_TRIAGE_TOP_N must be an integer, got {raw!r}.") from None

    if llm is None:
        llm = _build_quick_llm()

    screen_by_ticker = {r.ticker: r for r in candidates}
    scored: dict[str, dict] = {}
    for i in range(0, len(candidates), _BATCH_SIZE):
        batch = candidates[i : i + _BATCH_SIZE]
        try:
            scored.update(_triage_batch(llm, batch))
        except Exception as exc:  # one bad batch shouldn't kill the funnel
            logger.warning("Triage batch %d failed: %s", i // _BATCH_SIZE, exc)

    results = [
        TriageResult(
            ticker=t,
            triage_score=v["score"],
            thesis=v["thesis"],
            red_flag=v["red_flag"],
            screen_score=screen_by_ticker[t].score,
        )
        for t, v in scored.items()
    ]
    results.sort(key=lambda r: (r.triage_score, r.screen_score), reverse=True)
    return results[:top_n] if top_n > 0 else results
