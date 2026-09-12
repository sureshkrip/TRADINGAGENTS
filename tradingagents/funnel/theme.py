"""Theme → tickers discovery.

Given only a theme *name* (e.g. "data center", "cybersecurity", "nuclear SMR"),
ask the cheap ``quick_think_llm`` which public companies are central to that
theme's value chain — so the funnel can screen a theme without any
hand-maintained ticker list. Works for arbitrary theme names, not a fixed set.

Discovered tickers are only *candidates*: the Stage 0 screen fetches prices for
each and silently drops anything with no market data, so a hallucinated or
delisted symbol never reaches the expensive stages.
"""

from __future__ import annotations

import logging
import os
import re

from tradingagents.funnel.triage import _build_quick_llm, _parse_json_array

logger = logging.getLogger(__name__)

# Same shape gate the universe/screen use: 1-5 letters + optional class suffix.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")

_PROMPT = (
    "You are an equity analyst. For the investment theme below, list the "
    "US-exchange-listed public companies most central to its core value chain "
    "(the businesses whose fortunes are genuinely driven by this theme — not "
    "tangential mega-caps). Prefer liquid common stocks.\n\n"
    "Return ONLY a JSON array of ticker symbols, e.g. [\"NVDA\",\"AMD\",\"VRT\"] "
    "— up to {n} tickers, correct US symbols, no commentary, no prose.\n\n"
    "Theme: {theme}"
)


def _extract_ticker(item) -> str | None:
    """Accept both ["NVDA", ...] and [{"ticker": "NVDA"}, ...] shapes."""
    if isinstance(item, dict):
        item = item.get("ticker") or item.get("symbol") or ""
    sym = str(item).strip().upper()
    return sym if _TICKER_RE.fullmatch(sym) else None


def expand_theme(theme_name: str, *, llm=None, max_tickers: int | None = None) -> list[str]:
    """Return candidate tickers for ``theme_name`` via the cheap LLM.

    Args:
        theme_name: free-text theme (e.g. "data center value chain").
        llm: optional pre-built chat model (injected in tests); built from
            config (``quick_think_llm``) when omitted.
        max_tickers: cap the list; defaults to ``TRADINGAGENTS_THEME_MAX_TICKERS``
            (30). The Stage 0 screen ranks these down further anyway.

    De-duplicated, upper-cased, ticker-shaped. Raises ValueError on an empty
    theme name; returns [] if the model yields nothing parseable.
    """
    if not theme_name or not theme_name.strip():
        raise ValueError("theme_name is required")

    if max_tickers is None:
        raw = os.getenv("TRADINGAGENTS_THEME_MAX_TICKERS", "30")
        try:
            max_tickers = int(raw)
        except ValueError:
            raise ValueError(
                f"TRADINGAGENTS_THEME_MAX_TICKERS must be an integer, got {raw!r}."
            ) from None

    llm = llm or _build_quick_llm()

    from langchain_core.messages import HumanMessage

    resp = llm.invoke([HumanMessage(content=_PROMPT.format(n=max_tickers, theme=theme_name.strip()))])
    content = getattr(resp, "content", resp)
    if isinstance(content, list):  # some providers return content parts
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )

    seen: set[str] = set()
    out: list[str] = []
    for item in _parse_json_array(str(content)):
        sym = _extract_ticker(item)
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)

    logger.info("Theme %r expanded to %d candidate tickers", theme_name, len(out))
    return out[:max_tickers]


def theme_slug(theme_name: str) -> str:
    """Filesystem-safe slug for a theme name: 'Data Center' -> 'data-center'."""
    return re.sub(r"[^a-z0-9]+", "-", theme_name.strip().lower()).strip("-") or "theme"
