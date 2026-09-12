"""Stock universe sourcing — where the batch pipeline's candidate list comes from.

The pipeline never asks a human to hand-type hundreds of tickers. Instead a
single env var selects *how* the universe is generated, so widening the net (or
narrowing it) is a one-line config change with no code edit:

    TRADINGAGENTS_UNIVERSE=sp500     # ~500 large caps  (default; trusted, liquid)
    TRADINGAGENTS_UNIVERSE=all_us    # ~5,000 US common stocks (Alpha Vantage)
    TRADINGAGENTS_UNIVERSE=file      # your own list (TRADINGAGENTS_UNIVERSE_FILE)

``get_universe()`` is the single entry point; the funnel's Stage 0 screen
consumes whatever list it returns. All sources return a clean, de-duplicated,
upper-cased list of equity tickers — downstream code stays identical regardless
of source, which is the whole point of the knob.
"""

from __future__ import annotations

import os
import re
from io import StringIO

import pandas as pd

from .alpha_vantage_common import _make_api_request

# S&P 500 constituents, maintained as a stable CSV (symbol,name,sector,...). Used
# for the default universe: a trusted, liquid 500-name set that's ideal for
# validating the funnel before widening to the full market. No API key needed.
_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)

# Exchanges we treat as the tradeable US common-stock market. Alpha Vantage's
# LISTING_STATUS also returns OTC / pink-sheet names, which are illiquid and
# data-poor — excluded here so they never reach the (token-spending) stages.
_US_EXCHANGES = frozenset({"NYSE", "NASDAQ"})

# A plausible common-equity ticker: 1-5 letters, optionally a single class
# suffix (BRK.B, BF-B). This is a cheap sanity filter, not a guarantee — the
# funnel's liquidity gate is the real junk filter. Warrants/units/rights, which
# carry extra suffix letters (…W, …WS, …U, …R), fail the length/shape check.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")

# Requests timeout for the non-Alpha-Vantage fetch (S&P 500 CSV); mirrors the
# vendor layer's 30s ceiling so a stalled fetch can't hang the batch run.
_REQUEST_TIMEOUT = 30


def _clean(symbols: list[str]) -> list[str]:
    """Upper-case, strip, drop non-ticker-shaped junk, de-dup (order-stable)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        if not isinstance(raw, str):
            continue
        sym = raw.strip().upper()
        if not _TICKER_RE.fullmatch(sym) or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _sp500_constituents() -> list[str]:
    """~500 S&P 500 tickers from the maintained constituents CSV."""
    df = pd.read_csv(_SP500_CSV_URL, storage_options={"User-Agent": "trading_agents"})
    # The dataset uses "Symbol"; fall back to the first column if it's renamed.
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    # BRK.B/BF.B ship with a dot; that's already a shape we accept.
    return _clean(df[col].astype(str).tolist())


def _alpha_vantage_listing() -> list[str]:
    """All active US common stocks on NYSE/NASDAQ via Alpha Vantage LISTING_STATUS.

    One API call returns every active US-listed security as CSV; we keep only
    ``assetType == Stock`` on a major exchange, which drops ETFs, funds, and
    OTC names in one pass.
    """
    # LISTING_STATUS defaults to state=active and always returns CSV.
    csv_text = _make_api_request("LISTING_STATUS", {})
    df = pd.read_csv(StringIO(csv_text))

    # Columns: symbol,name,exchange,assetType,ipoDate,delistingDate,status.
    mask = (
        df["assetType"].astype(str).str.strip().eq("Stock")
        & df["exchange"].astype(str).str.strip().str.upper().isin(_US_EXCHANGES)
    )
    return _clean(df.loc[mask, "symbol"].astype(str).tolist())


def _file_list() -> list[str]:
    """A user-supplied list from ``TRADINGAGENTS_UNIVERSE_FILE``.

    Accepts one ticker per line and/or comma-separated; blank lines and
    ``#`` comments are ignored, so a hand-kept watchlist works as-is.
    """
    path = os.getenv("TRADINGAGENTS_UNIVERSE_FILE")
    if not path:
        raise ValueError(
            "TRADINGAGENTS_UNIVERSE=file requires TRADINGAGENTS_UNIVERSE_FILE "
            "to point at a ticker list."
        )
    if not os.path.isfile(path):
        raise ValueError(f"Universe file not found: {path!r}")
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    tokens: list[str] = []
    for line in raw.splitlines():
        line = line.split("#", 1)[0]  # strip inline comments
        tokens.extend(part for part in line.replace(",", " ").split())
    return _clean(tokens)


# Registry of sources: source name -> the module-level fetcher's attribute name.
# Storing the name (not the function object) means dispatch resolves the current
# module attribute at call time, so a fetcher stays overridable in tests. Add a
# row to support a new universe (e.g. "nasdaq100") — get_universe() and every
# downstream caller pick it up with no other change.
_SOURCES = {
    "sp500": "_sp500_constituents",
    "all_us": "_alpha_vantage_listing",
    "file": "_file_list",
}


def get_universe() -> list[str]:
    """Return the candidate ticker universe selected by ``TRADINGAGENTS_UNIVERSE``.

    Defaults to ``sp500``. An optional ``TRADINGAGENTS_UNIVERSE_LIMIT`` caps the
    result (useful for a cheap dry run of the whole pipeline). Raises
    ``ValueError`` for an unknown source name.
    """
    source = os.getenv("TRADINGAGENTS_UNIVERSE", "sp500").strip().lower()
    fetch_name = _SOURCES.get(source)
    if fetch_name is None:
        raise ValueError(
            f"Unknown TRADINGAGENTS_UNIVERSE={source!r}. "
            f"Expected one of: {', '.join(sorted(_SOURCES))}."
        )
    symbols = globals()[fetch_name]()

    limit_raw = os.getenv("TRADINGAGENTS_UNIVERSE_LIMIT")
    if limit_raw:
        try:
            limit = int(limit_raw)
        except ValueError:
            raise ValueError(
                f"TRADINGAGENTS_UNIVERSE_LIMIT must be an integer, got {limit_raw!r}."
            ) from None
        if limit > 0:
            symbols = symbols[:limit]
    return symbols
