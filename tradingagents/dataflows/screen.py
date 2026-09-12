"""Stage 0 — the cheap, no-LLM quant screen (top of the funnel).

Takes the raw universe (hundreds–thousands of tickers from ``get_universe()``)
and ranks it down to a shortlist using only price/volume data, spending **zero
LLM tokens**. This is what makes analyzing a large universe tractable: the
expensive multi-agent graph only ever sees the handful of names that survive
here.

Design choices that keep it cheap and fast:
  * **Bulk price download.** All OHLCV comes from ``yfinance.download`` in
    batched requests (a few calls for the whole universe), not one API call per
    ticker. No fundamentals lookups (those are per-ticker and rate-limited) —
    valuation/quality is Stage 1's job on the much smaller survivor set.
  * **Cross-sectional percentile scoring.** Each factor is ranked *relative to
    the rest of the universe* (0..1), so the composite is robust to scale and
    outliers and the factor weights actually mean something.
  * **Liquidity is a hard gate, everything else is scored.** Untradeable names
    (penny/illiquid) are dropped outright; the rest get a 0–100 composite so you
    always get a rankable list, never an empty one.

Factors (weights sum to 1.0):
    momentum 0.35  |  trend 0.25  |  relative strength 0.20  |  risk-adjusted 0.20
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from .stockstats_utils import yf_retry

# Trading-day lookbacks. ~252 trading days ≈ 1 year; 63 ≈ 3 months, 126 ≈ 6.
_TD_3M = 63
_TD_6M = 126
_TD_YEAR = 252

# Minimum history to score a name at all — below this the trend/momentum
# windows are too short to be meaningful, so the ticker is dropped.
_MIN_ROWS = 90

# Factor weights (must sum to 1.0). Momentum-led, with trend confirmation, a
# benchmark-relative kicker, and a risk-adjustment so smooth movers beat
# lottery tickets with the same raw return.
_WEIGHTS = {"momentum": 0.35, "trend": 0.25, "rel_strength": 0.20, "risk_adj": 0.20}

# Yahoo can be flaky on very large ticker lists; fetch in chunks this size.
_DOWNLOAD_CHUNK = 100


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from None


@dataclass
class ScreenResult:
    """One ranked ticker with its composite score and the evidence behind it.

    ``components`` holds the four 0..1 factor sub-scores; ``metrics`` holds the
    human-readable raw numbers (price, dollar volume, returns, trend posture) so
    a report — or Stage 1's triage prompt — can explain *why* a name ranked
    where it did without recomputing anything.
    """

    ticker: str
    score: float  # 0..100 composite
    components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def _ticker_metrics(close: pd.Series, volume: pd.Series) -> dict[str, float] | None:
    """Raw per-ticker metrics from one clean OHLCV history, or None if too short.

    Pure: no network, no cross-sectional context. Cross-sectional ranking and
    the liquidity gate are applied later, over the whole surviving frame.
    """
    close = close.dropna()
    volume = volume.reindex(close.index).fillna(0)
    if len(close) < _MIN_ROWS:
        return None

    price = float(close.iloc[-1])
    if price <= 0:
        return None

    def ret(window: int) -> float:
        if len(close) <= window:
            return float("nan")
        return float(close.iloc[-1] / close.iloc[-1 - window] - 1.0)

    ret_3m, ret_6m = ret(_TD_3M), ret(_TD_6M)

    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(min(len(close), _TD_YEAR)).mean())
    above_200 = price / sma200 - 1.0 if sma200 > 0 else float("nan")
    # Trend structure: 1.0 = price>50>200 (clean uptrend), 0.5 = above 200 only,
    # 0.0 = below the long-term average.
    if price > sma50 > sma200:
        trend_struct = 1.0
    elif price > sma200:
        trend_struct = 0.5
    else:
        trend_struct = 0.0

    daily = close.pct_change().dropna()
    vol_ann = float(daily.tail(_TD_3M).std() * (252 ** 0.5)) if len(daily) else float("nan")
    # Risk-adjusted momentum: reward return earned smoothly over jumpy return.
    risk_adj = ret_3m / vol_ann if vol_ann and vol_ann > 0 else float("nan")

    dollar_vol = float((close * volume).tail(20).mean())

    return {
        "price": price,
        "dollar_vol": dollar_vol,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "above_200sma": above_200,
        "trend_struct": trend_struct,
        "vol_ann": vol_ann,
        "risk_adj": risk_adj,
    }


def _pct_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0,1]; NaNs rank at the bottom (0)."""
    return s.rank(pct=True, na_option="bottom")


def score_frame(
    metrics: pd.DataFrame,
    benchmark_ret_3m: float | None = None,
    *,
    min_price: float,
    min_dollar_vol: float,
    top_n: int,
) -> list[ScreenResult]:
    """Apply the liquidity gate, compute the composite, and return the top N.

    ``metrics`` is one row per ticker (index = ticker) as produced by
    ``_ticker_metrics``. Pure and deterministic — this is the unit-tested core;
    all network I/O lives in ``screen_universe``.
    """
    if metrics.empty:
        return []

    # --- Liquidity gate (hard filter) --------------------------------------
    gated = metrics[
        (metrics["price"] >= min_price) & (metrics["dollar_vol"] >= min_dollar_vol)
    ].copy()
    if gated.empty:
        return []

    # --- Factor sub-scores (cross-sectional, 0..1) -------------------------
    momentum = (_pct_rank(gated["ret_3m"]) + _pct_rank(gated["ret_6m"])) / 2.0
    trend = (_pct_rank(gated["above_200sma"]) + gated["trend_struct"]) / 2.0
    if benchmark_ret_3m is not None:
        rel = _pct_rank(gated["ret_3m"] - benchmark_ret_3m)
    else:
        rel = _pct_rank(gated["ret_3m"])  # no benchmark -> fall back to raw momentum
    risk_adj = _pct_rank(gated["risk_adj"])

    composite = (
        _WEIGHTS["momentum"] * momentum
        + _WEIGHTS["trend"] * trend
        + _WEIGHTS["rel_strength"] * rel
        + _WEIGHTS["risk_adj"] * risk_adj
    ) * 100.0

    results: list[ScreenResult] = []
    for ticker in gated.index:
        results.append(
            ScreenResult(
                ticker=str(ticker),
                score=round(float(composite[ticker]), 2),
                components={
                    "momentum": round(float(momentum[ticker]), 4),
                    "trend": round(float(trend[ticker]), 4),
                    "rel_strength": round(float(rel[ticker]), 4),
                    "risk_adj": round(float(risk_adj[ticker]), 4),
                },
                metrics={k: round(float(gated.loc[ticker, k]), 4) for k in gated.columns},
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n] if top_n > 0 else results


def _download_prices(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Bulk-download recent OHLCV for many tickers, chunked and retried.

    Returns ``{ticker: DataFrame}`` (auto-adjusted close). Missing/delisted
    names are simply absent. Isolated here so tests can stub the network out.
    """
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    period = f"{max(lookback_days, _MIN_ROWS) + 5}d"
    for i in range(0, len(tickers), _DOWNLOAD_CHUNK):
        chunk = tickers[i : i + _DOWNLOAD_CHUNK]
        data = yf_retry(
            lambda c=chunk: yf.download(
                c,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        )
        if data is None or data.empty:
            continue
        # Single-ticker downloads come back with flat columns; multi-ticker with
        # a (ticker, field) MultiIndex. Normalize both to per-ticker frames.
        if isinstance(data.columns, pd.MultiIndex):
            for t in chunk:
                if t in data.columns.get_level_values(0):
                    out[t] = data[t].dropna(how="all")
        else:
            out[chunk[0]] = data.dropna(how="all")
    return out


def screen_universe(
    tickers: list[str],
    *,
    top_n: int | None = None,
) -> list[ScreenResult]:
    """Rank a universe down to the highest-scoring ``top_n`` names (no LLM).

    Knobs (env vars, all optional):
        TRADINGAGENTS_SCREEN_TOP_N            shortlist size          (default 60)
        TRADINGAGENTS_SCREEN_MIN_PRICE        liquidity gate, $/share (default 5)
        TRADINGAGENTS_SCREEN_MIN_DOLLAR_VOL   liquidity gate, $/day   (default 5e6)
        TRADINGAGENTS_SCREEN_BENCHMARK        relative-strength ref   (default SPY)
        TRADINGAGENTS_SCREEN_LOOKBACK_DAYS    calendar days of history(default 400)
    """
    if not tickers:
        return []

    top_n = top_n if top_n is not None else _env_int("TRADINGAGENTS_SCREEN_TOP_N", 60)
    min_price = _env_float("TRADINGAGENTS_SCREEN_MIN_PRICE", 5.0)
    min_dollar_vol = _env_float("TRADINGAGENTS_SCREEN_MIN_DOLLAR_VOL", 5_000_000.0)
    benchmark = os.getenv("TRADINGAGENTS_SCREEN_BENCHMARK", "SPY").strip().upper()
    lookback_days = _env_int("TRADINGAGENTS_SCREEN_LOOKBACK_DAYS", 400)

    # One bulk fetch for the universe + the benchmark.
    fetch = sorted(set(tickers) | {benchmark})
    frames = _download_prices(fetch, lookback_days)

    # Benchmark 3-month return for the relative-strength factor (None if absent).
    benchmark_ret_3m: float | None = None
    bench_df = frames.pop(benchmark, None)
    if bench_df is not None and "Close" in bench_df:
        bm = _ticker_metrics(bench_df["Close"], bench_df.get("Volume", pd.Series(dtype=float)))
        if bm is not None:
            benchmark_ret_3m = bm["ret_3m"]

    rows: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        df = frames.get(ticker)
        if df is None or "Close" not in df:
            continue
        m = _ticker_metrics(df["Close"], df.get("Volume", pd.Series(dtype=float)))
        if m is not None:
            rows[ticker] = m

    metrics = pd.DataFrame.from_dict(rows, orient="index")
    return score_frame(
        metrics,
        benchmark_ret_3m,
        min_price=min_price,
        min_dollar_vol=min_dollar_vol,
        top_n=top_n,
    )
