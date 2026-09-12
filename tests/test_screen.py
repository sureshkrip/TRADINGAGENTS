"""Stage 0 quant screen: metric computation, liquidity gate, and ranking.

The pure core (_ticker_metrics, score_frame) is tested with synthetic price
series — no network. screen_universe is tested with _download_prices stubbed,
so the whole path is exercised without hitting Yahoo.
"""

import numpy as np
import pandas as pd
import pytest

import tradingagents.dataflows.screen as sc
from tradingagents.dataflows.screen import ScreenResult, score_frame, screen_universe


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


@pytest.mark.unit
def test_ticker_metrics_none_when_too_short():
    close = _series(list(range(50)))  # < _MIN_ROWS
    assert sc._ticker_metrics(close, _series([1] * 50)) is None


@pytest.mark.unit
def test_ticker_metrics_uptrend_shape():
    # Steady uptrend: price above both SMAs -> trend_struct 1.0, positive momentum.
    close = _series([100 * (1.002 ** i) for i in range(260)])
    vol = _series([1_000_000] * 260)
    m = sc._ticker_metrics(close, vol)
    assert m is not None
    assert m["trend_struct"] == 1.0
    assert m["ret_3m"] > 0 and m["ret_6m"] > 0
    assert m["above_200sma"] > 0
    assert m["dollar_vol"] > 0


@pytest.mark.unit
def test_ticker_metrics_downtrend_below_200sma():
    close = _series([100 * (0.998 ** i) for i in range(260)])
    m = sc._ticker_metrics(close, _series([1_000_000] * 260))
    assert m["trend_struct"] == 0.0
    assert m["ret_3m"] < 0


def _metrics_row(price, dollar_vol, ret_3m, ret_6m, above_200, trend_struct, risk_adj):
    return {
        "price": price, "dollar_vol": dollar_vol, "ret_3m": ret_3m,
        "ret_6m": ret_6m, "above_200sma": above_200, "trend_struct": trend_struct,
        "vol_ann": 0.3, "risk_adj": risk_adj,
    }


@pytest.mark.unit
def test_liquidity_gate_drops_penny_and_illiquid():
    metrics = pd.DataFrame.from_dict(
        {
            "GOOD": _metrics_row(50, 20e6, 0.2, 0.3, 0.1, 1.0, 0.8),
            "PENNY": _metrics_row(2, 20e6, 0.9, 0.9, 0.5, 1.0, 2.0),      # price < $5
            "ILLIQUID": _metrics_row(50, 1e5, 0.9, 0.9, 0.5, 1.0, 2.0),  # < $5M/day
        },
        orient="index",
    )
    out = score_frame(metrics, 0.05, min_price=5, min_dollar_vol=5e6, top_n=10)
    assert [r.ticker for r in out] == ["GOOD"]


@pytest.mark.unit
def test_ranking_orders_by_composite_and_respects_top_n():
    # STRONG dominates every factor; WEAK trails; MID between.
    metrics = pd.DataFrame.from_dict(
        {
            "STRONG": _metrics_row(50, 20e6, 0.40, 0.60, 0.30, 1.0, 1.5),
            "MID": _metrics_row(50, 20e6, 0.10, 0.15, 0.05, 0.5, 0.5),
            "WEAK": _metrics_row(50, 20e6, -0.10, -0.05, -0.10, 0.0, -0.3),
        },
        orient="index",
    )
    ranked = score_frame(metrics, 0.05, min_price=5, min_dollar_vol=5e6, top_n=10)
    assert [r.ticker for r in ranked] == ["STRONG", "MID", "WEAK"]
    assert ranked[0].score >= ranked[1].score >= ranked[2].score
    # top_n truncates
    assert len(score_frame(metrics, 0.05, min_price=5, min_dollar_vol=5e6, top_n=2)) == 2


@pytest.mark.unit
def test_score_components_present_and_bounded():
    metrics = pd.DataFrame.from_dict(
        {"A": _metrics_row(50, 20e6, 0.2, 0.3, 0.1, 1.0, 0.8)}, orient="index"
    )
    r = score_frame(metrics, 0.05, min_price=5, min_dollar_vol=5e6, top_n=10)[0]
    assert set(r.components) == {"momentum", "trend", "rel_strength", "risk_adj"}
    assert 0.0 <= r.score <= 100.0
    assert "price" in r.metrics and "ret_3m" in r.metrics


@pytest.mark.unit
def test_empty_inputs():
    assert score_frame(pd.DataFrame(), None, min_price=5, min_dollar_vol=5e6, top_n=10) == []
    assert screen_universe([]) == []


@pytest.mark.unit
def test_yahoo_symbol_maps_class_shares():
    assert sc._yahoo_symbol("BRK.B") == "BRK-B"
    assert sc._yahoo_symbol("AAPL") == "AAPL"


@pytest.mark.unit
def test_download_prices_requeries_dot_shares_as_dash(monkeypatch):
    # Yahoo is queried with BRK-B; result must be re-keyed to the original BRK.B.
    captured = {}

    def fake_download(syms, **kwargs):
        captured["syms"] = syms
        cols = pd.MultiIndex.from_product([["BRK-B"], ["Close", "Volume"]])
        return pd.DataFrame(
            [[100.0, 1_000_000]] * 5,
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
            columns=cols,
        )

    monkeypatch.setattr("yfinance.download", fake_download)
    out = sc._download_prices(["BRK.B"], 400)
    assert captured["syms"] == ["BRK-B"]     # queried with the dashed symbol
    assert "BRK.B" in out                     # returned under the original ticker
    assert "Close" in out["BRK.B"].columns


@pytest.mark.unit
def test_screen_universe_end_to_end_with_stubbed_download(monkeypatch):
    def fake_download(tickers, lookback_days):
        out = {}
        for t in tickers:
            drift = 0.003 if t == "WIN" else (0.0005 if t == "SPY" else -0.001)
            close = _series(100 * np.exp(drift * np.arange(260)))
            df = pd.DataFrame({"Close": close, "Volume": _series([2_000_000] * 260)})
            out[t] = df
        return out

    monkeypatch.setattr(sc, "_download_prices", fake_download)
    monkeypatch.delenv("TRADINGAGENTS_SCREEN_TOP_N", raising=False)
    res = screen_universe(["WIN", "LOSE"], top_n=10)
    tickers = [r.ticker for r in res]
    assert "SPY" not in tickers          # benchmark is excluded from results
    assert tickers[0] == "WIN"           # stronger drift ranks first
    assert all(isinstance(r, ScreenResult) for r in res)
