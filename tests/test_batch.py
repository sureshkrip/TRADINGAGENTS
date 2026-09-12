"""Stage 2 deep-analysis orchestration: capture, cap, isolation, ordering.

A FakeGraph stands in for TradingAgentsGraph so no LLM/network runs. It records
which tickers it was asked to analyze and can be scripted to fail on specific
ones, exercising per-ticker error isolation.
"""

import pytest

from tradingagents.funnel.batch import run_deep_analysis
from tradingagents.funnel.triage import TriageResult


class FakeGraph:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.seen = []

    def propagate(self, ticker, trade_date, asset_type="stock"):
        self.seen.append((ticker, trade_date, asset_type))
        if ticker in self.fail_on:
            raise RuntimeError("boom")
        return {"final_trade_decision": f"writeup-{ticker}"}, f"BUY-{ticker}"


def _cand(ticker, triage=80.0, screen=70.0):
    return TriageResult(
        ticker=ticker, triage_score=triage, thesis=f"t-{ticker}",
        red_flag=None, screen_score=screen,
    )


@pytest.mark.unit
def test_empty_returns_empty():
    assert run_deep_analysis([], "2024-05-10") == []


@pytest.mark.unit
def test_captures_decision_and_report_in_order():
    cands = [_cand("AAA"), _cand("BBB")]
    g = FakeGraph()
    out = run_deep_analysis(cands, "2024-05-10", graph=g, max_deep=10)
    assert [r.ticker for r in out] == ["AAA", "BBB"]           # input order preserved
    assert out[0].decision == "BUY-AAA"
    assert out[0].report == "writeup-AAA"
    assert out[0].error is None
    assert g.seen[0] == ("AAA", "2024-05-10", "stock")


@pytest.mark.unit
def test_max_deep_caps_runs():
    cands = [_cand(f"T{i}") for i in range(5)]
    g = FakeGraph()
    out = run_deep_analysis(cands, "2024-05-10", graph=g, max_deep=2)
    assert [r.ticker for r in out] == ["T0", "T1"]  # only the top 2 by triage rank
    assert len(g.seen) == 2                         # graph never called for the rest


@pytest.mark.unit
def test_one_failure_does_not_sink_batch():
    cands = [_cand("AAA"), _cand("BAD"), _cand("CCC")]
    g = FakeGraph(fail_on={"BAD"})
    out = run_deep_analysis(cands, "2024-05-10", graph=g, max_deep=10)
    assert len(out) == 3
    bad = next(r for r in out if r.ticker == "BAD")
    assert bad.decision is None and bad.report is None
    assert "RuntimeError: boom" in bad.error
    # the others still succeeded
    assert next(r for r in out if r.ticker == "CCC").decision == "BUY-CCC"


@pytest.mark.unit
def test_provenance_carried_through():
    cands = [_cand("AAA", triage=91.0, screen=88.0)]
    out = run_deep_analysis(cands, "2024-05-10", graph=FakeGraph(), max_deep=10)
    assert out[0].triage_score == 91.0
    assert out[0].screen_score == 88.0
    assert out[0].thesis == "t-AAA"


@pytest.mark.unit
def test_asset_type_forwarded():
    g = FakeGraph()
    run_deep_analysis([_cand("BTC")], "2024-05-10", asset_type="crypto", graph=g, max_deep=1)
    assert g.seen[0][2] == "crypto"


@pytest.mark.unit
def test_max_deep_from_env(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_BATCH_MAX_DEEP", "1")
    g = FakeGraph()
    out = run_deep_analysis([_cand("AAA"), _cand("BBB")], "2024-05-10", graph=g)
    assert len(out) == 1 and out[0].ticker == "AAA"


@pytest.mark.unit
def test_bad_env_raises(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_BATCH_MAX_DEEP", "lots")
    with pytest.raises(ValueError, match="must be an integer"):
        run_deep_analysis([_cand("AAA")], "2024-05-10", graph=FakeGraph())
