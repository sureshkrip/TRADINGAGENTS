"""Stage 3 report: decision classification, tiering/ranking, CSV, and the
end-to-end run_funnel orchestrator (with each stage stubbed — no LLM/network)."""

import csv
import io

import pytest

import tradingagents.funnel.pipeline as pl
from tradingagents.funnel.batch import AnalysisResult
from tradingagents.funnel.report import build_report, classify_decision


def _res(ticker, decision, triage=80.0, screen=70.0, error=None, report="body"):
    return AnalysisResult(
        ticker=ticker, decision=decision, report=report, error=error,
        triage_score=triage, screen_score=screen,
        thesis=f"thesis-{ticker}", red_flag=None,
    )


@pytest.mark.unit
@pytest.mark.parametrize("text,bucket", [
    ("BUY", "BUY"),
    ("Overweight", "BUY"),
    ("Conviction Buy", "BUY"),
    ("SELL", "SELL"),
    ("Underweight", "SELL"),
    ("reduce position", "SELL"),
    ("HOLD", "HOLD"),
    ("Neutral", "HOLD"),
    ("", "UNKNOWN"),
    (None, "UNKNOWN"),
    ("banana", "UNKNOWN"),
])
def test_classify_decision(text, bucket):
    assert classify_decision(text) == bucket


@pytest.mark.unit
def test_stage_detail_helpers():
    from tradingagents.dataflows.screen import ScreenResult
    from tradingagents.funnel.report import screened_to_dicts, triaged_to_dicts
    from tradingagents.funnel.triage import TriageResult

    scr = screened_to_dicts([ScreenResult("AAA", 90.0, {}, {"ret_3m": 0.2, "dollar_vol": 2e7})])
    assert scr[0]["ticker"] == "AAA" and scr[0]["score"] == 90.0 and scr[0]["ret_3m"] == 0.2

    tri = triaged_to_dicts([TriageResult("AAA", 80.0, "th", "rf", 90.0)])
    assert tri[0] == {"ticker": "AAA", "triage_score": 80.0, "screen_score": 90.0,
                      "thesis": "th", "red_flag": "rf"}


@pytest.mark.unit
def test_report_tiers_and_ranks_buys_first():
    results = [
        _res("LOW", "BUY", triage=60),
        _res("HIGH", "BUY", triage=95),
        _res("HELD", "HOLD", triage=99),   # higher triage but HOLD -> below buys
    ]
    md, _ = build_report(results, "2024-05-10", include_writeups=False)
    # Buy section precedes Hold section
    assert md.index("Buy / Overweight") < md.index("Hold / Neutral")
    # Within buys, HIGH ranks above LOW
    assert md.index("**HIGH**") < md.index("**LOW**")


@pytest.mark.unit
def test_report_lists_failures_separately():
    results = [_res("OK", "BUY"), _res("BAD", None, error="OpenAIError: 404")]
    md, _ = build_report(results, "2024-05-10")
    assert "Failed to analyze" in md
    assert "BAD" in md and "404" in md


@pytest.mark.unit
def test_markdown_escapes_pipes_in_cells():
    r = _res("X", "BUY")
    r.thesis = "cheap | growing fast"
    md, _ = build_report([r], "2024-05-10", include_writeups=False)
    assert "cheap \\| growing fast" in md


@pytest.mark.unit
def test_csv_has_row_per_result_with_buckets():
    results = [_res("A", "BUY"), _res("B", None, error="boom")]
    _, csv_text = build_report(results, "2024-05-10")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [r["ticker"] for r in rows] == ["A", "B"]
    assert rows[0]["bucket"] == "BUY"
    assert rows[1]["bucket"] == "FAILED" and rows[1]["error"] == "boom"


@pytest.mark.unit
def test_writeups_included_and_excludable():
    r = _res("A", "BUY", report="DETAILED WRITEUP HERE")
    with_wu, _ = build_report([r], "2024-05-10", include_writeups=True)
    without_wu, _ = build_report([r], "2024-05-10", include_writeups=False)
    assert "DETAILED WRITEUP HERE" in with_wu
    assert "DETAILED WRITEUP HERE" not in without_wu


@pytest.mark.unit
def test_empty_results_still_renders():
    md, csv_text = build_report([], "2024-05-10")
    assert "Analyzed **0**" in md
    assert "ticker,bucket" in csv_text  # header row present


@pytest.mark.unit
def test_run_funnel_chains_stages_and_writes(monkeypatch, tmp_path):
    from tradingagents.dataflows.screen import ScreenResult
    from tradingagents.funnel.triage import TriageResult
    monkeypatch.setattr(pl, "get_universe", lambda: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(pl, "screen_universe", lambda uni, top_n=None: [ScreenResult("AAA", 90.0, {}, {})])
    monkeypatch.setattr(pl, "triage_candidates", lambda scr, top_n=None: [TriageResult("AAA", 88.0, "t", None, 90.0)])
    monkeypatch.setattr(
        pl, "run_deep_analysis",
        lambda tri, date, atype, max_deep=None: [_res("AAA", "BUY", report="WRITEUP-AAA")],
    )

    out = pl.run_funnel("2024-05-10", out_dir=str(tmp_path))
    assert out.universe_size == 3
    assert out.report_path and out.csv_path and out.run_json_path
    written = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "**AAA**" in written and "2024-05-10" in written
    assert (tmp_path / "report.csv").read_text(encoding="utf-8").startswith("ticker,bucket")


@pytest.mark.unit
def test_run_funnel_persists_structured_run_json(monkeypatch, tmp_path):
    import json

    from tradingagents.dataflows.screen import ScreenResult
    from tradingagents.funnel.triage import TriageResult
    monkeypatch.setattr(pl, "get_universe", lambda: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(
        pl, "screen_universe",
        lambda uni, top_n=None: [ScreenResult("AAA", 91.0, {}, {"ret_3m": 0.2, "dollar_vol": 2e7}),
                                 ScreenResult("BBB", 70.0, {}, {"ret_3m": 0.1, "dollar_vol": 1e7})],
    )
    monkeypatch.setattr(
        pl, "triage_candidates",
        lambda scr, top_n=None: [TriageResult("AAA", 88.0, "great", None, 91.0)],
    )
    monkeypatch.setattr(
        pl, "run_deep_analysis",
        lambda tri, date, atype, max_deep=None: [_res("AAA", "BUY", report="WRITEUP-AAA")],
    )
    pl.run_funnel("2024-05-10", out_dir=str(tmp_path))
    rec = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert rec["trade_date"] == "2024-05-10" and rec["universe_size"] == 3
    assert rec["generated_at"]  # timestamp present
    pick = rec["picks"][0]
    assert pick["ticker"] == "AAA" and pick["bucket"] == "BUY"
    assert pick["report"] == "WRITEUP-AAA"   # full write-up archived for later viewing
    # Stage 0 + Stage 1 details archived for the report
    assert [s["ticker"] for s in rec["screened_detail"]] == ["AAA", "BBB"]
    assert rec["screened_detail"][0]["score"] == 91.0
    assert rec["triaged_detail"][0]["ticker"] == "AAA"
    assert rec["triaged_detail"][0]["triage_score"] == 88.0


@pytest.mark.unit
def test_run_funnel_no_write(monkeypatch):
    monkeypatch.setattr(pl, "get_universe", lambda: [])
    monkeypatch.setattr(pl, "screen_universe", lambda uni, top_n=None: [])
    monkeypatch.setattr(pl, "triage_candidates", lambda scr, top_n=None: [])
    monkeypatch.setattr(pl, "run_deep_analysis", lambda tri, date, atype, max_deep=None: [])
    out = pl.run_funnel("2024-05-10", write=False)
    assert out.report_path is None
    assert "Analyzed **0**" in out.report_md  # empty funnel still renders
