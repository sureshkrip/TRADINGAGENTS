"""tradingagents-funnel console entry point: arg parsing and run_funnel call."""

import pytest

import tradingagents.funnel.pipeline as pl
from tradingagents.funnel.pipeline import FunnelOutput


@pytest.mark.unit
def test_cli_defaults_date_to_today_and_calls_run_funnel(monkeypatch, capsys):
    seen = {}

    def fake_run_funnel(date, asset_type, *, top_screen, top_triage, max_deep, write):
        seen.update(date=date, asset_type=asset_type, top_screen=top_screen,
                    max_deep=max_deep, write=write)
        return FunnelOutput(trade_date=date, universe_size=3, report_md="", report_csv="",
                            report_path="/tmp/report.md")

    monkeypatch.setattr(pl, "run_funnel", fake_run_funnel)
    monkeypatch.setattr("sys.argv", ["tradingagents-funnel"])
    pl.main()

    assert seen["asset_type"] == "stock" and seen["write"] is True
    assert len(seen["date"]) == 10 and seen["date"][4] == "-"   # YYYY-MM-DD
    assert "report:" in capsys.readouterr().out


@pytest.mark.unit
def test_cli_passes_through_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(pl, "run_funnel", lambda date, atype, **kw: seen.update(
        date=date, atype=atype, **kw) or FunnelOutput(
        trade_date=date, universe_size=0, report_path=None))
    monkeypatch.setattr("sys.argv", [
        "tradingagents-funnel", "--date", "2024-05-10", "--asset-type", "crypto",
        "--top-screen", "40", "--max-deep", "3",
    ])
    pl.main()
    assert seen["date"] == "2024-05-10" and seen["atype"] == "crypto"
    assert seen["top_screen"] == 40 and seen["max_deep"] == 3
