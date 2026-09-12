"""Universe sourcing: the env-var knob, per-source parsing, and cleaning.

Covers the three sources (sp500 / all_us / file), the shared _clean() junk
filter, TRADINGAGENTS_UNIVERSE_LIMIT, and error paths — all without network:
the S&P 500 CSV read and the Alpha Vantage request are monkeypatched.
"""

import pandas as pd
import pytest

import tradingagents.dataflows.universe as u


@pytest.mark.unit
def test_clean_dedupes_uppercases_and_drops_junk():
    got = u._clean(["aapl", "AAPL", " msft ", "BRK.B", "BF-B", "", "TOOLONGX", "A@B", None])
    # order-stable, first-seen wins; class-share suffixes kept; junk dropped.
    assert got == ["AAPL", "MSFT", "BRK.B", "BF-B"]


@pytest.mark.unit
def test_default_source_is_sp500(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_UNIVERSE", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_UNIVERSE_LIMIT", raising=False)
    monkeypatch.setattr(u, "_sp500_constituents", lambda: ["AAPL", "MSFT"])
    assert u.get_universe() == ["AAPL", "MSFT"]


@pytest.mark.unit
def test_unknown_source_raises(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE", "nope")
    with pytest.raises(ValueError, match="Unknown TRADINGAGENTS_UNIVERSE"):
        u.get_universe()


@pytest.mark.unit
def test_limit_truncates(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE", "sp500")
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE_LIMIT", "2")
    monkeypatch.setattr(u, "_sp500_constituents", lambda: ["A", "B", "C", "D"])
    assert u.get_universe() == ["A", "B"]


@pytest.mark.unit
def test_limit_non_integer_raises(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE", "sp500")
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE_LIMIT", "lots")
    monkeypatch.setattr(u, "_sp500_constituents", lambda: ["A"])
    with pytest.raises(ValueError, match="must be an integer"):
        u.get_universe()


@pytest.mark.unit
def test_sp500_reads_symbol_column(monkeypatch):
    df = pd.DataFrame({"Symbol": ["AAPL", "brk.b"], "Name": ["Apple", "Berkshire"]})
    monkeypatch.setattr(u.pd, "read_csv", lambda *a, **k: df)
    assert u._sp500_constituents() == ["AAPL", "BRK.B"]


@pytest.mark.unit
def test_alpha_vantage_listing_filters_to_us_common_stock(monkeypatch):
    csv_text = (
        "symbol,name,exchange,assetType,ipoDate,delistingDate,status\n"
        "AAPL,Apple,NASDAQ,Stock,1980-12-12,null,Active\n"
        "SPY,SPDR S&P 500,NYSE,ETF,1993-01-29,null,Active\n"       # ETF -> dropped
        "PINK,Pinksheet Co,OTC,Stock,2000-01-01,null,Active\n"      # OTC -> dropped
        "MSFT,Microsoft,NASDAQ,Stock,1986-03-13,null,Active\n"
    )
    monkeypatch.setattr(u, "_make_api_request", lambda fn, params: csv_text)
    assert u._alpha_vantage_listing() == ["AAPL", "MSFT"]


@pytest.mark.unit
def test_file_source_parses_lines_commas_and_comments(monkeypatch, tmp_path):
    p = tmp_path / "watchlist.txt"
    p.write_text("AAPL, MSFT  # my tech\n# a comment line\nNVDA\n\ngoog\n")
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE", "file")
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE_FILE", str(p))
    monkeypatch.delenv("TRADINGAGENTS_UNIVERSE_LIMIT", raising=False)
    assert u.get_universe() == ["AAPL", "MSFT", "NVDA", "GOOG"]


@pytest.mark.unit
def test_file_source_requires_path(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE", "file")
    monkeypatch.delenv("TRADINGAGENTS_UNIVERSE_FILE", raising=False)
    with pytest.raises(ValueError, match="requires TRADINGAGENTS_UNIVERSE_FILE"):
        u.get_universe()


@pytest.mark.unit
def test_file_source_missing_file_raises(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE", "file")
    monkeypatch.setenv("TRADINGAGENTS_UNIVERSE_FILE", "/no/such/list.txt")
    with pytest.raises(ValueError, match="not found"):
        u.get_universe()
