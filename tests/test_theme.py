"""Theme → tickers discovery and theme funnel wiring (no network / LLM)."""

import pytest

import tradingagents.funnel.pipeline as pl
from tradingagents.funnel.theme import expand_theme, theme_slug


class _Msg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def invoke(self, messages):
        self.seen = messages[-1].content
        return _Msg(self.reply)


@pytest.mark.unit
def test_expand_theme_parses_ticker_array_and_cleans():
    llm = FakeLLM('["NVDA", "amd", "VRT", "not a ticker", "NVDA"]')
    out = expand_theme("data center", llm=llm)
    assert out == ["NVDA", "AMD", "VRT"]          # upper, deduped, junk dropped
    assert "data center" in llm.seen                # theme name reached the prompt


@pytest.mark.unit
def test_expand_theme_accepts_object_shape():
    llm = FakeLLM('[{"ticker":"PLTR"},{"symbol":"CRWD"}]')
    assert expand_theme("cybersecurity", llm=llm) == ["PLTR", "CRWD"]


@pytest.mark.unit
def test_expand_theme_respects_max_tickers():
    llm = FakeLLM('["A","B","C","D","E"]')
    assert expand_theme("x", llm=llm, max_tickers=3) == ["A", "B", "C"]


@pytest.mark.unit
def test_expand_theme_empty_name_raises():
    with pytest.raises(ValueError, match="required"):
        expand_theme("   ", llm=FakeLLM("[]"))


@pytest.mark.unit
def test_expand_theme_unparseable_returns_empty():
    assert expand_theme("x", llm=FakeLLM("sorry, no list")) == []


@pytest.mark.unit
@pytest.mark.parametrize("name,slug", [
    ("Data Center", "data-center"),
    ("nuclear / SMR", "nuclear-smr"),
    ("  EV & Mobility  ", "ev-mobility"),
])
def test_theme_slug(name, slug):
    assert theme_slug(name) == slug


@pytest.mark.unit
def test_run_theme_funnel_expands_then_runs_with_override(monkeypatch, tmp_path):
    monkeypatch.setattr(pl, "expand_theme", lambda name: ["AAA", "BBB"])
    captured = {}

    def fake_run_funnel(date, asset_type="stock", **kw):
        captured.update(date=date, **kw)
        from tradingagents.funnel.pipeline import FunnelOutput
        return FunnelOutput(trade_date=date, universe_size=len(kw.get("tickers") or []))

    monkeypatch.setattr(pl, "run_funnel", fake_run_funnel)
    out = pl.run_theme_funnel("data center", "2024-05-10", out_dir=str(tmp_path))
    assert captured["tickers"] == ["AAA", "BBB"]     # discovered list passed through
    assert captured["label"] == "data center"
    assert captured["max_deep"] == 2                  # theme default
    assert out.universe_size == 2


@pytest.mark.unit
def test_run_funnel_tickers_override_skips_get_universe(monkeypatch, tmp_path):
    # get_universe must NOT be called when an explicit list is supplied.
    monkeypatch.setattr(pl, "get_universe", lambda: (_ for _ in ()).throw(AssertionError("called")))
    monkeypatch.setattr(pl, "screen_universe", lambda uni, top_n=None: [])
    monkeypatch.setattr(pl, "triage_candidates", lambda scr, top_n=None: [])
    monkeypatch.setattr(pl, "run_deep_analysis", lambda tri, d, a, max_deep=None: [])
    out = pl.run_funnel("2024-05-10", tickers=["AAA", "BBB"], label="mytheme", out_dir=str(tmp_path))
    assert out.universe_size == 2
    import json
    rec = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert rec["label"] == "mytheme" and rec["universe"] == ["AAA", "BBB"]
