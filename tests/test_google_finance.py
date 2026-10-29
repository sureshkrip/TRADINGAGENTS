"""Unit tests for Google Finance & Google News RSS vendor integration."""

from __future__ import annotations

import copy

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import google_finance as gf, interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError


@pytest.fixture(autouse=True)
def _reset_cfg():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


_SAMPLE_GF_HTML = """
<html><body>
  <div class="zzDege">NVIDIA Corp</div>
  <div class="YMlKec fxKbKc">$138.60</div>
  <div class="gyFHrc"><div class="mfs7Fc">Previous close</div><div class="P6K39c">$135.40</div></div>
  <div class="gyFHrc"><div class="mfs7Fc">Market cap</div><div class="P6K39c">3.39T USD</div></div>
  <div class="gyFHrc"><div class="mfs7Fc">P/E ratio</div><div class="P6K39c">54.12</div></div>
  <div class="gyFHrc"><div class="mfs7Fc">Primary exchange</div><div class="P6K39c">NASDAQ</div></div>
  <div class="bLLb2d">NVIDIA Corporation designs graphics processing units and AI accelerators.</div>
</body></html>
"""

_SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>NVIDIA unveils next-gen data center GPU - Reuters</title>
      <link>https://example.com/nvda-gpu</link>
      <pubDate>Thu, 09 May 2024 14:30:00 GMT</pubDate>
      <source url="https://reuters.com">Reuters</source>
      <description>&lt;p&gt;Strong demand from cloud providers.&lt;/p&gt;</description>
    </item>
    <item>
      <title>Future leaked headline after trade date - Bloomberg</title>
      <link>https://example.com/future</link>
      <pubDate>Sun, 12 May 2024 10:00:00 GMT</pubDate>
      <source url="https://bloomberg.com">Bloomberg</source>
      <description>Should be filtered out by look-ahead guard.</description>
    </item>
  </channel>
</rss>
"""


@pytest.mark.unit
def test_resolve_google_finance_url_variants():
    assert (
        gf.resolve_google_finance_url("NVDA", "NMS")
        == "https://www.google.com/finance/quote/NVDA:NASDAQ?hl=en"
    )
    assert (
        gf.resolve_google_finance_url("0700.HK")
        == "https://www.google.com/finance/quote/0700:HKG?hl=en"
    )
    assert (
        gf.resolve_google_finance_url("7203.T")
        == "https://www.google.com/finance/quote/7203:TYO?hl=en"
    )
    assert (
        gf.resolve_google_finance_url("RELIANCE.NS")
        == "https://www.google.com/finance/quote/RELIANCE:NSE?hl=en"
    )
    assert (
        gf.resolve_google_finance_url("BRK-B", "NYQ")
        == "https://www.google.com/finance/quote/BRK.B:NYSE?hl=en"
    )
    assert (
        gf.resolve_google_finance_url("BTC-USD")
        == "https://www.google.com/finance/quote/BTC-USD?hl=en"
    )
    assert (
        gf.resolve_google_finance_url("NVDA")
        == "https://www.google.com/finance?q=NVDA&hl=en"
    )


@pytest.mark.unit
def test_parse_google_finance_html_extracts_price_stats_and_about():
    parsed = gf._parse_google_finance_html(
        _SAMPLE_GF_HTML, "NVDA", "https://www.google.com/finance/quote/NVDA:NASDAQ?hl=en"
    )
    assert parsed is not None
    assert parsed["name"] == "NVIDIA Corp"
    assert parsed["price"] == "$138.60"
    assert parsed["stats"]["Market cap"] == "3.39T USD"
    assert parsed["stats"]["P/E ratio"] == "54.12"
    assert parsed["exchange"] == "NASDAQ"
    assert "graphics processing units" in parsed["about"]


@pytest.mark.unit
def test_get_google_finance_fundamentals_renders_markdown(monkeypatch):
    monkeypatch.setattr(gf, "_http_get_text", lambda url, timeout=10.0: _SAMPLE_GF_HTML)
    md = gf.get_google_finance_fundamentals("NVDA", "2024-05-10")
    assert "## Google Finance Fundamentals & Quote Snapshot for NVDA" in md
    assert "NVIDIA Corp" in md
    assert "$138.60" in md
    assert "| Market cap | 3.39T USD |" in md
    assert "graphics processing units" in md


@pytest.mark.unit
def test_get_google_finance_fundamentals_raises_no_data_when_empty(monkeypatch):
    monkeypatch.setattr(gf, "_http_get_text", lambda url, timeout=10.0: "<html></html>")
    with pytest.raises(NoMarketDataError):
        gf.get_google_finance_fundamentals("INVALID")


@pytest.mark.unit
def test_get_google_finance_news_filters_lookahead_and_formats(monkeypatch):
    seen_urls: list[str] = []

    def fake_get(url, timeout=10.0):
        seen_urls.append(url)
        return _SAMPLE_RSS_XML

    monkeypatch.setattr(gf, "_http_get_text", fake_get)
    report = gf.get_google_finance_news("NVDA", "2024-05-03", "2024-05-10")
    assert "after%3A2024-05-03" in seen_urls[0]
    assert "before%3A2024-05-11" in seen_urls[0]
    assert "NVIDIA unveils next-gen data center GPU" in report
    assert "Reuters" in report
    # Article dated 2024-05-12 must be excluded by the look-ahead guard
    assert "Future leaked headline" not in report


@pytest.mark.unit
def test_google_finance_vendor_routable_via_interface(monkeypatch):
    monkeypatch.setattr(gf, "_http_get_text", lambda url, timeout=10.0: _SAMPLE_RSS_XML)
    set_config({"data_vendors": {"news_data": "google_finance"}})
    res = interface.route_to_vendor("get_news", "NVDA", "2024-05-03", "2024-05-10")
    assert "NVIDIA unveils next-gen data center GPU" in res
