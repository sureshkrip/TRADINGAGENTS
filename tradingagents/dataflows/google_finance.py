"""Google Finance & Google News RSS data vendor and quote utilities.

Provides keyless integration with Google Finance:
  1. Canonical URL resolution (``resolve_google_finance_url``) mapping Yahoo /
     broker symbols and international suffixes to Google Finance quote URLs.
  2. Live quote & company fundamentals scraping (``fetch_google_finance_quote``,
     ``get_google_finance_fundamentals``) using ``parsel`` (already a core
     dependency) to extract price, valuation metrics (Market Cap, P/E, Dividend
     Yield, 52-week range), and company profile from ``google.com/finance``.
  3. Date-bounded ticker and global financial news (``get_google_finance_news``,
     ``get_google_finance_global_news``) via Google News RSS with ``after:`` and
     ``before:`` operators and strict UTC look-ahead filtering.
"""

from __future__ import annotations

import contextlib
import html
import http.client
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from dateutil.relativedelta import relativedelta
from parsel import Selector

from .config import get_config
from .errors import NoMarketDataError
from .symbol_utils import crypto_base, normalize_symbol

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_GF_QUOTE_URL = "https://www.google.com/finance/quote/{symbol}:{exchange}?hl=en"
_GF_CRYPTO_URL = "https://www.google.com/finance/quote/{base}-USD?hl=en"
_GF_SEARCH_URL = "https://www.google.com/finance?q={query}&hl=en"
_GN_RSS_URL = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

# Map Yahoo Finance exchange codes / names to Google Finance exchange codes.
_EXCHANGE_TO_GF: dict[str, str] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NYQ": "NYSE",
    "NYSE": "NYSE",
    "PCX": "NYSEARCA",
    "ARCA": "NYSEARCA",
    "NYSEARCA": "NYSEARCA",
    "ASE": "NYSEAMERICAN",
    "AMEX": "NYSEAMERICAN",
    "BATS": "BATS",
    "TOR": "TSE",
    "TSX": "TSE",
    "LSE": "LON",
    "LON": "LON",
    "HKG": "HKG",
    "HKSE": "HKG",
    "JPX": "TYO",
    "TYO": "TYO",
    "NSI": "NSE",
    "NSE": "NSE",
    "BSP": "BOM",
    "BOM": "BOM",
    "BSE": "BOM",
    "ASX": "ASX",
    "SHH": "SHA",
    "SHA": "SHA",
    "SHZ": "SHE",
    "SHE": "SHE",
}

# Map ticker exchange suffixes (.HK, .T, .NS, etc.) to Google Finance exchanges.
_SUFFIX_TO_GF: dict[str, str] = {
    ".HK": "HKG",
    ".T": "TYO",
    ".L": "LON",
    ".NS": "NSE",
    ".BO": "BOM",
    ".TO": "TSE",
    ".AX": "ASX",
    ".SS": "SHA",
    ".SZ": "SHE",
}

# Candidate US exchanges to probe when only a bare US equity symbol is given.
_US_EXCHANGE_PROBES = ("NASDAQ", "NYSE", "NYSEARCA")


def _as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware; naive values are assumed UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _in_news_window(pub_date: datetime | None, start_dt: datetime, end_dt: datetime) -> bool:
    """Whether an article belongs in the half-open UTC window ``[start, end + 1 day)``."""
    end = _as_utc(end_dt)
    if pub_date is not None:
        return _as_utc(start_dt) <= _as_utc(pub_date) < end + timedelta(days=1)
    return end >= datetime.now(timezone.utc) - timedelta(days=1)


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(cleaned).split())


def _split_ticker_and_exchange(ticker: str, exchange: str | None = None) -> tuple[str, str | None]:
    """Resolve ``(google_symbol, google_exchange)`` from a ticker and optional exchange."""
    raw = (ticker or "").strip().upper()
    if not raw:
        return "", None

    # Allow explicit "NVDA:NASDAQ" input.
    if ":" in raw:
        sym, ex = raw.split(":", 1)
        return sym.strip(), _EXCHANGE_TO_GF.get(ex.strip(), ex.strip())

    # Check international exchange suffixes (.HK, .T, .NS, etc.).
    for suffix, gf_ex in _SUFFIX_TO_GF.items():
        if raw.endswith(suffix) and len(raw) > len(suffix):
            base_sym = raw[: -len(suffix)]
            # Hong Kong tickers on Google Finance use 4 digits (e.g. 0700:HKG).
            return base_sym, gf_ex

    # US class shares on Yahoo use a dash (BRK-B); Google Finance uses a dot (BRK.B).
    if re.fullmatch(r"[A-Z]{1,5}-[A-Z]", raw):
        raw = raw.replace("-", ".")

    gf_exchange = _EXCHANGE_TO_GF.get(exchange.strip().upper()) if exchange else None
    return raw, gf_exchange


def resolve_google_finance_url(ticker: str, exchange: str | None = None) -> str:
    """Return the canonical Google Finance URL for ``ticker``.

    Examples:
        - ``resolve_google_finance_url("NVDA", "NMS")`` -> ``https://www.google.com/finance/quote/NVDA:NASDAQ?hl=en``
        - ``resolve_google_finance_url("0700.HK")``     -> ``https://www.google.com/finance/quote/0700:HKG?hl=en``
        - ``resolve_google_finance_url("BTC-USD")``     -> ``https://www.google.com/finance/quote/BTC-USD?hl=en``
        - ``resolve_google_finance_url("NVDA")``        -> ``https://www.google.com/finance?q=NVDA&hl=en``
    """
    if not isinstance(ticker, str) or not ticker.strip():
        return "https://www.google.com/finance"

    c_base = crypto_base(ticker)
    if c_base:
        return _GF_CRYPTO_URL.format(base=c_base)

    sym, gf_ex = _split_ticker_and_exchange(ticker, exchange)
    if sym and gf_ex:
        return _GF_QUOTE_URL.format(symbol=sym, exchange=gf_ex)
    return _GF_SEARCH_URL.format(query=quote_plus(sym or ticker.strip().upper()))


def _parse_google_finance_html(html_text: str, ticker: str, url: str) -> dict[str, Any] | None:
    """Parse a Google Finance quote page HTML into a structured dictionary.

    Returns ``None`` if the page contains no recognizable quote price or stats
    (e.g., an unrecognized symbol exchange probe).
    """
    if not html_text:
        return None

    sel = Selector(text=html_text)
    price = (sel.css("div.YMlKec.fxKbKc::text").get() or "").strip()
    name = (sel.css("div.zzDege::text").get() or "").strip()

    # Extract key-value stats blocks (Previous close, Day range, Year range,
    # Market cap, Avg Volume, P/E ratio, Dividend yield, Primary exchange).
    stats: dict[str, str] = {}
    for row in sel.css("div.gyFHrc"):
        label = (row.css("div.mfs7Fc::text").get() or "").strip()
        val = (row.css("div.P6K39c::text").get() or "").strip()
        if not val:
            val = " ".join(t.strip() for t in row.css("div.P6K39c *::text").getall() if t.strip())
        if label and val:
            stats[label] = val

    about = " ".join(
        t.strip() for t in sel.css("div.bLLb2d::text, div.bLLb2d *::text").getall() if t.strip()
    ).strip()

    if not price and not stats:
        return None

    return {
        "ticker": ticker.strip().upper(),
        "name": name or None,
        "price": price or None,
        "stats": stats,
        "about": about or None,
        "url": url,
        "exchange": stats.get("Primary exchange"),
    }


def _http_get_text(url: str, timeout: float = 10.0) -> str | None:
    """Fetch URL text with a standard User-Agent; returns None on error."""
    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException) as exc:
        logger.debug("Google Finance HTTP request failed for %s: %s", url, exc)
        return None


def fetch_google_finance_quote(
    ticker: str,
    exchange: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Fetch quote and fundamental snapshot for ``ticker`` from Google Finance.

    Never raises on network errors: returns a dictionary with at least
    ``{"ticker": ..., "url": ..., "stats": {}}`` so callers (including the Web
    UI quote endpoint) can degrade gracefully.
    """
    c_base = crypto_base(ticker)
    if c_base:
        url = _GF_CRYPTO_URL.format(base=c_base)
        parsed = _parse_google_finance_html(_http_get_text(url, timeout) or "", ticker, url)
        if parsed:
            return parsed
        return {"ticker": ticker.upper(), "name": f"{c_base} / USD", "price": None, "stats": {}, "about": None, "url": url, "exchange": "CRYPTO"}

    sym, gf_ex = _split_ticker_and_exchange(ticker, exchange)
    candidates: list[str] = []
    if gf_ex:
        candidates.append(_GF_QUOTE_URL.format(symbol=sym, exchange=gf_ex))
    else:
        for probe_ex in _US_EXCHANGE_PROBES:
            candidates.append(_GF_QUOTE_URL.format(symbol=sym, exchange=probe_ex))

    for url in candidates:
        html_text = _http_get_text(url, timeout=timeout)
        parsed = _parse_google_finance_html(html_text or "", ticker, url)
        if parsed:
            return parsed

    fallback_url = resolve_google_finance_url(ticker, exchange)
    return {
        "ticker": ticker.strip().upper(),
        "name": None,
        "price": None,
        "stats": {},
        "about": None,
        "url": fallback_url,
        "exchange": gf_ex,
    }


def get_google_finance_fundamentals(
    ticker: str,
    curr_date: str = "",
) -> str:
    """Retrieve company fundamentals and quote snapshot from Google Finance.

    Conforms to the ``get_fundamentals`` vendor signature in ``interface.py``.
    Raises :class:`NoMarketDataError` when no usable quote/stats are found so
    multi-vendor fallback can try the next configured vendor.
    """
    canonical = normalize_symbol(ticker)
    snap = fetch_google_finance_quote(canonical)
    price = snap.get("price")
    stats: dict[str, str] = snap.get("stats") or {}
    if not price and not stats:
        raise NoMarketDataError(
            ticker,
            canonical,
            "no quote or fundamental stats found on Google Finance",
        )

    lines = [
        f"## Google Finance Fundamentals & Quote Snapshot for {ticker.upper()}",
        "",
    ]
    if snap.get("name"):
        lines.append(f"- **Company / Instrument**: {snap['name']}")
    if price:
        lines.append(f"- **Latest Price**: {price}")
    if curr_date:
        lines.append(f"- **Requested Analysis Date**: {curr_date}")
    if snap.get("url"):
        lines.append(f"- **Google Finance Quote**: {snap['url']}")
    lines.append("")

    if stats:
        lines += [
            "### Key Statistics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        for k, v in stats.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if snap.get("about"):
        lines += [
            "### Company Profile",
            "",
            snap["about"],
            "",
        ]

    return "\n".join(lines)


def _parse_rss_items(xml_text: str) -> list[dict[str, Any]]:
    """Parse Google News RSS XML into a list of article dicts."""
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("Google News RSS XML parse failed: %s", exc)
        return []

    articles: list[dict[str, Any]] = []
    for item in root.findall("./channel/item"):
        raw_title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source_el = item.find("source")
        publisher = (source_el.text or "").strip() if source_el is not None else ""
        if not publisher and " - " in raw_title:
            raw_title, publisher = raw_title.rsplit(" - ", 1)
        desc = _strip_html(item.findtext("description") or "")
        # Google News often repeats the headline inside description; omit if redundant.
        if desc and raw_title and desc.startswith(raw_title[:40]):
            desc = ""

        pub_date: datetime | None = None
        pub_str = (item.findtext("pubDate") or "").strip()
        if pub_str:
            with contextlib.suppress(ValueError, TypeError):
                pub_date = _as_utc(parsedate_to_datetime(pub_str))

        articles.append(
            {
                "title": raw_title or "No title",
                "publisher": publisher or "Google News",
                "summary": desc,
                "link": link,
                "pub_date": pub_date,
            }
        )
    return articles


def _fetch_google_news_rss(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Query Google News RSS with ``after:`` and ``before:`` bounds and filter strictly."""
    after_str = start_dt.strftime("%Y-%m-%d")
    before_str = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    dated_query = f"{query} after:{after_str} before:{before_str}"
    url = _GN_RSS_URL.format(query=quote_plus(dated_query))
    xml_text = _http_get_text(url, timeout=timeout)
    if not xml_text:
        return []

    items = _parse_rss_items(xml_text)
    kept: list[dict[str, Any]] = []
    for art in items:
        if not _in_news_window(art["pub_date"], start_dt, end_dt):
            continue
        kept.append(art)
        if len(kept) >= limit:
            break
    return kept


def get_google_finance_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve date-bounded financial news for ``ticker`` via Google Finance / News RSS.

    Supports historical date windows via Google Search's ``after:YYYY-MM-DD`` and
    ``before:YYYY-MM-DD`` operators and enforces UTC look-ahead filtering.
    """
    article_limit = get_config().get("news_article_limit", 20)
    canonical = normalize_symbol(ticker)
    c_base = crypto_base(ticker)
    search_term = f"{c_base} crypto" if c_base else f"{canonical} stock"
    resolved = "" if canonical == ticker else f" (resolved to {canonical})"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    articles = _fetch_google_news_rss(search_term, start_dt, end_dt, limit=article_limit)
    if not articles:
        return f"No news found for {ticker}{resolved} between {start_date} and {end_date}"

    blocks: list[str] = []
    for data in articles:
        entry = f"### {data['title']} (source: {data['publisher']})\n"
        if data["summary"]:
            entry += f"{data['summary']}\n"
        if data["link"]:
            entry += f"Link: {data['link']}\n"
        blocks.append(entry)

    return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n" + "\n".join(blocks)


def get_google_finance_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Retrieve global/macro economic news via Google Finance / News RSS."""
    config = get_config()
    if look_back_days is None:
        look_back_days = config.get("global_news_lookback_days", 7)
    if limit is None:
        limit = config.get("global_news_article_limit", 10)
    search_queries = config.get("global_news_queries", [])

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    all_articles: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for query in search_queries:
        fetched = _fetch_google_news_rss(query, start_dt, curr_dt, limit=limit)
        for art in fetched:
            title = art["title"]
            if title and title not in seen_titles:
                seen_titles.add(title)
                all_articles.append(art)
                if len(all_articles) >= limit:
                    break
        if len(all_articles) >= limit:
            break

    if not all_articles:
        return f"No global news found between {start_date} and {curr_date}"

    blocks: list[str] = []
    for data in all_articles[:limit]:
        entry = f"### {data['title']} (source: {data['publisher']})\n"
        if data["summary"]:
            entry += f"{data['summary']}\n"
        if data["link"]:
            entry += f"Link: {data['link']}\n"
        blocks.append(entry)

    return f"## Global Market News, from {start_date} to {curr_date}:\n\n" + "\n".join(blocks)
