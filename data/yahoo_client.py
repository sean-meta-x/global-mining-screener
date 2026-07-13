"""
Direct Yahoo Finance client using requests + crumb auth.
Replaces yfinance for ticker info and price history.
"""
import time
import logging
import requests

log = logging.getLogger(__name__)

_SESSION: requests.Session | None = None
_CRUMB: str = ""

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
    "Origin": "https://finance.yahoo.com",
}


def _init_session() -> bool:
    """Initialise session: fetch cookies then crumb."""
    global _SESSION, _CRUMB
    s = requests.Session()
    s.headers.update(_HEADERS)

    # 1. Hit consent page to populate cookies
    try:
        s.get("https://fc.yahoo.com/", timeout=15)
    except Exception:
        pass
    try:
        s.get("https://finance.yahoo.com/", timeout=15)
    except Exception:
        pass

    # 2. Fetch crumb
    for attempt in range(3):
        try:
            r = s.get(
                "https://query2.finance.yahoo.com/v1/test/getcrumb",
                timeout=15,
            )
            if r.status_code == 200 and r.text.strip():
                _CRUMB = r.text.strip()
                _SESSION = s
                log.info(f"Yahoo crumb acquired: {_CRUMB[:8]}…")
                return True
        except Exception as e:
            log.warning(f"Crumb attempt {attempt+1}: {e}")
        time.sleep(2)

    log.error("Failed to acquire Yahoo Finance crumb.")
    return False


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _init_session()
    return _SESSION  # type: ignore[return-value]


# ── Public helpers ─────────────────────────────────────────────────────────────

QUOTE_MODULES = (
    "financialData,quoteType,defaultKeyStatistics,"
    "assetProfile,summaryDetail,price"
)


def get_info(symbol: str, retries: int = 3) -> dict:
    """Return the merged quoteSummary dict for *symbol*, or {} on failure."""
    s = _session()
    url = (
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        f"?modules={QUOTE_MODULES}&crumb={_CRUMB}&formatted=false"
        f"&corsDomain=finance.yahoo.com&symbol={symbol}"
    )
    for attempt in range(retries):
        try:
            r = s.get(url, timeout=20)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                log.warning(f"[{symbol}] 429 – sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                log.warning(f"[{symbol}] HTTP {r.status_code}")
                return {}
            data = r.json()
            result = data.get("quoteSummary", {}).get("result") or []
            if not result:
                return {}
            merged: dict = {}
            for module_data in result:
                for v in module_data.values():
                    if isinstance(v, dict):
                        merged.update(v)
            return merged
        except Exception as e:
            log.warning(f"[{symbol}] attempt {attempt+1}: {e}")
            time.sleep(2)
    return {}


def get_price_history(symbol: str, period: str = "1y") -> list[tuple[int, float]]:
    """
    Return [(unix_timestamp, close_price), …] sorted ascending, or [].
    *period* follows Yahoo convention: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
    """
    s = _session()
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range={period}&crumb={_CRUMB}"
    )
    for attempt in range(3):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return []
            data = r.json()
            chart = data.get("chart", {}).get("result") or []
            if not chart:
                return []
            ts = chart[0].get("timestamp", [])
            closes = (
                chart[0]
                .get("indicators", {})
                .get("adjclose", [{}])[0]
                .get("adjclose", [])
            )
            if not closes:
                closes = (
                    chart[0]
                    .get("indicators", {})
                    .get("quote", [{}])[0]
                    .get("close", [])
                )
            pairs = [
                (t, c) for t, c in zip(ts, closes) if c is not None
            ]
            return sorted(pairs, key=lambda x: x[0])
        except Exception as e:
            log.warning(f"[{symbol}] history attempt {attempt+1}: {e}")
            time.sleep(2)
    return []
