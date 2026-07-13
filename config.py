"""Global configuration — market-aware via the MARKET env var."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# ── Market registry ───────────────────────────────────────────────────────────
# One deployment per market: set MARKET=ca|us|uk|za|id|hk|cn (Streamlit secrets).
# fx_to_usd: rough local→USD factor used for market-cap bucket thresholds.
MARKETS = {
    "ca": dict(name="Canada",         exch="TSX · TSXV · CSE",  flag="🇨🇦",
               cur="C$",  fx_to_usd=0.73,     tz="America/Toronto",      hour=7,  minute=30),
    "us": dict(name="United States",  exch="NYSE · NASDAQ",     flag="🇺🇸",
               cur="US$", fx_to_usd=1.00,     tz="America/New_York",     hour=7,  minute=30),
    "uk": dict(name="United Kingdom", exch="LSE · AIM",         flag="🇬🇧",
               cur="£",   fx_to_usd=1.27,     tz="Europe/London",        hour=7,  minute=30),
    "za": dict(name="South Africa",   exch="JSE",               flag="🇿🇦",
               cur="R",   fx_to_usd=0.055,    tz="Africa/Johannesburg",  hour=8,  minute=0),
    "id": dict(name="Indonesia",      exch="IDX",               flag="🇮🇩",
               cur="Rp",  fx_to_usd=0.000062, tz="Asia/Jakarta",         hour=8,  minute=30),
    "hk": dict(name="Hong Kong",      exch="HKEX",              flag="🇭🇰",
               cur="HK$", fx_to_usd=0.128,    tz="Asia/Hong_Kong",       hour=8,  minute=30),
    "cn": dict(name="China A-shares", exch="SSE · SZSE",        flag="🇨🇳",
               cur="¥",   fx_to_usd=0.14,     tz="Asia/Shanghai",        hour=8,  minute=30),
}

MARKET = os.getenv("MARKET", "ca").strip().lower()
if MARKET not in MARKETS:
    MARKET = "ca"
_M = MARKETS[MARKET]

MARKET_NAME = _M["name"]
MARKET_EXCH = _M["exch"]
MARKET_FLAG = _M["flag"]
CURRENCY    = _M["cur"]
FX_TO_USD   = _M["fx_to_usd"]

DB_PATH       = BASE_DIR / f"screener_{MARKET}.db"
UNIVERSE_JSON = BASE_DIR / "universes" / f"{MARKET}.json"

# ── Refresh schedule (local exchange time; used only for local APScheduler) ───
REFRESH_HOUR   = _M["hour"]
REFRESH_MINUTE = _M["minute"]
TIMEZONE       = _M["tz"]

# ── Data source ───────────────────────────────────────────────────────────────
YFINANCE_BATCH = 50          # tickers per yfinance batch call
FETCH_DELAY    = 1.5         # seconds between batches (rate limit)

# SPG (S&P / mining-specific) data source:
#   "ciq" — pre-fetched spg_{market}.json exported via the S&P Capital IQ Pro
#           Excel add-in (same schema as the AU screener's asx_spg_results.json).
# No SNL Snowflake here (subscription paused; mapping only exists for ASX).
SPG_SOURCE = os.getenv("SPG_SOURCE", "ciq")
SPG_JSON   = BASE_DIR / f"spg_{MARKET}.json"

# ── Bloomberg (optional) ──────────────────────────────────────────────────────
USE_BLOOMBERG  = os.getenv("USE_BLOOMBERG", "false").lower() == "true"
BLOOMBERG_HOST = os.getenv("BLOOMBERG_HOST", "localhost")
BLOOMBERG_PORT = int(os.getenv("BLOOMBERG_PORT", 8194))

# ── Scoring weights ───────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "valuation":   0.30,   # P/B, EV/EBITDA, P/CF, P/NAV
    "health":      0.20,   # debt, cash, liquidity
    "momentum":    0.15,   # 52-wk position, RSI
    "mining":      0.25,   # CIQ/SNL: AISC margin, reserves backing, NAV discount
    "commodity":   0.05,   # commodity outlook multiplier
    "stage":       0.05,   # stage-adjusted potential
}

# ── Spot commodity prices (USD; overwritten at runtime by fetch_spot_prices) ──
COMMODITY_SPOT = {
    "Gold":      4614.0,   # USD/oz
    "Silver":      74.15,  # USD/oz
    "Copper":   13247.0,   # USD/tonne
    "Zinc":      3365.0,   # USD/tonne
    "Nickel":   19050.0,   # USD/tonne
    "Uranium":     77.40,  # USD/lb
    "Iron Ore":   105.0,   # USD/tonne
    "Coal":       210.0,   # USD/tonne
    "Rare Earths":  0.0,
}

# ── Commodity outlook (1.0 = neutral, >1 bullish, <1 bearish) ─────────────────
COMMODITY_OUTLOOK = {
    "Gold":        1.20,
    "Silver":      1.15,
    "Copper":      1.15,
    "Uranium":     1.25,
    "Lithium":     0.80,
    "Nickel":      0.90,
    "Zinc":        1.00,
    "Iron Ore":    0.90,
    "Coal":        0.75,
    "Rare Earths": 1.15,
    "Manganese":   0.95,
    "PGM":         1.05,
    "Potash":      1.00,
    "Aluminum":    1.00,
    "Diversified": 1.05,
}

# ── Stage labels & risk/reward profile ────────────────────────────────────────
STAGE_ORDER = [
    "Major Producer",
    "Mid-tier Producer",
    "Producer",
    "Developer",
    "Explorer",
    "Royalty/Streaming",
]

# ── Market cap buckets (local currency, anchored to USD 5B/500M/50M) ──────────
def _fmt_local(usd: float) -> str:
    local = usd / FX_TO_USD
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if local >= div:
            v = local / div
            return f"{CURRENCY}{v:,.0f}{unit}" if v >= 10 else f"{CURRENCY}{v:.1f}{unit}"
    return f"{CURRENCY}{local:,.0f}"

_B5, _B05, _B005 = 5e9 / FX_TO_USD, 5e8 / FX_TO_USD, 5e7 / FX_TO_USD
MCAP_BUCKETS = {
    f"Large (>{_fmt_local(5e9)})":                    (_B5,   float("inf")),
    f"Mid ({_fmt_local(5e8)}–{_fmt_local(5e9)})":     (_B05,  _B5),
    f"Small ({_fmt_local(5e7)}–{_fmt_local(5e8)})":   (_B005, _B05),
    f"Micro (<{_fmt_local(5e7)})":                    (0,     _B005),
}
