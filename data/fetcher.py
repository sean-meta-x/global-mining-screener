"""
Data fetcher: Yahoo Finance API + S&P CIQ JSON overlay + SNL Snowflake overlay.
Returns a DataFrame with one row per ASX ticker.
"""
import os
import re
import time
import logging
import warnings
import requests
import numpy as np
import pandas as pd
import ta
from bs4 import BeautifulSoup

import config
from config import FETCH_DELAY
from data.yahoo_client import get_info, get_price_history

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Commodity spot price fetchers ─────────────────────────────────────────────
#
# Source 1 — Yahoo Finance futures (no API key):
#   GC=F gold USD/oz, SI=F silver USD/oz, HG=F copper USD/lb, NI=F nickel USD/t
#
# Source 2 — westmetall.com (free LME daily settlement, no API key):
#   Zinc and Nickel as fallback when Yahoo has no data
#
# Fallback — config.COMMODITY_SPOT static values (manually updated)

# Yahoo Finance futures: symbol → (commodity, display_unit, multiply_to_get_config_unit)
_YAHOO_SPOTS: dict[str, tuple[str, str, float]] = {
    "GC=F": ("Gold",    "USD/oz",    1.0),         # COMEX gold, USD/oz
    "SI=F": ("Silver",  "USD/oz",    1.0),          # COMEX silver, USD/oz
    "HG=F": ("Copper",  "USD/tonne", 2_204.623),    # COMEX copper, USD/lb → tonne
    # "NI=F": ("Nickel",  "USD/tonne", 1.0),          # REMOVED: Yahoo NI=F returns HTTP 404 — Nickel falls back to westmetall.com
    "UX=F": ("Uranium", "USD/lb",    1.0),          # Uranium (CME UxC futures)
}

# westmetall.com href field → commodity key in config.COMMODITY_SPOT
_WESTMETALL_FIELDS: dict[str, str] = {
    "LME_Zn_cash": "Zinc",
    "LME_Ni_cash": "Nickel",
}

_WM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _fetch_yahoo_spots() -> dict[str, float]:
    """Fetch commodity futures prices from Yahoo Finance. Returns {commodity: price}."""
    fetched: dict[str, float] = {}
    for symbol, (commodity, unit, factor) in _YAHOO_SPOTS.items():
        try:
            info = get_info(symbol)
            raw = info.get("regularMarketPrice")
            if raw is not None:
                price = round(float(raw) * factor, 2)
                fetched[commodity] = price
                log.info(f"[spot/yahoo] {commodity}: {price:,.2f} {unit} ({symbol})")
            else:
                log.warning(f"[spot/yahoo] {symbol}: no price returned")
        except Exception as e:
            log.warning(f"[spot/yahoo] {symbol}: {e}")
    return fetched


def _fetch_westmetall_spots() -> dict[str, float]:
    """
    Scrape LME daily settlement prices from westmetall.com (free, no API key).
    Parses <a href="...LME_Zn_cash...">3,430.00</a> style links.
    Returns {commodity: price_usd_per_tonne}.
    """
    fetched: dict[str, float] = {}
    try:
        r = requests.get(
            "https://www.westmetall.com/en/markdaten.php",
            headers=_WM_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for field, commodity in _WESTMETALL_FIELDS.items():
            for tag in soup.find_all("a", href=lambda h: h and field in h):
                text = tag.get_text(strip=True).replace(",", "")
                try:
                    price = float(text)
                    if price > 100:   # sanity: real metal prices never < $100/t
                        fetched[commodity] = price
                        log.info(f"[spot/westmetall] {commodity}: {price:,.2f} USD/t")
                        break
                except ValueError:
                    continue
    except Exception as e:
        log.warning(f"[spot/westmetall] {e}")
    return fetched


def _fetch_spg_spots() -> dict[str, float]:
    """
    Load commodity spot prices from commodity_spots.json produced by the CIQ add-in.
    Only used if file exists and was written within the last 24 hours.
    Returns {commodity: price} or {}.
    """
    import datetime, json
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "commodity_spots.json")
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        fetched_at = datetime.datetime.fromisoformat(payload.get("fetched_at", "2000-01-01"))
        age_hours  = (datetime.datetime.now() - fetched_at).total_seconds() / 3600
        if age_hours > 24:
            log.info(f"[spot/spg] commodity_spots.json is {age_hours:.0f}h old — skipping")
            return {}
        prices = {k: float(v) for k, v in payload.get("prices", {}).items() if v}
        if prices:
            log.info(f"[spot/spg] Loaded from S&P Capital IQ ({age_hours:.1f}h ago): {list(prices.keys())}")
        return prices
    except Exception as e:
        log.warning(f"[spot/spg] {e}")
        return {}


def fetch_spot_prices() -> dict[str, float]:
    """
    Fetch live commodity spot prices and update config.COMMODITY_SPOT in-place.

    Priority chain:
      1. S&P Capital IQ (commodity_spots.json from CIQ add-in) — most accurate
      2. Yahoo Finance futures (GC=F, SI=F, HG=F, NI=F)
      3. westmetall.com LME daily settlement (Zinc, Nickel)
      4. config static fallback values

    Returns {commodity: price} for every commodity successfully fetched.
    """
    fetched: dict[str, float] = {}

    # Source 1: S&P Capital IQ (highest priority — authoritative SNL data)
    fetched.update(_fetch_spg_spots())

    # Source 2: Yahoo Finance futures (fills gaps not covered by SPG)
    yahoo = _fetch_yahoo_spots()
    for commodity, price in yahoo.items():
        if commodity not in fetched:
            fetched[commodity] = price

    # Source 3: westmetall.com LME for anything still missing
    missing = [c for c in config.COMMODITY_SPOT if c not in fetched]
    if missing:
        wm = _fetch_westmetall_spots()
        for commodity, price in wm.items():
            if commodity not in fetched:
                fetched[commodity] = price

    # Apply to config (scorer reads from config.COMMODITY_SPOT at call time)
    for commodity, price in fetched.items():
        config.COMMODITY_SPOT[commodity] = price

    not_fetched = [c for c in config.COMMODITY_SPOT if c not in fetched]
    if not_fetched:
        log.warning(f"[spot] Using static fallback for: {not_fetched}")
    log.info(f"[spot] Live prices updated for: {list(fetched.keys())}")
    return fetched

# ── Field map from quoteSummary merged dict ────────────────────────────────────
# Keys that come back from Yahoo quoteSummary (merged across all modules).
_INFO_FIELDS = [
    "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
    "priceToBook", "priceToSalesTrailingTwelveMonths",
    "ebitda", "ebitdaMargins", "operatingCashflow", "freeCashflow",
    "totalDebt", "totalCash", "totalRevenue",
    "debtToEquity", "currentRatio", "returnOnEquity", "returnOnAssets",
    "grossMargins", "operatingMargins", "profitMargins",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "regularMarketPrice",
    "averageVolume", "sharesOutstanding",
    "shortRatio", "beta",
    "dividendYield", "dividendRate", "trailingAnnualDividendYield",
    "currency", "exchange", "sector", "industry",
    "longName", "country",
    # Analyst consensus
    "targetMeanPrice", "targetHighPrice", "targetLowPrice", "targetMedianPrice",
    "numberOfAnalystOpinions", "recommendationKey", "recommendationMean",
]


# ── Quote-currency normalization ───────────────────────────────────────────────
# LSE quotes in pence (GBp) and JSE in ZAR cents (ZAc). Yahoo returns price-level
# fields in the subunit but book value, marketCap and enterpriseValue in the major
# unit, so its priceToBook comes back exactly 100x too high. Separately, financial-
# statement fields arrive in financialCurrency (e.g. USD for Glencore/South32),
# which corrupts EV/EBITDA, P/CF etc. when the quote currency differs.
_SUBUNIT_CURRENCIES = {"GBp": ("GBP", 100.0), "ZAc": ("ZAR", 100.0)}

# Fields quoted in the exchange's (sub)unit
_PRICE_LEVEL_FIELDS = (
    "regularMarketPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "dividendRate",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice", "targetMedianPrice",
)

# Absolute financial-statement fields reported in financialCurrency
_FINANCIAL_FIELDS = (
    "ebitda", "totalRevenue", "totalCash", "totalDebt",
    "operatingCashflow", "freeCashflow",
)

_FX_CACHE: dict[str, float | None] = {}


def _fx_rate(from_cur: str, to_cur: str) -> float | None:
    """Spot FX rate from Yahoo (e.g. USDZAR=X). Cached per pair per run."""
    if from_cur == to_cur:
        return 1.0
    pair = f"{from_cur}{to_cur}"
    if pair not in _FX_CACHE:
        raw = get_info(f"{pair}=X").get("regularMarketPrice")
        try:
            _FX_CACHE[pair] = float(raw) if raw else None
        except (TypeError, ValueError):
            _FX_CACHE[pair] = None
        if _FX_CACHE[pair] is None:
            log.warning(f"[fx] No rate for {pair} — financials left unconverted")
    return _FX_CACHE[pair]


def _normalize_currency(info: dict) -> dict:
    """
    Return a copy of *info* with every monetary field in the quote's major currency.

    1. Subunit quotes (GBp / ZAc): price-level fields and Yahoo's priceToBook
       (subunit price ÷ major-unit book value) are divided by 100; the currency
       label is upgraded to GBP / ZAR. marketCap and enterpriseValue are already
       major-unit and stay untouched.
    2. financialCurrency ≠ quote currency (e.g. USD reporters on LSE/JSE):
       statement absolutes are FX-converted so ratios vs marketCap/EV are
       consistent. On FX failure the fields are left as-is (logged).
    """
    out = dict(info)

    cur = out.get("currency")
    if cur in _SUBUNIT_CURRENCIES:
        major, factor = _SUBUNIT_CURRENCIES[cur]
        for field in _PRICE_LEVEL_FIELDS + ("priceToBook",):
            val = out.get(field)
            if val is not None:
                try:
                    out[field] = float(val) / factor
                except (TypeError, ValueError):
                    pass
        out["currency"] = major
        cur = major

    fin_cur = out.get("financialCurrency")
    if cur and fin_cur and fin_cur != cur:
        rate = _fx_rate(fin_cur, cur)
        if rate:
            for field in _FINANCIAL_FIELDS:
                val = out.get(field)
                if val is not None:
                    try:
                        out[field] = float(val) * rate
                    except (TypeError, ValueError):
                        pass
    return out


def _fetch_info_batch(tickers: list[str]) -> dict[str, dict]:
    """Fetch quoteSummary info for each ticker, one at a time."""
    results = {}
    for tk in tickers:
        info = get_info(tk)
        if info:
            info = _normalize_currency(info)
            results[tk] = {k: info.get(k) for k in _INFO_FIELDS}
        else:
            log.warning(f"[info] {tk}: no data")
            results[tk] = {}
        time.sleep(FETCH_DELAY)
    return results


def _price_technicals(tickers: list[str]) -> dict[str, dict]:
    """RSI, 52-wk position, 50/200-day MA from price history."""
    techs = {}
    for tk in tickers:
        pairs = get_price_history(tk, period="1y")
        if len(pairs) < 14:
            techs[tk] = {}
            continue
        closes = [c for _, c in pairs]
        s = pd.Series(closes, dtype=float)
        try:
            rsi_val = float(ta.momentum.RSIIndicator(s, window=14).rsi().iloc[-1])
        except Exception:
            rsi_val = float("nan")
        ma50  = s.rolling(50).mean().iloc[-1]
        ma200 = s.rolling(200).mean().iloc[-1]
        px    = s.iloc[-1]
        hi52  = s.rolling(min(252, len(s))).max().iloc[-1]
        lo52  = s.rolling(min(252, len(s))).min().iloc[-1]
        rng   = hi52 - lo52 if hi52 != lo52 else 1.0
        # Period returns: 1M ≈ 21 trading days, 3M ≈ 63 trading days
        n = len(s)
        return_1m = round((px / float(s.iloc[-22]) - 1) * 100, 1) if n >= 22 else None
        return_3m = round((px / float(s.iloc[-63]) - 1) * 100, 1) if n >= 63 else None

        techs[tk] = {
            "rsi":            round(rsi_val, 1) if not np.isnan(rsi_val) else None,
            "ma50":           round(float(ma50),  4) if not np.isnan(ma50)  else None,
            "ma200":          round(float(ma200), 4) if not np.isnan(ma200) else None,
            "pct_from_52hi":  round((px - hi52) / hi52 * 100, 1),
            "pct_from_52lo":  round((px - lo52) / lo52 * 100, 1),
            "wk52_position":  round((px - lo52) / rng * 100, 1),
            "price_vs_ma50":  round((px - ma50)  / ma50  * 100, 1) if not np.isnan(ma50)  else None,
            "price_vs_ma200": round((px - ma200) / ma200 * 100, 1) if not np.isnan(ma200) else None,
            "return_1m":      return_1m,
            "return_3m":      return_3m,
        }
        time.sleep(FETCH_DELAY)
    return techs


def _spg_overlay(tickers: list[str]) -> dict[str, dict]:
    """
    Load S&P Capital IQ + SNL Mining overlay from pre-fetched JSON.
    Run the CIQ add-in export first to generate spg_{market}.json (see config.SPG_JSON).

    Returns a dict keyed by Yahoo Finance ticker (e.g. 'BHP.AX') with fields:
      nav_per_shr   — NAV per share (USD, from SPG)
      p_nav         — Price / NAV (computed using CIQ close price)
      reserves_m    — Reserves in-situ value ($M)
      resources_m   — Resources in-situ value ($M)
      aisc_per_oz   — AISC in $/oz (gold; converted from $/tonne)
      aisc_per_t    — AISC in $/tonne (copper/other)
    """
    import json
    json_path = str(config.SPG_JSON)
    if not os.path.exists(json_path):
        log.info(f"[SPG] {os.path.basename(json_path)} not found — CIQ overlay skipped")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        spg_data = json.load(f)

    # Build CIQ ticker → Yahoo Finance ticker map for ASX tickers.
    # ASX tickers stored as "ASX:BHP" → "BHP.AX" etc.
    _CIQ_YF_OVERRIDES: dict[str, str] = {}   # Add ASX-specific overrides here as needed

    def _to_yf(ciq_ticker: str) -> str:
        if ciq_ticker in _CIQ_YF_OVERRIDES:
            return _CIQ_YF_OVERRIDES[ciq_ticker]
        if ciq_ticker.startswith("ASX:"):
            return ciq_ticker[4:] + ".AX"   # e.g. "ASX:BHP" → "BHP.AX"
        return ""

    spg_by_yf: dict[str, dict] = {}
    for company_name, data in spg_data.items():
        yf_tk = _to_yf(data.get("ticker", ""))
        if yf_tk:
            spg_by_yf[yf_tk] = data

    def _num(val):
        """Convert SPG value to float, returning None for NA/null."""
        if val in (None, "NA", "", "N/A"):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    results: dict[str, dict] = {}
    for tk in tickers:
        spg = spg_by_yf.get(tk)
        if spg is None:
            continue

        commodity = spg.get("commodity", "")
        row: dict = {}

        # NAV per share (USD) — key is date-stamped e.g. "NAV/Share Est [04/21/2026]"
        nav_key = next((k for k in spg if k.startswith("NAV/Share Est")), None)
        nav = _num(spg.get(nav_key)) if nav_key else None
        if nav is not None:
            row["nav_per_shr"] = nav

        # P/NAV using CIQ close price (same currency as NAV)
        ciq_price = _num(spg.get("Last Close Price"))
        if nav is not None and ciq_price is not None and nav > 0 and ciq_price > 0:
            row["p_nav"] = round(ciq_price / nav, 3)

        # Reserves in-situ value ($M) — key is quarter-stamped; match dynamically
        resv_key = next((k for k in spg if k.startswith("Reserves (in-situ $)")), None)
        resv = _num(spg.get(resv_key)) if resv_key else None
        if resv is not None:
            row["reserves_m"] = resv

        # Resources in-situ value ($M) — same dynamic key matching
        rsrc_key = next((k for k in spg if k.startswith("Resources (in-situ $)")), None)
        rsrc = _num(spg.get(rsrc_key)) if rsrc_key else None
        if rsrc is not None:
            row["resources_m"] = rsrc

        # AISC — store only the commodity-appropriate unit to avoid bogus cross-unit values.
        aisc_oz_key = next((k for k in spg if k.startswith("AISC ($/oz)")), None)
        aisc_t_key  = next((k for k in spg if k.startswith("AISC ($/t)")),  None)
        aisc_lb_key = next((k for k in spg if k.startswith("AISC ($/lb)")), None)

        aisc_oz = _num(spg.get(aisc_oz_key)) if aisc_oz_key else None
        aisc_t  = _num(spg.get(aisc_t_key))  if aisc_t_key  else None
        aisc_lb = _num(spg.get(aisc_lb_key)) if aisc_lb_key else None

        comm = commodity.lower()
        if comm in ("gold", "silver"):
            # Only $/oz is meaningful for precious metals
            if aisc_oz is not None and 50 < aisc_oz < 5000:
                row["aisc_per_oz"] = round(aisc_oz, 2)
        elif comm == "uranium":
            # Only $/lb U3O8 is meaningful
            if aisc_lb is not None and 0.5 < aisc_lb < 200:
                row["aisc_per_lb"] = round(aisc_lb, 4)
        elif comm in ("copper", "zinc", "nickel", "iron ore"):
            # $/tonne and $/lb are both meaningful; filter out implausibly large values
            if aisc_t is not None and 0 < aisc_t < 50_000:
                row["aisc_per_t"] = round(aisc_t, 2)
            if aisc_lb is not None and 0 < aisc_lb < 20:
                row["aisc_per_lb"] = round(aisc_lb, 4)
        else:
            # Diversified / other — accept any reasonable value
            if aisc_oz is not None and 50 < aisc_oz < 5000:
                row["aisc_per_oz"] = round(aisc_oz, 2)
            if aisc_t is not None and 0 < aisc_t < 50_000:
                row["aisc_per_t"] = round(aisc_t, 2)
            if aisc_lb is not None and 0 < aisc_lb < 20:
                row["aisc_per_lb"] = round(aisc_lb, 4)

        # Production Cost ($/t) — confirmed CIQ mnemonic: SNL_PRODUCTION_COST_TONNE
        prod_key = next((k for k in spg if k.startswith("Production Cost ($/t)")), None)
        prod_t   = _num(spg.get(prod_key)) if prod_key else None
        if prod_t is not None and 0 < prod_t < 50_000:
            row["production_cost_t"] = round(prod_t, 2)

        # ── Primary Grade (Reserves + Resources) ─────────────────────────
        _grade_map = {
            # commodity name (lower) → (json_label_prefix, field_key, unit)
            "gold":       ("Grade Au (g/t)",  "grade_gpt", "g/t"),
            "silver":     ("Grade Ag (g/t)",  "grade_gpt", "g/t"),
            "copper":     ("Grade Cu (%)",    "grade_pct", "%"),
            "uranium":    ("Grade U3O8 (%)",  "grade_pct", "%"),
            "nickel":     ("Grade Ni (%)",    "grade_pct", "%"),
            "zinc":       ("Grade Zn (%)",    "grade_pct", "%"),
            "lithium":    ("Grade Li (%)",    "grade_pct", "%"),
            "potash":     ("Grade K2O (%)",   "grade_pct", "%"),
            "iron ore":   ("Grade Fe (%)",    "grade_pct", "%"),
            "rare earths":("Grade TREO (%)",  "grade_pct", "%"),
            "manganese":  ("Grade Mn (%)",    "grade_pct", "%"),
        }
        comm_lower = commodity.lower()
        grade_info = _grade_map.get(comm_lower)
        if grade_info:
            label_prefix, field_key, _unit = grade_info
            gk = next((k for k in spg if k.startswith(label_prefix)), None)
            gv = _num(spg.get(gk)) if gk else None
            if gv is not None and gv > 0:
                row[field_key] = round(gv, 4)

        # ── Cash Cost ─────────────────────────────────────────────────────
        cc_oz_key = next((k for k in spg if k.startswith("Cash Cost ($/oz)")),  None)
        cc_t_key  = next((k for k in spg if k.startswith("Cash Cost ($/t)")),   None)
        cc_lb_key = next((k for k in spg if k.startswith("Cash Cost ($/lb)")),  None)
        cc_oz = _num(spg.get(cc_oz_key)) if cc_oz_key else None
        cc_t  = _num(spg.get(cc_t_key))  if cc_t_key  else None
        cc_lb = _num(spg.get(cc_lb_key)) if cc_lb_key else None

        if comm in ("gold", "silver") and cc_oz is not None and 50 < cc_oz < 3000:
            row["cash_cost_oz"] = round(cc_oz, 2)
        elif comm == "uranium" and cc_lb is not None and 0.5 < cc_lb < 150:
            row["cash_cost_lb"] = round(cc_lb, 4)
        elif comm in ("copper", "zinc", "nickel", "iron ore") and cc_t is not None and 0 < cc_t < 40_000:
            row["cash_cost_t"] = round(cc_t, 2)

        # ── Production Volume ──────────────────────────────────────────────
        prod_oz_key = next((k for k in spg if k.startswith("Production (oz)")),    None)
        prod_t_key  = next((k for k in spg if k.startswith("Production (t)")),     None)
        prod_lb_key = next((k for k in spg if k.startswith("Production (lb)")),    None)
        prod_oz = _num(spg.get(prod_oz_key)) if prod_oz_key else None
        prod_t  = _num(spg.get(prod_t_key))  if prod_t_key  else None
        prod_lb = _num(spg.get(prod_lb_key)) if prod_lb_key else None

        if comm in ("gold", "silver"):
            if prod_oz is not None and prod_oz > 0:
                row["production_oz"] = round(prod_oz, 0)
        elif comm == "uranium":
            if prod_lb is not None and prod_lb > 0:
                row["production_lb"] = round(prod_lb, 0)
        elif comm in ("copper", "zinc", "nickel", "lithium", "iron ore"):
            if prod_t is not None and prod_t > 0:
                row["production_t"] = round(prod_t, 0)
        else:
            # Diversified / potash — accept any reasonable value
            if prod_oz is not None and prod_oz > 0:
                row["production_oz"] = round(prod_oz, 0)
            if prod_t is not None and prod_t > 0:
                row["production_t"] = round(prod_t, 0)

        # ── Average Realized Price ─────────────────────────────────────────
        real_oz_key = next((k for k in spg if k.startswith("Realized Price ($/oz)")),  None)
        real_t_key  = next((k for k in spg if k.startswith("Realized Price ($/t)")),   None)
        real_lb_key = next((k for k in spg if k.startswith("Realized Price ($/lb)")),  None)
        real_oz = _num(spg.get(real_oz_key)) if real_oz_key else None
        real_t  = _num(spg.get(real_t_key))  if real_t_key  else None
        real_lb = _num(spg.get(real_lb_key)) if real_lb_key else None

        if comm in ("gold", "silver"):
            if real_oz is not None and real_oz > 50:
                row["realized_price_oz"] = round(real_oz, 2)
        elif comm == "uranium":
            if real_lb is not None and real_lb > 0.5:
                row["realized_price_lb"] = round(real_lb, 4)
        elif comm in ("copper", "zinc", "nickel", "lithium", "iron ore"):
            if real_t is not None and real_t > 0:
                row["realized_price_t"] = round(real_t, 2)
        else:
            if real_oz is not None and real_oz > 50:
                row["realized_price_oz"] = round(real_oz, 2)

        # ── Primary Contained Metal in Reserves ───────────────────────────
        cont_oz_key = next((k for k in spg if k.startswith("Contained Reserves (oz)")), None)
        cont_lb_key = next((k for k in spg if k.startswith("Contained Reserves (lb)")), None)
        cont_oz = _num(spg.get(cont_oz_key)) if cont_oz_key else None
        cont_lb = _num(spg.get(cont_lb_key)) if cont_lb_key else None

        if comm in ("gold", "silver"):
            if cont_oz is not None and cont_oz > 0:
                row["contained_reserves_oz"] = round(cont_oz, 0)
        elif comm == "uranium":
            if cont_lb is not None and cont_lb > 0:
                row["contained_reserves_lb"] = round(cont_lb, 0)
        elif comm in ("copper", "zinc", "nickel", "lithium"):
            # lb is the industry standard for EV/lb reserve (copper)
            if cont_lb is not None and cont_lb > 0:
                row["contained_reserves_lb"] = round(cont_lb, 0)
        else:
            if cont_oz is not None and cont_oz > 0:
                row["contained_reserves_oz"] = round(cont_oz, 0)

        # ── Reserve Life Index (computed) ─────────────────────────────────
        rli = None
        if "contained_reserves_oz" in row and "production_oz" in row:
            p = row["production_oz"]
            c = row["contained_reserves_oz"]
            if p and p > 0:
                rli = round(c / p, 1)
        elif "contained_reserves_lb" in row and "production_lb" in row:
            p = row["production_lb"]
            c = row["contained_reserves_lb"]
            if p and p > 0:
                rli = round(c / p, 1)
        if rli is not None and 0 < rli < 100:
            row["reserve_life"] = rli

        if row:
            results[tk] = row

    log.info(f"[SPG] Overlay loaded for {len(results)}/{len(tickers)} tickers")
    return results


def _snl_overlay(tickers: list[str]) -> dict[str, dict]:
    """
    Pull production/cost metrics from SNL Metals & Mining via Snowflake.
    Fills spg_ fields for tickers in our SNL mapping that don't already have CIQ data.
    Live query — not stored locally (trial §3.2.2 compliant).
    Returns dict keyed by Yahoo Finance ticker → {field_suffix: value}.
    """
    try:
        from data import snl_client
    except ImportError:
        return {}

    if not snl_client.is_configured():
        return {}

    try:
        batch = snl_client.get_batch_metrics("2024Y")
    except Exception as e:
        log.warning(f"[SNL] Batch metrics failed: {e}")
        return {}

    _PRIORITY = ["Gold", "Silver", "Copper", "Zinc",
                 "Nickel", "Uranium", "PGM", "Iron Ore"]

    # Commodity → which unit column to trust
    _UNIT_MAP = {
        "Gold":     "oz",
        "Silver":   "oz",
        "PGM":      "oz",
        "Copper":   "t",
        "Zinc":     "t",
        "Nickel":   "t",
        "Iron Ore": "t",
        "Uranium":  "lb",
    }

    results: dict[str, dict] = {}
    for tk in tickers:
        snl_key = snl_client.get_snl_key(tk)
        if not snl_key or snl_key not in batch:
            continue

        comm_data = batch[snl_key]

        # Choose the commodity with the best cost coverage:
        # prefer one that has AISC data; fall back to any available commodity.
        def _has_aisc(r: dict) -> bool:
            return any(
                r.get(k) is not None and float(r.get(k) or 0) > 0
                for k in ("AISC_OZ", "AISC_T", "AISC_LB")
            )

        chosen_comm = (
            next((c for c in _PRIORITY if c in comm_data and _has_aisc(comm_data[c])), None)
            or next((c for c in _PRIORITY if c in comm_data), None)
        )
        if not chosen_comm:
            continue
        chosen = comm_data[chosen_comm]
        unit   = _UNIT_MAP.get(chosen_comm, "oz")

        def _num(val):
            if val is None:
                return None
            try:
                v = float(val)
                return v if np.isfinite(v) and v > 0 else None
            except (TypeError, ValueError):
                return None

        row: dict = {}

        if unit == "oz":
            aisc = _num(chosen.get("AISC_OZ"))
            cc   = _num(chosen.get("CASH_COST_OZ"))
            prod = _num(chosen.get("PROD_OZ"))
            real = _num(chosen.get("REALIZED_PRICE_OZ"))
            # Sanity bounds for $/oz
            if aisc and 50 < aisc < 6000:   row["aisc_per_oz"]       = round(aisc, 2)
            if cc   and 50 < cc   < 5000:   row["cash_cost_oz"]      = round(cc,   2)
            if prod and prod > 0:            row["production_oz"]     = round(prod,  0)
            if real and real > 50:           row["realized_price_oz"] = round(real,  2)

        elif unit == "t":
            aisc = _num(chosen.get("AISC_T"))
            cc   = _num(chosen.get("CASH_COST_T"))
            prod = _num(chosen.get("PROD_T"))
            real = _num(chosen.get("REALIZED_PRICE_T"))
            if aisc and 0 < aisc < 50_000:  row["aisc_per_t"]        = round(aisc, 2)
            if cc   and 0 < cc   < 40_000:  row["cash_cost_t"]       = round(cc,   2)
            if prod and prod > 0:            row["production_t"]      = round(prod,  0)
            if real and real > 0:            row["realized_price_t"]  = round(real,  2)

        elif unit == "lb":
            aisc = _num(chosen.get("AISC_LB"))
            cc   = _num(chosen.get("CASH_COST_LB"))
            prod = _num(chosen.get("PROD_LB"))
            real = _num(chosen.get("REALIZED_PRICE_LB"))
            if aisc and 0 < aisc < 300:     row["aisc_per_lb"]       = round(aisc, 4)
            if cc   and 0 < cc   < 200:     row["cash_cost_lb"]      = round(cc,   4)
            if prod and prod > 0:           row["production_lb"]     = round(prod,  0)
            if real and real > 0:           row["realized_price_lb"] = round(real,  4)

        if row:
            results[tk] = row

    log.info(f"[SNL] Overlay loaded for {len(results)}/{len(tickers)} tickers")
    return results


# ── Snowflake full SPG replacement ────────────────────────────────────────────

def _spg_overlay_snowflake(tickers: list[str]) -> dict[str, dict]:
    """
    Snowflake-based replacement for the CIQ JSON SPG overlay.
    Combines production/AISC + reserves + grade from SNL Metals & Mining.
    Used when config.SPG_SOURCE == "snowflake".
    Live queries — not stored (trial §3.2.2 compliant).
    Returns dict keyed by Yahoo Finance ticker → field dict (same structure as _spg_overlay).
    """
    try:
        from data import snl_client
    except ImportError:
        return {}

    if not snl_client.is_configured():
        return {}

    # ── Fetch all SNL tables (sequential — each opens its own connection) ──────
    prod_data: dict = {}
    resv_data: dict = {}
    grade_data: dict = {}

    try:
        prod_data = snl_client.get_batch_metrics("2024Y")
    except Exception as e:
        log.warning(f"[SNL] get_batch_metrics failed: {e}")

    try:
        resv_data = snl_client.get_batch_reserves()
    except Exception as e:
        log.warning(f"[SNL] get_batch_reserves failed: {e}")

    try:
        grade_data = snl_client.get_batch_grade()
    except Exception as e:
        log.warning(f"[SNL] get_batch_grade failed: {e}")

    _PRIORITY = ["Gold", "Silver", "Copper", "Zinc",
                 "Nickel", "Uranium", "PGM", "Iron Ore"]
    _UNIT_MAP = {
        "Gold": "oz", "Silver": "oz", "PGM": "oz",
        "Copper": "t", "Zinc": "t", "Nickel": "t", "Iron Ore": "t",
        "Uranium": "lb",
    }

    def _n(val) -> float | None:
        if val is None:
            return None
        try:
            v = float(val)
            return v if np.isfinite(v) and v > 0 else None
        except (TypeError, ValueError):
            return None

    def _has_aisc(r: dict) -> bool:
        return any(_n(r.get(k)) is not None for k in ("AISC_OZ", "AISC_T", "AISC_LB"))

    results: dict[str, dict] = {}

    for tk in tickers:
        snl_key = snl_client.get_snl_key(tk)
        if not snl_key:
            continue

        row: dict = {}

        # ── Production + AISC ─────────────────────────────────────────────────
        comm_data = prod_data.get(snl_key, {})
        if comm_data:
            chosen_comm = (
                next((c for c in _PRIORITY if c in comm_data and _has_aisc(comm_data[c])), None)
                or next((c for c in _PRIORITY if c in comm_data), None)
            )
            if chosen_comm:
                chosen = comm_data[chosen_comm]
                unit   = _UNIT_MAP.get(chosen_comm, "oz")

                if unit == "oz":
                    aisc = _n(chosen.get("AISC_OZ"))
                    cc   = _n(chosen.get("CASH_COST_OZ"))
                    prod = _n(chosen.get("PROD_OZ"))
                    real = _n(chosen.get("REALIZED_PRICE_OZ"))
                    if aisc and 50 < aisc < 6000:  row["aisc_per_oz"]       = round(aisc, 2)
                    if cc   and 50 < cc   < 5000:  row["cash_cost_oz"]      = round(cc,   2)
                    if prod:                        row["production_oz"]     = round(prod,  0)
                    if real and real > 50:          row["realized_price_oz"] = round(real,  2)
                elif unit == "t":
                    aisc = _n(chosen.get("AISC_T"))
                    cc   = _n(chosen.get("CASH_COST_T"))
                    prod = _n(chosen.get("PROD_T"))
                    real = _n(chosen.get("REALIZED_PRICE_T"))
                    if aisc and 0 < aisc < 50_000: row["aisc_per_t"]        = round(aisc, 2)
                    if cc   and 0 < cc   < 40_000: row["cash_cost_t"]       = round(cc,   2)
                    if prod:                        row["production_t"]      = round(prod,  0)
                    if real and real > 0:           row["realized_price_t"]  = round(real,  2)
                elif unit == "lb":
                    aisc = _n(chosen.get("AISC_LB"))
                    cc   = _n(chosen.get("CASH_COST_LB"))
                    prod = _n(chosen.get("PROD_LB"))
                    real = _n(chosen.get("REALIZED_PRICE_LB"))
                    if aisc and 0 < aisc < 300:    row["aisc_per_lb"]       = round(aisc, 4)
                    if cc   and 0 < cc   < 200:    row["cash_cost_lb"]      = round(cc,   4)
                    if prod:                        row["production_lb"]     = round(prod,  0)
                    if real and real > 0:           row["realized_price_lb"] = round(real,  4)

        # ── Reserves / Resources in-situ value ($M) ───────────────────────────
        rv = resv_data.get(snl_key, {})
        if "reserves_m"  in rv: row["reserves_m"]  = rv["reserves_m"]
        if "resources_m" in rv: row["resources_m"] = rv["resources_m"]

        # ── Grade and contained metals ─────────────────────────────────────────
        gd = grade_data.get(snl_key, {})
        for field in ("grade_gpt", "grade_pct",
                      "contained_reserves_oz", "contained_reserves_lb",
                      "contained_reserves_t"):
            if field in gd:
                row[field] = gd[field]

        # ── Reserve Life Index (computed from SNL contained metal + production) ─
        rli = None
        if "contained_reserves_oz" in row and "production_oz" in row:
            p = row["production_oz"]
            c = row["contained_reserves_oz"]
            if p > 0:
                rli = round(c / p, 1)
        elif "contained_reserves_lb" in row and "production_lb" in row:
            p = row["production_lb"]
            c = row["contained_reserves_lb"]
            if p > 0:
                rli = round(c / p, 1)
        if rli is not None and 0 < rli < 100:
            row["reserve_life"] = rli

        if row:
            results[tk] = row

    # ── Mine life from local mine_econ tables (no extra Snowflake query) ─────────
    try:
        mine_life_data  = snl_client.get_batch_mine_life_local()
        global_rank_data = snl_client.get_batch_global_rank_local()
    except Exception as _e:
        log.warning(f"[SNL local] mine_life/rank failed: {_e}")
        mine_life_data = {}
        global_rank_data = {}

    _RANK_PRIORITY = ["Gold", "Silver", "Copper", "Nickel", "Zinc", "Uranium",
                      "PGM", "Iron Ore"]

    for tk in tickers:
        snl_key = snl_client.get_snl_key(tk)
        if not snl_key:
            continue
        row = results.setdefault(tk, {})

        ml = mine_life_data.get(snl_key)
        if ml and 0 < ml < 100:
            row["mine_life"] = ml

        ranks = global_rank_data.get(snl_key, {})
        if ranks:
            # Pick rank for the most relevant commodity (priority order)
            chosen_rank = next(
                (ranks[c] for c in _RANK_PRIORITY if c in ranks), None
            ) or min(ranks.values())
            row["global_rank"] = chosen_rank

    log.info(f"[SNL/Snowflake] Full SPG overlay for {len(results)}/{len(tickers)} tickers")
    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_all(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch all data for a list of tickers.
    Returns a DataFrame indexed by ticker with all available fields.
    """
    log.info(f"Fetching {len(tickers)} tickers ...")
    _FX_CACHE.clear()   # re-fetch FX rates each run (long-lived scheduler process)

    log.info("  Fetching live commodity spot prices ...")
    fetch_spot_prices()

    log.info("  Fetching fundamentals via Yahoo Finance API ...")
    all_info = _fetch_info_batch(tickers)

    log.info("  Fetching price history & technicals ...")
    all_techs = _price_technicals(tickers)

    if config.SPG_SOURCE == "snowflake":
        log.info("  SNL Snowflake overlay (production + reserves + grade) ...")
        all_spg = _spg_overlay_snowflake(tickers)
        all_snl: dict = {}   # no separate gap-fill needed; Snowflake handles everything
        # Snowflake has no NAV/P-NAV estimates — supplement from CIQ JSON when available.
        _ciq_nav = _spg_overlay(tickers)
        _NAV_FIELDS = ("p_nav", "nav_per_shr")
        for _tk, _ciq_row in _ciq_nav.items():
            for _fld in _NAV_FIELDS:
                if _fld in _ciq_row:
                    _spg_row = all_spg.setdefault(_tk, {})
                    if _fld not in _spg_row:      # never overwrite Snowflake data
                        _spg_row[_fld] = _ciq_row[_fld]
        _nav_count = sum(1 for v in all_spg.values() if "p_nav" in v)
        log.info(f"  CIQ NAV supplement: P/NAV now available for {_nav_count}/{len(tickers)} tickers")
    else:
        log.info("  S&P Capital IQ / SNL Mining overlay (CIQ JSON) ...")
        all_spg = _spg_overlay(tickers)
        log.info("  SNL Metals & Mining overlay (Snowflake live, gap-fill) ...")
        all_snl = _snl_overlay(tickers)

    rows = []
    for tk in tickers:
        info = all_info.get(tk, {})
        tech = all_techs.get(tk, {})
        spg  = all_spg.get(tk, {})
        snl  = all_snl.get(tk, {})

        # SNL fills gaps only — CIQ data takes precedence
        merged_spg: dict = {}
        merged_spg.update(snl)   # SNL base
        merged_spg.update(spg)   # CIQ overwrites where present

        row  = {"ticker": tk}
        row.update(info)
        row.update(tech)
        row.update({f"spg_{k}": v for k, v in merged_spg.items()})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("ticker")
    log.info(f"Fetch complete: {len(df)} stocks, {len(df.columns)} fields.")
    return df


# ── SNL local SQLite enrichment for scoring pipeline ─────────────────────────
# This is called from scheduler/jobs.py BEFORE compute_scores() so that
# SNL-derived metrics (p_insitu, prod_growth, best_irr, ev_per_oz_rr)
# are available to the mining_score() function.

def apply_snl_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Read per-ticker SNL metrics from local SQLite and merge into df (index=ticker).

    Adds columns (all float, NaN when unavailable):
      snl_p_insitu        — market cap / in-situ R&R value (ratio; lower = cheaper)
      snl_prod_growth_pct — forward guidance vs actual production growth %
      snl_best_irr        — best post-tax IRR from FS/PEA (%)
      snl_ev_per_oz_rr    — EV / total R&R oz (raw $/oz; used for peer-pct score)

    Also fills spg_ gaps (aisc, production, reserves, grade) from SNL.

    Raises no exceptions — logs warnings and returns df unchanged on any error.
    """
    import sqlite3 as _sl3, json as _js2
    _db  = str(config.DB_PATH)
    _map = os.path.join(os.path.dirname(__file__), "..", "_asx_snl_ticker_mapping.json")
    if not os.path.exists(_db):
        log.debug("apply_snl_for_scoring: DB not found, skipping")
        return df
    try:
        with open(_map) as _f:
            _mapping = _js2.load(_f)
    except Exception:
        log.debug("apply_snl_for_scoring: mapping not found, skipping")
        return df

    _key2ticker = {str(v["snl_key"]): k for k, v in _mapping.items()}
    _PRIO = ["Gold", "Silver", "Platinum", "Palladium", "PGM",
             "Copper", "Nickel", "Zinc", "Uranium", "Iron Ore"]

    def _best(group):
        if group.empty:
            return None
        # Prefer real date-periods (YYYYQN, YYYYY) over special tokens like "MRQ"
        _date_periods = group[group["period"].str.match(r"^\d{4}", na=False)]
        _pool = _date_periods if not _date_periods.empty else group
        sub = _pool[_pool["period"] == _pool["period"].max()]
        for _c in _PRIO:
            _r = sub[sub["commodity"] == _c]
            if not _r.empty:
                return _r.iloc[0]
        return sub.iloc[0]

    try:
        _conn = _sl3.connect(_db)
        _cur_yr = pd.Timestamp.now("UTC").year

        _prod = pd.read_sql_query(
            "SELECT snl_key,period,commodity,prod_oz,prod_t,prod_lb,"
            "aisc_oz,aisc_t,aisc_lb,cash_cost_oz,realized_price_oz "
            "FROM snl_company_production WHERE period>='2022Y'", _conn)
        _prod["snl_key"] = _prod["snl_key"].astype(str)

        _rr = pd.read_sql_query(
            "SELECT snl_key,period,commodity,grade_gpt,grade_pct,"
            "contained_reserves_oz,contained_rr_oz,contained_rr_lb "
            "FROM snl_company_rr WHERE period>='2022Y'", _conn)
        _rr["snl_key"] = _rr["snl_key"].astype(str)

        _ins = pd.read_sql_query(
            "SELECT i.snl_key,i.insitu_reserves_m,i.insitu_rr_m "
            "FROM snl_company_insitu i "
            "INNER JOIN (SELECT snl_key,MAX(period) mp FROM snl_company_insitu "
            "GROUP BY snl_key) m ON i.snl_key=m.snl_key AND i.period=m.mp", _conn)
        _ins["snl_key"] = _ins["snl_key"].astype(str)

        _proj = pd.read_sql_query(
            "SELECT snl_key, estimate_period, description, "
            "prod_high_oz, prod_low_oz "
            f"FROM snl_company_projections WHERE estimate_period >= '{_cur_yr - 1}Y' "
            "ORDER BY snl_key, estimate_period", _conn)
        _proj["snl_key"] = _proj["snl_key"].astype(str)

        _study_rank = {"Full Feasibility": 4, "Feasibility": 4,
                       "Prefeasibility": 3, "Preliminary Economic Assessment": 2,
                       "Mine Plan": 1}
        _studies = pd.read_sql_query(
            "SELECT o.snl_key, s.posttax_irr_pct, s.posttax_npv_m, "
            "s.study_type, s.study_year "
            "FROM snl_property_studies s "
            "JOIN snl_property_owner o ON s.property_id = o.property_id "
            "WHERE s.posttax_irr_pct IS NOT NULL OR s.posttax_npv_m IS NOT NULL "
            "ORDER BY o.snl_key, s.study_year DESC", _conn)
        _studies["snl_key"] = _studies["snl_key"].astype(str)
        _studies["_rank"] = _studies["study_type"].map(
            lambda t: _study_rank.get(t, 0))
        _conn.close()

    except Exception as _ex:
        log.warning(f"apply_snl_for_scoring: DB read failed: {_ex}")
        return df

    _rows = []
    for _sk, _tk in _key2ticker.items():
        _r = {"ticker": _tk}

        # Production (actual)
        _bp = _best(_prod[_prod["snl_key"] == _sk])
        if _bp is not None:
            _r.update({
                "snl_aisc_oz":  _bp.get("aisc_oz"),
                "snl_aisc_t":   _bp.get("aisc_t"),
                "snl_aisc_lb":  _bp.get("aisc_lb"),
                "snl_cc_oz":    _bp.get("cash_cost_oz"),
                "snl_prod_oz":  _bp.get("prod_oz"),
                "snl_prod_t":   _bp.get("prod_t"),
                "snl_prod_lb":  _bp.get("prod_lb"),
                "snl_rp_oz":    _bp.get("realized_price_oz"),
            })

        # R&R (precious)
        _rr_prec = _rr[(_rr["snl_key"] == _sk) &
                       (_rr["commodity"].isin(["Gold","Silver","Platinum","Palladium","PGM"]))]
        _bprec = _best(_rr_prec) if not _rr_prec.empty else None
        if _bprec is not None:
            _r.update({
                "snl_grade_gpt": _bprec.get("grade_gpt"),
                "snl_rsv_oz":    _bprec.get("contained_reserves_oz"),
                "snl_rr_oz":     _bprec.get("contained_rr_oz"),
            })

        # R&R (base)
        _rr_base = _rr[(_rr["snl_key"] == _sk) &
                       (~_rr["commodity"].isin(["Gold","Silver","Platinum","Palladium","PGM"]))]
        _bbase = _best(_rr_base) if not _rr_base.empty else None
        if _bbase is not None:
            _r.update({
                "snl_grade_pct": _bbase.get("grade_pct"),
                "snl_rsv_lb":    _bbase.get("contained_reserves_lb"),
                "snl_rr_lb":     _bbase.get("contained_rr_lb"),
            })

        # In-situ
        _ig = _ins[_ins["snl_key"] == _sk]
        if not _ig.empty:
            _r["snl_insitu_rr_m"]  = _ig.iloc[0]["insitu_rr_m"]
            _r["snl_insitu_rsv_m"] = _ig.iloc[0]["insitu_reserves_m"]

        # Production growth from forward guidance
        _pk = _proj[_proj["snl_key"] == _sk].copy()
        if not _pk.empty:
            _latest = _pk["estimate_period"].max()
            _fwd_row = _pk[_pk["estimate_period"] == _latest]
            _prod_fwd = _fwd_row[
                _fwd_row["description"].str.contains("Production|production", na=False)]
            if _prod_fwd.empty:
                _prod_fwd = _fwd_row
            _fr = _prod_fwd.iloc[0]
            _hi, _lo = _fr["prod_high_oz"], _fr["prod_low_oz"]
            if (_hi and _lo and float(_hi) > 1_000 and float(_lo) > 1_000):
                _r["snl_fwd_prod_oz"] = (float(_hi) + float(_lo)) / 2

        _actual = _r.get("snl_prod_oz")
        _fwd    = _r.get("snl_fwd_prod_oz")
        if (_actual and _fwd
                and float(_actual) > 1_000 and float(_actual) < 15_000_000
                and float(_fwd)    > 1_000 and float(_fwd)    < 15_000_000):
            _ratio = float(_fwd) / float(_actual)
            if 0.1 <= _ratio <= 4.0:
                _r["snl_prod_growth_pct"] = round((_ratio - 1.0) * 100, 1)

        # Best FS/PEA IRR
        _stk = _studies[_studies["snl_key"] == _sk]
        if not _stk.empty:
            _best_s = _stk.sort_values(["_rank", "study_year"],
                                        ascending=[False, False]).iloc[0]
            _irr = _best_s.get("posttax_irr_pct")
            if _irr is not None:
                _r["snl_best_irr"] = float(_irr)

        _rows.append(_r)

    _enrich = pd.DataFrame(_rows)
    if _enrich.empty:
        return df

    # ── Merge into df (index=ticker) ──────────────────────────────────────────
    # _asx_snl_ticker_mapping.json keys are bare tickers (BHP, RIO, …).
    # df.index may be exchange-suffixed (BHP.AX, RIO.AX, …).
    # Build a lookup: bare_ticker → full_ticker from df.index.
    _bare2full: dict[str, str] = {}
    for _full in df.index:
        _bare = str(_full).split(".")[0]
        _bare2full[_bare] = str(_full)

    # Re-map _enrich tickers from bare → full so the merge key aligns.
    _enrich["ticker"] = _enrich["ticker"].map(
        lambda t: _bare2full.get(str(t), str(t)))

    df = df.reset_index().merge(_enrich, on="ticker", how="left").set_index("ticker")
    df.index.name = "ticker"   # preserve

    # Fill spg_ gaps from SNL
    # Support both camelCase (fresh fetch) and snake_case (DB reload)
    _mc_col  = "marketCap"      if "marketCap"      in df.columns else "market_cap"
    _ev_col  = "enterpriseValue" if "enterpriseValue" in df.columns else "enterprise_value"
    for _spg, _snl in [
        ("spg_aisc_per_oz",           "snl_aisc_oz"),
        ("spg_aisc_per_t",            "snl_aisc_t"),
        ("spg_aisc_per_lb",           "snl_aisc_lb"),
        ("spg_cash_cost_oz",          "snl_cc_oz"),
        ("spg_production_oz",         "snl_prod_oz"),
        ("spg_grade_gpt",             "snl_grade_gpt"),
        ("spg_grade_pct",             "snl_grade_pct"),
        ("spg_contained_reserves_oz", "snl_rsv_oz"),
        ("spg_contained_reserves_lb", "snl_rsv_lb"),
    ]:
        if _spg in df.columns and _snl in df.columns:
            _mask = df[_spg].isna() & df[_snl].notna()
            df.loc[_mask, _spg] = pd.to_numeric(df.loc[_mask, _snl], errors="coerce")

    # Fill spg_reserves_m gap from in-situ reserves value
    if "snl_insitu_rsv_m" in df.columns:
        _irsv = pd.to_numeric(df["snl_insitu_rsv_m"], errors="coerce")
        if "spg_reserves_m" not in df.columns:
            df["spg_reserves_m"] = np.nan
        _miss = df["spg_reserves_m"].isna()
        df.loc[_miss & _irsv.notna(), "spg_reserves_m"] = _irsv[_miss & _irsv.notna()]

    # Derive P/in-situ ratio
    if "snl_insitu_rr_m" in df.columns:
        _iv = pd.to_numeric(df["snl_insitu_rr_m"], errors="coerce")
        _mc = pd.to_numeric(df.get(_mc_col, pd.Series(np.nan, index=df.index)), errors="coerce")
        df["snl_p_insitu"] = np.where(
            _iv.notna() & (_iv > 0) & _mc.notna() & (_mc > 0),
            (_mc / (_iv * 1e6)).round(4), np.nan)

    # Derive EV/oz R&R
    if "snl_rr_oz" in df.columns:
        _rr_oz = pd.to_numeric(df["snl_rr_oz"], errors="coerce")
        _ev    = pd.to_numeric(df.get(_ev_col, pd.Series(np.nan, index=df.index)), errors="coerce")
        df["snl_ev_per_oz_rr"] = np.where(
            _rr_oz.notna() & (_rr_oz > 0) & _ev.notna() & (_ev > 0),
            (_ev / _rr_oz).round(0), np.nan)

    # Ensure all SNL scorer columns are numeric
    for _col in ("snl_p_insitu", "snl_prod_growth_pct", "snl_best_irr", "snl_ev_per_oz_rr"):
        if _col not in df.columns:
            df[_col] = np.nan
        df[_col] = pd.to_numeric(df[_col], errors="coerce")

    log.info(
        f"[SNL] Enrichment complete — "
        f"p_insitu: {df['snl_p_insitu'].notna().sum()}, "
        f"prod_growth: {df['snl_prod_growth_pct'].notna().sum()}, "
        f"irr: {df['snl_best_irr'].notna().sum()}, "
        f"ev_rr: {df['snl_ev_per_oz_rr'].notna().sum()} tickers"
    )
    return df
