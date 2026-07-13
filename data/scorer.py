"""
Multi-factor undervaluation scorer for Australian mining stocks.

Score architecture (each sub-score 0–100, higher = more undervalued / attractive):
  1. Valuation Score   (30%) — P/B, EV/EBITDA, P/CF, P/NAV vs sector peers
  2. Health Score      (20%) — balance sheet strength, cash runway
  3. Momentum Score    (15%) — 52-wk position, RSI (contrarian: low = cheap)
  4. Mining Score      (25%) — S&P/SNL: AISC margin, reserves backing, NAV discount
  5. Commodity Score   ( 5%) — outlook multiplier per commodity
  6. Stage Score       ( 5%) — stage-specific opportunity premium
  Composite = weighted sum → clamped [0, 100]
"""
import numpy as np
import pandas as pd

from config import SCORE_WEIGHTS, COMMODITY_OUTLOOK, COMMODITY_SPOT


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _percentile_score(series: pd.Series, val: float, invert: bool = False) -> float:
    """
    Score val relative to its peer distribution.
    invert=True  → lower value is better (e.g. P/B, EV/EBITDA)
    invert=False → higher value is better (e.g. cash ratio)
    Returns 0–100.
    """
    clean = series.dropna()
    if len(clean) < 3 or np.isnan(val):
        return 50.0   # neutral when insufficient data
    pct = (clean < val).mean() * 100        # percentile of val in peer group
    return (100 - pct) if invert else pct


def _clip(val: float, lo: float = 0, hi: float = 100) -> float:
    return float(np.clip(val, lo, hi))


# ── sub-scores ─────────────────────────────────────────────────────────────────

def valuation_score(row: pd.Series, peers: pd.DataFrame) -> float:
    """
    Lower P/B, EV/EBITDA, P/CF, P/NAV → higher score.
    P/NAV (Price to Net Asset Value) is the primary mining valuation metric
    and receives 40% weight when available.
    Falls back to P/B alone for explorers/developers with no earnings.
    """
    scores = []

    pb = _safe(row.get("priceToBook"))
    if not np.isnan(pb) and pb > 0:
        if   pb < 0.5:  s = 95
        elif pb < 1.0:  s = 80
        elif pb < 1.5:  s = 65
        elif pb < 2.5:  s = 45
        elif pb < 4.0:  s = 25
        else:           s = 10
        peer_s = _percentile_score(peers["priceToBook"].where(peers["priceToBook"] > 0), pb, invert=True)
        scores.append(0.5 * s + 0.5 * peer_s)

    ev      = _safe(row.get("enterpriseValue"))
    ebitda  = _safe(row.get("ebitda"))
    if not np.isnan(ev) and not np.isnan(ebitda) and ebitda > 0:
        ev_ebitda = ev / ebitda
        if   ev_ebitda < 4:  s = 95
        elif ev_ebitda < 6:  s = 80
        elif ev_ebitda < 10: s = 60
        elif ev_ebitda < 15: s = 35
        else:                s = 15
        peer_ev = peers["enterpriseValue"] / peers["ebitda"].replace(0, np.nan)
        peer_s  = _percentile_score(peer_ev.where(peer_ev > 0), ev_ebitda, invert=True)
        scores.append(0.5 * s + 0.5 * peer_s)

    ocf = _safe(row.get("operatingCashflow"))
    mcap = _safe(row.get("marketCap"))
    if not np.isnan(ocf) and not np.isnan(mcap) and ocf > 0 and mcap > 0:
        p_cf = mcap / ocf
        if   p_cf < 5:   s = 95
        elif p_cf < 10:  s = 80
        elif p_cf < 15:  s = 60
        elif p_cf < 25:  s = 35
        else:            s = 15
        peer_pcf = peers["marketCap"] / peers["operatingCashflow"].replace(0, np.nan)
        peer_s   = _percentile_score(peer_pcf.where(peer_pcf > 0), p_cf, invert=True)
        scores.append(0.5 * s + 0.5 * peer_s)

    # ── FCF Yield: cash generative producers score better ───────────────────
    fcf  = _safe(row.get("freeCashflow"))
    mcap_v = _safe(row.get("marketCap"))
    if not np.isnan(fcf) and not np.isnan(mcap_v) and mcap_v > 0 and fcf > 0:
        fcf_yield_pct = fcf / mcap_v * 100
        if   fcf_yield_pct >= 15: s = 95
        elif fcf_yield_pct >= 10: s = 85
        elif fcf_yield_pct >=  7: s = 72
        elif fcf_yield_pct >=  4: s = 55
        elif fcf_yield_pct >=  2: s = 38
        else:                     s = 22
        scores.append(s)

    # ── Analyst Consensus Upside ─────────────────────────────────────────────
    # "The market's forward-looking value gap" — valid for all stages.
    # Require ≥2 covering analysts to avoid single-analyst distortion.
    analyst_count = _safe(row.get("analyst_count"))
    analyst_up    = _safe(row.get("analyst_upside"))
    if (not np.isnan(analyst_up)
            and not np.isnan(analyst_count)
            and analyst_count >= 2):
        if   analyst_up > 60:  s = 95
        elif analyst_up > 40:  s = 85
        elif analyst_up > 25:  s = 70
        elif analyst_up > 15:  s = 55
        elif analyst_up > 5:   s = 42
        elif analyst_up > -5:  s = 30
        else:                  s = 16
        scores.append(s)

    # ── P/NAV: primary mining valuation metric (S&P/SNL data) ──────────────
    p_nav = _safe(row.get("spg_p_nav"))
    if not np.isnan(p_nav) and p_nav > 0:
        if   p_nav < 0.60:  s = 98
        elif p_nav < 0.75:  s = 90
        elif p_nav < 0.90:  s = 78
        elif p_nav < 1.00:  s = 65
        elif p_nav < 1.25:  s = 48
        elif p_nav < 1.75:  s = 30
        elif p_nav < 2.50:  s = 18
        else:               s = 8
        # P/NAV gets 40% blend weight when available
        if scores:
            base_score = float(np.mean(scores))
            return _clip(0.40 * s + 0.60 * base_score)
        else:
            return _clip(float(s))

    if not scores:
        return 50.0
    return _clip(float(np.mean(scores)))


def health_score(row: pd.Series) -> float:
    """
    Balance sheet quality: cash, debt, liquidity — stage-aware.

    All stages: debt-to-equity, current ratio, cash % of market cap, FCF signal.
    Producers only: ROE, operating margins, gross margins (meaningful signals only
      when the company generates revenue; penalising explorers on profitability
      metrics creates noise, not signal).
    """
    scores: list[float] = []

    stage = str(row.get("stage", ""))
    is_producer = any(s in stage for s in ("Producer", "Royalty"))

    # ── Universal balance-sheet signals ──────────────────────────────────────

    # Debt-to-equity (lower = better; explorers with no debt score well naturally)
    de = _safe(row.get("debtToEquity"))
    if not np.isnan(de) and de >= 0:
        if   de < 20:   scores.append(95)
        elif de < 50:   scores.append(80)
        elif de < 100:  scores.append(60)
        elif de < 200:  scores.append(35)
        else:           scores.append(10)

    # Current ratio (higher = better)
    cr = _safe(row.get("currentRatio"))
    if not np.isnan(cr) and cr > 0:
        if   cr > 3:    scores.append(90)
        elif cr > 2:    scores.append(75)
        elif cr > 1.5:  scores.append(60)
        elif cr > 1:    scores.append(40)
        else:           scores.append(15)

    # Cash as % of market cap (critical for explorers — runway)
    cash = _safe(row.get("totalCash"))
    mcap = _safe(row.get("marketCap"))
    if not np.isnan(cash) and not np.isnan(mcap) and mcap > 0:
        cash_pct = cash / mcap * 100
        if   cash_pct > 30: scores.append(90)
        elif cash_pct > 15: scores.append(70)
        elif cash_pct > 8:  scores.append(50)
        elif cash_pct > 3:  scores.append(30)
        else:               scores.append(15)

    # FCF signal: positive FCF = cash-generative (strong for producers; mildly
    # penalises explorers burning cash — intentional but gentle via low weight)
    fcf = _safe(row.get("freeCashflow"))
    if not np.isnan(fcf):
        scores.append(80 if fcf > 0 else 30)

    # ── Producer-only profitability signals ──────────────────────────────────
    # Skipped entirely for Explorers / Developers to avoid meaningless negative
    # signals from pre-revenue companies.
    if is_producer:
        # Handle both fresh-fetch (camelCase) and DB-reload (snake_case) names.
        _roe_raw = row.get("returnOnEquity", row.get("return_on_equity"))
        roe = _safe(_roe_raw)
        if not np.isnan(roe):
            roe_pct = roe * 100   # Yahoo returns as decimal (0.20 = 20%)
            if   roe_pct >= 20:  scores.append(92)
            elif roe_pct >= 12:  scores.append(78)
            elif roe_pct >= 6:   scores.append(62)
            elif roe_pct >= 0:   scores.append(45)
            elif roe_pct >= -10: scores.append(28)
            else:                scores.append(12)

        _om_raw = row.get("operatingMargins", row.get("operating_margins"))
        op_mgn = _safe(_om_raw)
        if not np.isnan(op_mgn):
            op_pct = op_mgn * 100
            if   op_pct >= 40:  scores.append(95)
            elif op_pct >= 25:  scores.append(82)
            elif op_pct >= 12:  scores.append(65)
            elif op_pct >= 0:   scores.append(45)
            elif op_pct >= -15: scores.append(25)
            else:               scores.append(10)

        _gm_raw = row.get("grossMargins", row.get("gross_margins"))
        gm = _safe(_gm_raw)
        if not np.isnan(gm):
            gm_pct = gm * 100
            if   gm_pct >= 60:  scores.append(92)
            elif gm_pct >= 45:  scores.append(78)
            elif gm_pct >= 30:  scores.append(60)
            elif gm_pct >= 15:  scores.append(42)
            elif gm_pct >= 0:   scores.append(25)
            else:               scores.append(10)

    return _clip(float(np.mean(scores))) if scores else 50.0


def momentum_score(row: pd.Series) -> float:
    """
    Contrarian momentum: stock near 52-wk low + oversold RSI → higher score.
    We want to buy quality stocks when they're beaten down.
    """
    scores = []

    # 52-week position (0=at low, 100=at high) → invert for score
    pos = _safe(row.get("wk52_position"))
    if not np.isnan(pos):
        # Near 52-wk low → high contrarian score
        scores.append(_clip(100 - pos))

    # RSI (0–100): below 30 = oversold = buy signal
    rsi = _safe(row.get("rsi"))
    if not np.isnan(rsi):
        if   rsi < 25:  scores.append(95)
        elif rsi < 30:  scores.append(85)
        elif rsi < 40:  scores.append(65)
        elif rsi < 50:  scores.append(50)
        elif rsi < 60:  scores.append(35)
        else:           scores.append(20)

    # Price vs 200-day MA: deeply below = more discounted
    vs_ma200 = _safe(row.get("price_vs_ma200"))
    if not np.isnan(vs_ma200):
        if   vs_ma200 < -40: scores.append(90)
        elif vs_ma200 < -25: scores.append(75)
        elif vs_ma200 < -10: scores.append(60)
        elif vs_ma200 <   0: scores.append(50)
        else:                scores.append(30)

    return _clip(float(np.mean(scores))) if scores else 50.0


def _royalty_mining_score(row: pd.Series) -> float:
    """
    Dedicated mining score for Royalty / Streaming companies.

    Royalties are fundamentally different businesses — they have no AISC, no direct
    production, and no in-situ resource ownership. Penalising them on mine-operations
    metrics creates a category error.

    This track uses only signals that are meaningful for royalty economics:
      EV/Revenue multiplier  (35%) — lower = cheaper stream portfolio
      P/NAV                  (30%) — S&P NAV discount; most royalties trade at premium
      P/CF peer-percentile   (25%) — pre-computed; royalties deserve high P/CF so
                                     we rank vs royalty peers, not vs miners
      Balance sheet proxy    (10%) — cash % of market cap (debt headroom)

    Royalties are expected to trade at premium multiples — thresholds are calibrated
    accordingly (EV/Revenue 5–20× is normal; 30× is expensive for a royalty).
    """
    comps: list[tuple[float, float]] = []

    # EV / Revenue  (royalties are valued on stream revenue, not mine EBITDA)
    ev  = _safe(row.get("enterpriseValue", row.get("enterprise_value")))
    rev = _safe(row.get("totalRevenue",    row.get("revenue")))
    if not np.isnan(ev) and not np.isnan(rev) and rev > 0:
        ev_rev = ev / rev
        if   ev_rev <  8:  s = 92   # very cheap stream portfolio
        elif ev_rev < 12:  s = 78
        elif ev_rev < 18:  s = 62
        elif ev_rev < 25:  s = 46
        elif ev_rev < 35:  s = 32
        else:              s = 18   # expensive even for a royalty
        comps.append((s, 0.35))

    # P/NAV (S&P data; royalties often trade at 1.0–2.0× NAV, not at the deep
    # discounts seen for developers — calibrated for royalty premium range)
    p_nav = _safe(row.get("spg_p_nav"))
    if not np.isnan(p_nav) and p_nav > 0:
        if   p_nav < 0.80:  s = 95   # exceptional discount — rare for quality royalty
        elif p_nav < 1.00:  s = 82
        elif p_nav < 1.25:  s = 68
        elif p_nav < 1.60:  s = 52   # fair value for a top-tier royalty
        elif p_nav < 2.20:  s = 36
        else:               s = 20   # very expensive
        comps.append((s, 0.30))

    # P/CF peer-percentile (pre-computed within royalty peer group)
    pcf_s = _safe(row.get("_royalty_pcf_score"))
    if not np.isnan(pcf_s):
        comps.append((pcf_s, 0.25))

    # Cash % of market cap (royalties should maintain balance-sheet strength)
    cash = _safe(row.get("totalCash", row.get("total_cash")))
    mcap = _safe(row.get("marketCap",  row.get("market_cap")))
    if not np.isnan(cash) and not np.isnan(mcap) and mcap > 0:
        cash_pct = cash / mcap * 100
        if   cash_pct > 20: s = 90
        elif cash_pct > 10: s = 72
        elif cash_pct > 5:  s = 55
        elif cash_pct > 2:  s = 40
        else:               s = 25
        comps.append((s, 0.10))

    if not comps:
        return 50.0
    total_w  = sum(w for _, w in comps)
    weighted = sum(sc * w for sc, w in comps)
    return _clip(weighted / total_w)


def mining_score(row: pd.Series) -> float:
    """
    S&P/SNL-powered mining quality score.

    Royalty/Streaming companies are routed to _royalty_mining_score() which uses
    stream-economics metrics instead of mine-operations metrics.

    For all other stages, components weighted proportionally among those available:
      AISC Margin        (30%) — (spot − AISC) / spot → operational profitability
      NAV Discount       (20%) — P/NAV discount to intrinsic value
      P/In-Situ Ratio    (15%) — market cap / SNL in-situ R&R value; 99% coverage
      Reserves Backing   (15%) — in-situ reserves value / market cap
      EV/oz Production   (15%) — peer-percentile valuation (pre-computed)
      Mine Life          (15%) — SNL mine_econ weighted life (> spg_reserve_life fallback)
      EV/oz R&R          (12%) — peer-percentile EV per ounce of total R&R (pre-computed)
      Grade Quality      (10%) — high grade → lower unit costs
      Global Rank        (10%) — SNL global production rank (scale + credibility signal)
      Production Growth  (10%) — fwd guidance vs actual production growth %
      FS/PEA IRR         (10%) — best feasibility study post-tax IRR (developers/producers)
      Reserve Life       ( 5%) — spg_reserve_life fallback when mine_life absent

    Returns 50 (neutral) when no SPG/SNL data is available for a ticker.
    """
    # ── Royalty / Streaming: dedicated track ─────────────────────────────────
    if "Royalty" in str(row.get("stage", "")):
        return _royalty_mining_score(row)
    commodity = str(row.get("commodity", ""))
    base_commodity = commodity.split("/")[0]
    components: list[tuple[float, float]] = []   # (score, weight)

    # ── AISC Margin ─────────────────────────────────────────────────────────
    aisc_oz = _safe(row.get("spg_aisc_per_oz"))
    aisc_t  = _safe(row.get("spg_aisc_per_t"))
    aisc_lb = _safe(row.get("spg_aisc_per_lb"))

    if not np.isnan(aisc_oz):
        spot = COMMODITY_SPOT.get(base_commodity, COMMODITY_SPOT.get("Gold", 4821.0))
        margin = (spot - aisc_oz) / spot
        if   margin > 0.60: s = 95
        elif margin > 0.50: s = 85
        elif margin > 0.40: s = 72
        elif margin > 0.30: s = 55
        elif margin > 0.20: s = 38
        elif margin > 0.05: s = 22
        else:               s = 10
        components.append((s, 0.30))

    elif not np.isnan(aisc_lb):
        spot = COMMODITY_SPOT.get("Uranium", 86.65)
        if spot > 0:
            margin = (spot - aisc_lb) / spot
            if   margin > 0.55: s = 95
            elif margin > 0.40: s = 82
            elif margin > 0.25: s = 65
            elif margin > 0.10: s = 45
            elif margin > 0.0:  s = 28
            else:               s = 10
            components.append((s, 0.30))

    elif not np.isnan(aisc_t):
        spot = COMMODITY_SPOT.get(base_commodity, 9200.0)
        if spot > 0:
            margin = (spot - aisc_t) / spot
            if   margin > 0.55: s = 95
            elif margin > 0.40: s = 82
            elif margin > 0.25: s = 65
            elif margin > 0.10: s = 45
            elif margin > 0.0:  s = 28
            else:               s = 10
            components.append((s, 0.30))

    # ── NAV Discount ─────────────────────────────────────────────────────────
    p_nav = _safe(row.get("spg_p_nav"))
    if not np.isnan(p_nav) and p_nav > 0:
        if   p_nav < 0.60:  s = 98
        elif p_nav < 0.75:  s = 88
        elif p_nav < 0.90:  s = 75
        elif p_nav < 1.00:  s = 60
        elif p_nav < 1.25:  s = 42
        elif p_nav < 1.75:  s = 25
        elif p_nav < 2.50:  s = 15
        else:               s = 8
        components.append((s, 0.20))

    # ── Reserves Backing ─────────────────────────────────────────────────────
    resv_m = _safe(row.get("spg_reserves_m"))
    mcap   = _safe(row.get("marketCap"))
    if not np.isnan(resv_m) and not np.isnan(mcap) and mcap > 0:
        backing = resv_m / (mcap / 1e6)
        if   backing > 6.0: s = 98
        elif backing > 4.0: s = 88
        elif backing > 2.5: s = 75
        elif backing > 1.5: s = 58
        elif backing > 1.0: s = 42
        elif backing > 0.5: s = 25
        else:               s = 12
        components.append((s, 0.15))

    # ── EV / oz Production — peer-relative (pre-computed percentile score) ────
    # Pre-computed in compute_scores() as _ev_oz_prod_score (0–100, higher = cheaper)
    ev_oz_s = _safe(row.get("_ev_oz_prod_score"))
    if not np.isnan(ev_oz_s):
        components.append((ev_oz_s, 0.15))

    # ── Mine Life (SNL mine_econ — ownership-weighted, primary signal) ───────
    mine_life = _safe(row.get("spg_mine_life"))
    if not np.isnan(mine_life) and mine_life > 0:
        if   mine_life > 25: s = 97
        elif mine_life > 20: s = 90
        elif mine_life > 15: s = 80
        elif mine_life > 10: s = 65
        elif mine_life >  7: s = 48
        elif mine_life >  5: s = 32
        else:                s = 18
        components.append((s, 0.15))
    else:
        # ── Reserve Life (spg fallback when mine_life unavailable) ────────────
        rli = _safe(row.get("spg_reserve_life"))
        if not np.isnan(rli) and rli > 0:
            if   rli > 20: s = 95
            elif rli > 15: s = 85
            elif rli > 10: s = 70
            elif rli >  7: s = 52
            elif rli >  5: s = 33
            else:          s = 18
            components.append((s, 0.05))   # lower weight: less reliable source

    # ── Grade Quality ─────────────────────────────────────────────────────────
    # Uses primary grade; commodity-specific thresholds
    comm_lower = base_commodity.lower()
    grade_gpt = _safe(row.get("spg_grade_gpt"))
    grade_pct = _safe(row.get("spg_grade_pct"))

    if not np.isnan(grade_gpt) and grade_gpt > 0:
        # g/t — gold and silver
        if comm_lower == "gold":
            if   grade_gpt > 5.0: s = 95
            elif grade_gpt > 3.0: s = 85
            elif grade_gpt > 1.5: s = 68
            elif grade_gpt > 0.8: s = 48
            elif grade_gpt > 0.4: s = 28
            else:                 s = 15
        else:   # silver — g/t scale is much higher
            if   grade_gpt > 300: s = 95
            elif grade_gpt > 150: s = 82
            elif grade_gpt > 80:  s = 65
            elif grade_gpt > 30:  s = 45
            else:                 s = 25
        components.append((s, 0.10))

    elif not np.isnan(grade_pct) and grade_pct > 0:
        # % — copper, uranium, nickel, zinc, lithium, iron ore, coal, rare earths, manganese
        if comm_lower == "copper":
            if   grade_pct > 1.5: s = 95
            elif grade_pct > 1.0: s = 82
            elif grade_pct > 0.5: s = 62
            elif grade_pct > 0.3: s = 40
            else:                 s = 20
        elif comm_lower == "uranium":
            if   grade_pct > 2.0: s = 98   # high-grade uranium (Athabasca)
            elif grade_pct > 0.5: s = 85
            elif grade_pct > 0.1: s = 65
            elif grade_pct > 0.05:s = 40
            else:                 s = 20
        elif comm_lower == "nickel":
            if   grade_pct > 1.5: s = 95
            elif grade_pct > 1.0: s = 80
            elif grade_pct > 0.5: s = 60
            else:                 s = 30
        elif comm_lower == "iron ore":
            # Fe grade — DSO typically 58–65% Fe, low-grade 50–58%
            if   grade_pct >= 65: s = 95   # high-grade DSO
            elif grade_pct >= 62: s = 82
            elif grade_pct >= 58: s = 65
            elif grade_pct >= 55: s = 45
            elif grade_pct >= 50: s = 28
            else:                 s = 15
        elif comm_lower == "coal":
            # Not graded the same way — skip grade signal for coal
            s = 50
        elif comm_lower == "rare earths":
            # TREO% (Total Rare Earth Oxide) — 0.1–5% typical range
            if   grade_pct >= 3.0: s = 95
            elif grade_pct >= 1.5: s = 80
            elif grade_pct >= 0.5: s = 60
            elif grade_pct >= 0.2: s = 40
            else:                  s = 22
        elif comm_lower == "manganese":
            # Mn% — typically 30–48% Mn for ore
            if   grade_pct >= 45: s = 92
            elif grade_pct >= 38: s = 75
            elif grade_pct >= 30: s = 55
            elif grade_pct >= 20: s = 35
            else:                 s = 18
        else:
            # Generic % grade (zinc, lithium, potash)
            s = min(95, max(20, grade_pct * 40))
        components.append((s, 0.10))

    # ── Global Production Rank (SNL 2024Y — scale & credibility signal) ───────
    # Smaller rank = larger global producer = higher operational credibility.
    # Not a pure value signal — weighted low so it amplifies quality, not replaces it.
    global_rank = _safe(row.get("spg_global_rank"))
    if not np.isnan(global_rank) and global_rank > 0:
        if   global_rank <= 10:  s = 90   # top-10 global: Tier-1 major
        elif global_rank <= 25:  s = 78   # top-25: large mid-tier
        elif global_rank <= 50:  s = 65   # top-50: established mid-tier
        elif global_rank <= 100: s = 52   # top-100: notable junior-mid
        elif global_rank <= 200: s = 40   # top-200: small producer
        else:                    s = 28   # ranked but minor
        components.append((s, 0.10))

    # ── P/In-Situ Ratio (SNL) ────────────────────────────────────────────────
    # market_cap / in-situ_R&R_value ($M). Lower = cheaper relative to ounces in ground.
    # 99% of our tickers have SNL in-situ data — this is our highest-coverage metric.
    # Thresholds calibrated against ASX gold producers & developers:
    #   < 0.05 = very cheap (junior with large resource)
    #   0.05–0.15 = cheap-to-fair
    #   0.15–0.30 = fair value
    #   > 0.50 = expensive
    p_insitu = _safe(row.get("snl_p_insitu"))
    if not np.isnan(p_insitu) and p_insitu > 0:
        if   p_insitu < 0.03:  s = 97
        elif p_insitu < 0.06:  s = 90
        elif p_insitu < 0.10:  s = 80
        elif p_insitu < 0.15:  s = 67
        elif p_insitu < 0.25:  s = 52
        elif p_insitu < 0.40:  s = 38
        elif p_insitu < 0.70:  s = 24
        else:                  s = 12
        components.append((s, 0.15))

    # ── EV/oz R&R peer-percentile score (pre-computed in compute_scores) ──────
    # Lower EV/oz R&R relative to peers in the same commodity = cheaper.
    ev_rr_s = _safe(row.get("_ev_oz_rr_score"))
    if not np.isnan(ev_rr_s):
        components.append((ev_rr_s, 0.12))

    # ── Production Growth (forward guidance vs last actual) ───────────────────
    # SNL projections for next year vs most recent actual production.
    # Rising production = growing business = premium signal.
    prod_growth = _safe(row.get("snl_prod_growth_pct"))
    if not np.isnan(prod_growth):
        if   prod_growth > 40:  s = 95
        elif prod_growth > 25:  s = 85
        elif prod_growth > 15:  s = 74
        elif prod_growth > 8:   s = 63
        elif prod_growth > 2:   s = 52
        elif prod_growth > -5:  s = 42
        elif prod_growth > -15: s = 28
        else:                   s = 15
        components.append((s, 0.10))

    # ── FS / PEA Best Post-Tax IRR ────────────────────────────────────────────
    # Most relevant for developers (PEA/PFS/FS exists before production).
    # Also meaningful for producers with expansion studies.
    best_irr = _safe(row.get("snl_best_irr"))
    if not np.isnan(best_irr) and best_irr > 0:
        if   best_irr > 50:  s = 97
        elif best_irr > 35:  s = 88
        elif best_irr > 25:  s = 78
        elif best_irr > 18:  s = 65
        elif best_irr > 12:  s = 50
        elif best_irr > 8:   s = 35
        else:                s = 20
        components.append((s, 0.10))

    # (Analyst consensus upside moved to valuation_score — it is a valuation
    #  signal, not a mining-operations signal, and must not inflate mining scores
    #  for explorers that have no AISC / production / NAV data.)

    if not components:
        return 50.0   # neutral — no SPG data for this ticker

    total_w  = sum(w for _, w in components)
    weighted = sum(s * w for s, w in components)
    return _clip(weighted / total_w)


def commodity_score(commodity: str) -> float:
    """Outlook multiplier converted to 0–100 score."""
    outlook = COMMODITY_OUTLOOK.get(commodity.split("/")[0], 1.0)
    # Map [0.7, 1.3] → [20, 100]
    return _clip((outlook - 0.7) / 0.6 * 80 + 20)


def stage_score(stage: str, row: pd.Series) -> float:
    """
    Stage-adjusted opportunity score.
    Explorers have higher upside if they have cash; producers are safer.
    """
    base = {
        "Major Producer":    40,   # stable but low upside
        "Mid-tier Producer": 55,
        "Producer":          60,
        "Developer":         70,
        "Explorer":          80,   # high upside if data supports
        "Royalty/Streaming": 50,
    }.get(stage, 50)

    # Penalize explorers with low cash (dilution risk)
    if stage == "Explorer":
        cash = _safe(row.get("totalCash"))
        mcap = _safe(row.get("marketCap"))
        if not np.isnan(cash) and not np.isnan(mcap) and mcap > 0:
            if cash / mcap < 0.10:
                base -= 20   # likely needs to raise money soon

    return _clip(float(base))


# ── composite ──────────────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame, meta: dict[str, dict]) -> pd.DataFrame:
    """
    Add score columns to df.
    meta: {ticker: {name, commodity, stage}}
    Returns df with new columns:
      score_valuation, score_health, score_momentum,
      score_commodity, score_stage, score_composite,
      ev_ebitda, p_cf, cash_pct_mcap  (derived metrics)
    """
    # Attach metadata
    df["name"]      = df.index.map(lambda t: meta.get(t, {}).get("name", t))
    df["commodity"] = df.index.map(lambda t: meta.get(t, {}).get("commodity", "Unknown"))
    df["stage"]     = df.index.map(lambda t: meta.get(t, {}).get("stage", "Unknown"))

    # Derived metrics
    df["ev_ebitda"] = (
        df["enterpriseValue"].astype(float) /
        df["ebitda"].astype(float).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).round(2)
    # Only meaningful when EBITDA > 0; negative EBITDA → NaN (explorers/developers)
    df.loc[df["ebitda"].astype(float) <= 0, "ev_ebitda"] = np.nan

    df["p_cf"] = (
        df["marketCap"].astype(float) /
        df["operatingCashflow"].astype(float).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).round(2)
    # Only meaningful when OCF > 0; negative OCF → NaN
    df.loc[df["operatingCashflow"].astype(float) <= 0, "p_cf"] = np.nan

    # EV/Revenue — works for all stages including pre-revenue explorers
    df["ev_revenue"] = (
        df["enterpriseValue"].astype(float) /
        df["totalRevenue"].astype(float).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).round(2)
    # Negative revenue → NaN (rare, but guard it)
    df.loc[df["totalRevenue"].astype(float) < 0, "ev_revenue"] = np.nan

    df["cash_pct_mcap"] = (
        df["totalCash"].astype(float) /
        df["marketCap"].astype(float).replace(0, np.nan) * 100
    ).replace([np.inf, -np.inf], np.nan).round(1)

    df["net_debt_m"] = (
        (df["totalDebt"].astype(float) - df["totalCash"].astype(float)) / 1e6
    ).round(1)

    # ── Derived SPG metrics ────────────────────────────────────────────────────
    # spg_p_nav already provided by _spg_overlay(); expose it as a display column
    if "spg_p_nav" not in df.columns:
        df["spg_p_nav"] = np.nan
    df["spg_p_nav"] = pd.to_numeric(df["spg_p_nav"], errors="coerce")

    # AISC margin % for display (gold + silver in $/oz; copper stays $/t path)
    if "spg_aisc_per_oz" not in df.columns:
        df["spg_aisc_per_oz"] = np.nan
    df["spg_aisc_per_oz"] = pd.to_numeric(df["spg_aisc_per_oz"], errors="coerce")

    def _aisc_margin_row(row: pd.Series) -> float:
        comm = str(row.get("commodity", "")).split("/")[0]
        # Gold / Silver — $/troy oz
        aisc_oz = pd.to_numeric(row.get("spg_aisc_per_oz"), errors="coerce")
        if pd.notna(aisc_oz) and aisc_oz > 0:
            spot = COMMODITY_SPOT.get(comm, COMMODITY_SPOT.get("Gold", 4821.0))
            return round((spot - aisc_oz) / spot * 100, 1) if spot > 0 else float("nan")
        # Copper / Zinc / Nickel / Iron Ore — $/tonne
        aisc_t = pd.to_numeric(row.get("spg_aisc_per_t"), errors="coerce")
        if pd.notna(aisc_t) and aisc_t > 0:
            spot = COMMODITY_SPOT.get(comm, 0.0)
            return round((spot - aisc_t) / spot * 100, 1) if spot > 0 else float("nan")
        # Uranium — $/lb U₃O₈
        aisc_lb = pd.to_numeric(row.get("spg_aisc_per_lb"), errors="coerce")
        if pd.notna(aisc_lb) and aisc_lb > 0:
            spot = COMMODITY_SPOT.get("Uranium", 86.65)
            return round((spot - aisc_lb) / spot * 100, 1) if spot > 0 else float("nan")
        return float("nan")

    df["spg_aisc_margin"] = df.apply(_aisc_margin_row, axis=1)

    # Ensure per-tonne and per-lb columns exist for downstream display
    for _aisc_col in ("spg_aisc_per_t", "spg_aisc_per_lb"):
        if _aisc_col not in df.columns:
            df[_aisc_col] = np.nan
        df[_aisc_col] = pd.to_numeric(df[_aisc_col], errors="coerce")

    # Reserves / market cap multiple (display)
    if "spg_reserves_m" not in df.columns:
        df["spg_reserves_m"] = np.nan
    df["spg_reserves_m"] = pd.to_numeric(df["spg_reserves_m"], errors="coerce")

    # ── SNL local-derived columns ─────────────────────────────────────────────
    for _col in ("spg_mine_life", "spg_global_rank"):
        if _col not in df.columns:
            df[_col] = np.nan
        df[_col] = pd.to_numeric(df[_col], errors="coerce")

    # ── Analyst upside % — computed from consensus target vs current price ───
    # Handles both fresh-fetch (camelCase) and DB-reload (snake_case) column names.
    _mean_tgt = pd.to_numeric(
        df.get("targetMeanPrice", df.get("analyst_target_mean",
               pd.Series(np.nan, index=df.index))),
        errors="coerce",
    )
    _px = pd.to_numeric(
        df.get("regularMarketPrice", df.get("price",
               pd.Series(np.nan, index=df.index))),
        errors="coerce",
    ).replace(0, np.nan)
    df["analyst_upside"] = ((_mean_tgt / _px) - 1.0).mul(100).round(1)

    # ── EV/oz Production peer-percentile score ─────────────────────────────────
    # Lower EV/oz = cheaper relative to peers in the same commodity group.
    # Pre-compute as a 0–100 score (higher = cheaper/more attractive) so
    # mining_score() can consume it without needing the full DataFrame.
    for col in ("spg_production_oz", "spg_production_lb", "spg_production_t",
                "spg_contained_reserves_oz", "spg_contained_reserves_lb",
                "spg_reserve_life", "spg_grade_gpt", "spg_grade_pct"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute EV/oz ratio inline (raw, before any app.py transforms)
    _ev_raw = pd.to_numeric(df.get("enterpriseValue", pd.Series(np.nan, index=df.index)), errors="coerce")
    _prod_oz = df["spg_production_oz"]
    _ev_oz_ratio = (_ev_raw / _prod_oz.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Percentile rank within same-commodity group (invert: lower EV/oz = higher score)
    def _peer_pct_score(ratio_series: pd.Series, commodity_series: pd.Series,
                        invert: bool = True) -> pd.Series:
        """Return 0–100 peer-percentile score per commodity group."""
        scores = pd.Series(np.nan, index=ratio_series.index)
        for _comm in commodity_series.dropna().unique():
            _mask = commodity_series == _comm
            _vals = ratio_series[_mask].dropna()
            if len(_vals) < 2:
                continue
            for idx in _vals.index:
                v = _vals[idx]
                pct = (_vals < v).mean()
                scores[idx] = (1 - pct) * 100 if invert else pct * 100
        return scores

    df["_ev_oz_prod_score"] = _peer_pct_score(_ev_oz_ratio, df["commodity"])

    # ── EV / oz Total R&R score (pre-computed) ────────────────────────────────
    # snl_ev_per_oz_rr is computed in app._apply_sqlite_snl; may be NaN for tickers
    # without SNL R&R data.
    for _snl_col in ("snl_ev_per_oz_rr", "snl_p_insitu",
                     "snl_prod_growth_pct", "snl_best_irr"):
        if _snl_col not in df.columns:
            df[_snl_col] = np.nan
        df[_snl_col] = pd.to_numeric(df[_snl_col], errors="coerce")

    _ev_rr = df.get("snl_ev_per_oz_rr", pd.Series(np.nan, index=df.index))
    df["_ev_oz_rr_score"] = _peer_pct_score(
        pd.to_numeric(_ev_rr, errors="coerce"), df["commodity"])

    # ── Royalty P/CF peer-percentile (within royalty group only) ─────────────
    # Royalties trade at 20–40× P/CF; ranking them against miners is meaningless.
    # We pre-compute a royalty-peer-relative P/CF score so _royalty_mining_score()
    # can use it without needing the full DataFrame.
    _royalty_mask = df["stage"].str.contains("Royalty", na=False)
    _mc_s  = pd.to_numeric(
        df.get("marketCap", df.get("market_cap", pd.Series(np.nan, index=df.index))),
        errors="coerce")
    _ocf_s = pd.to_numeric(
        df.get("operatingCashflow", df.get("operating_cf",
               pd.Series(np.nan, index=df.index))), errors="coerce").replace(0, np.nan)
    _pcf_s = (_mc_s / _ocf_s).replace([np.inf, -np.inf], np.nan)
    # Among royalty companies, lower P/CF = cheaper (same direction as miners)
    df["_royalty_pcf_score"] = np.nan
    _r_pcf = _pcf_s[_royalty_mask].dropna()
    if len(_r_pcf) >= 2:
        for idx in _r_pcf.index:
            v = _r_pcf[idx]
            pct = (_r_pcf < v).mean()
            df.at[idx, "_royalty_pcf_score"] = (1 - pct) * 100  # lower P/CF = higher score

    # Compute scores row by row
    v_scores, h_scores, m_scores, n_scores, c_scores, s_scores = [], [], [], [], [], []

    for ticker, row in df.iterrows():
        peers = df
        v_scores.append(valuation_score(row, peers))
        h_scores.append(health_score(row))
        m_scores.append(momentum_score(row))
        n_scores.append(mining_score(row))
        c_scores.append(commodity_score(str(row.get("commodity", ""))))
        s_scores.append(stage_score(str(row.get("stage", "")), row))

    df["score_valuation"] = [round(s, 1) for s in v_scores]
    df["score_health"]    = [round(s, 1) for s in h_scores]
    df["score_momentum"]  = [round(s, 1) for s in m_scores]
    df["score_mining"]    = [round(s, 1) for s in n_scores]
    df["score_commodity"] = [round(s, 1) for s in c_scores]
    df["score_stage"]     = [round(s, 1) for s in s_scores]

    w = SCORE_WEIGHTS
    df["score_composite"] = (
        df["score_valuation"] * w["valuation"] +
        df["score_health"]    * w["health"]    +
        df["score_momentum"]  * w["momentum"]  +
        df["score_mining"]    * w["mining"]    +
        df["score_commodity"] * w["commodity"] +
        df["score_stage"]     * w["stage"]
    ).clip(0, 100).round(1)

    # ── Market cap floor: micro-caps (<$10M) are too illiquid to rank meaningfully ──
    # Use camelCase (fresh fetch) or snake_case (DB reload) — whichever is available.
    _mc = df.get("marketCap", df.get("market_cap"))
    if _mc is not None:
        _mc = pd.to_numeric(_mc, errors="coerce")
        micro_cap_mask = _mc.notna() & (_mc < 10_000_000)
        df.loc[micro_cap_mask, "score_composite"] = 50.0

    # Grade label
    def grade(s):
        if s >= 75: return "🟢 Strong Buy"
        if s >= 60: return "🔵 Buy"
        if s >= 45: return "🟡 Watch"
        if s >= 30: return "🟠 Neutral"
        return             "🔴 Avoid"

    df["grade"] = df["score_composite"].apply(grade)

    # ── Peer group ranking ────────────────────────────────────────────────────
    # Companies are only truly comparable within the same stage × commodity bucket.
    # We compute peer_group, peer_rank (1=best), peer_n (group size), peer_pct
    # so the UI can show "2nd out of 9 gold producers" alongside the global composite.
    #
    # Stage buckets: Producer (all producer variants), Developer, Explorer, Royalty.
    # Commodity: primary commodity (before first "/"). Royalties ignore commodity.
    # Groups with <3 members fall back to stage-only bucket.

    def _stage_bucket(stage: str) -> str:
        s = str(stage)
        if "Royalty" in s:   return "Royalty"
        if "Producer" in s:  return "Producer"
        if "Developer" in s: return "Developer"
        if "Explorer" in s:  return "Explorer"
        return "Other"

    df["_comm_primary"] = df["commodity"].str.split("/").str[0].str.strip()
    df["_stage_bucket"] = df["stage"].apply(_stage_bucket)

    # Attempt fine group (commodity × stage), fall back to stage-only when <3 members
    df["_peer_fine"] = df["_comm_primary"] + " · " + df["_stage_bucket"]
    _fine_sizes = df["_peer_fine"].map(df["_peer_fine"].value_counts())
    df["peer_group"] = np.where(_fine_sizes >= 3,
                                df["_peer_fine"],
                                df["_stage_bucket"])
    # Royalties always stay in the royalty bucket regardless
    df.loc[df["_stage_bucket"] == "Royalty", "peer_group"] = "Royalty"

    df["peer_n"] = df.groupby("peer_group")["score_composite"] \
                     .transform("count").astype(int)
    df["peer_rank"] = df.groupby("peer_group")["score_composite"] \
                        .rank(ascending=False, method="min").astype(int)
    df["peer_pct"]  = df.groupby("peer_group")["score_composite"] \
                        .rank(ascending=True, pct=True).mul(100).round(0).astype(int)

    # Clean up internal columns
    df.drop(columns=["_comm_primary", "_stage_bucket", "_peer_fine"],
            errors="ignore", inplace=True)

    return df
