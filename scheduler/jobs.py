"""Daily refresh job — fetches data, scores stocks, persists to DB."""
import logging
from datetime import date

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from config import REFRESH_HOUR, REFRESH_MINUTE, TIMEZONE, FX_TO_USD
from data.universe import get_tickers, get_ticker_meta, UNIVERSE_SRC
from data.fetcher import fetch_all, fetch_spot_prices, apply_snl_for_scoring
from data.scorer import compute_scores
from data.database import init_db, upsert_snapshot, upsert_commodity_prices

log = logging.getLogger(__name__)

# Stale-price (trading halt / no-trade proxy) penalty
STALE_PRICE_LOOKBACK = 5                    # distinct snapshot dates with frozen price
STALE_HARD_LOOKBACK  = 10                   # frozen this long = stale, volume or not
STALE_TURNOVER_FLOOR = 10_000.0             # avg daily traded value (local ccy)
STALE_SCORE_CAP      = 40.0
STALE_GRADE          = "⏸️ Stale/Halted"


def _frozen_tickers(conn, lookback: int) -> set[str]:
    """Tickers whose price is identical across the last `lookback` snapshots."""
    from sqlalchemy import text

    # (ticker, snap_date) is UNIQUE, so row_number over snap_date DESC walks
    # distinct dates. NULL prices break the streak via COUNT(price). Cold
    # start: fewer than `lookback` snapshots never qualifies (COUNT(*) check).
    rows = conn.execute(text("""
        SELECT ticker FROM (
            SELECT ticker, price,
                   ROW_NUMBER() OVER (PARTITION BY ticker
                                      ORDER BY snap_date DESC) AS rn
            FROM stock_snapshots
        )
        WHERE rn <= :lookback
        GROUP BY ticker
        HAVING COUNT(*) = :lookback
           AND COUNT(price) = :lookback
           AND MIN(price) = MAX(price)
    """), {"lookback": lookback}).fetchall()
    return {r[0] for r in rows}


def apply_stale_price_penalty(snap_date) -> list[str]:
    """Cap scores for tickers that are frozen AND show no trading.

    A frozen price alone is not a halt: low-priced names quantised to coarse
    ticks close flat for a week while trading normally — the AU station's
    price-only rule flagged 62/420 names including an A$162M producer that was
    simply flat (#87). A real halt starves the 10-day average volume toward
    zero within days. So:

      frozen >= STALE_PRICE_LOOKBACK  AND  avg_turnover missing or < floor  -> stale
      frozen >= STALE_HARD_LOOKBACK                                         -> stale
                (two weeks flat is damning even if the volume field lies —
                 Yahoo's averageVolume fallback is a 3-month figure that stays
                 fat for weeks into a suspension)

    Note the floor is in LOCAL currency: 10k IDR ≈ nothing, 10k GBP is real
    money. It is deliberately loose — its job is telling "trading, just flat"
    from "no trade at all", not measuring liquidity; the hard rule backstops it.

    Runs as DB post-processing — like the micro-cap floor — because it needs
    multi-day history that compute_scores(), which only ever sees a single
    day's DataFrame, cannot access. Returns the penalised tickers.
    """
    from sqlalchemy import bindparam, text
    from data.database import _engine

    with _engine().begin() as conn:
        frozen = _frozen_tickers(conn, STALE_PRICE_LOOKBACK)
        if not frozen:
            return []
        frozen_long = _frozen_tickers(conn, STALE_HARD_LOOKBACK)

        turnover = dict(conn.execute(
            text("""
                SELECT ticker, avg_turnover FROM stock_snapshots
                WHERE snap_date = :d AND ticker IN :tickers
            """).bindparams(bindparam("tickers", expanding=True)),
            {"d": str(snap_date), "tickers": sorted(frozen)},
        ).fetchall())

        stale_tickers = sorted(
            t for t in frozen
            if t in frozen_long
            or turnover.get(t) is None
            or float(turnover[t]) < STALE_TURNOVER_FLOOR
        )
        if not stale_tickers:
            return []

        conn.execute(
            text("""
                UPDATE stock_snapshots
                SET score_composite = MIN(score_composite, :cap),
                    grade = :grade
                WHERE snap_date = :d
                  AND ticker IN :tickers
            """).bindparams(bindparam("tickers", expanding=True)),
            {"cap": STALE_SCORE_CAP, "grade": STALE_GRADE,
             "d": str(snap_date), "tickers": stale_tickers},
        )
    return stale_tickers


# Price scale-break quarantine (#104). Distinct from the stale penalty: that
# one catches a price that stops MOVING, this one a price that jumps MAGNITUDE
# (rand/cent flips ~100x, un-adjusted consolidations ~500x). The 12-month scan
# that motivated this found 16 live breaks, mostly in these global stations:
# BSAI x522, CUAI x198, CPR.JO, WEZ.JO, NEO ...
SCALE_BREAK_RATIO   = 5.0        # adjacent-snapshot ratio beyond this = unit change
SCALE_BREAK_GRADE   = "⚠️ Scale-break"
SCALE_BREAK_CAP     = 40.0
SCALE_BREAK_RESCAN_DAYS = 365


def apply_scale_break_quarantine(rescan_days: int = SCALE_BREAK_RESCAN_DAYS) -> list[tuple]:
    """Mark snapshot rows whose price jumped a magnitude vs the previous snapshot.

    Quarantine, not repair: the row's grade becomes ⚠️ Scale-break and its
    composite is capped so ranked consumers (radar, nominations, race, sweep)
    drop it. Prices are left as recorded — rewriting history would corrupt
    race attribution, and "we recorded what Yahoo said, and it was broken"
    is itself the honest record. Idempotent: re-marking an already-marked row
    is a no-op, so the daily trailing-12-month sweep IS the retro-fix.

    Returns [(ticker, snap_date, ratio), ...] quarantined in this run.
    """
    from sqlalchemy import text
    from data.database import _engine

    with _engine().begin() as conn:
        rows = conn.execute(text("""
            SELECT ticker, snap_date, price,
                   LAG(price) OVER (PARTITION BY ticker ORDER BY snap_date) AS prev
            FROM stock_snapshots
            WHERE snap_date >= date('now', :window)
        """), {"window": f"-{rescan_days} day"}).fetchall()

        breaks = []
        for ticker, day, price, prev in rows:
            if not price or not prev:
                continue
            ratio = float(price) / float(prev)
            if ratio > SCALE_BREAK_RATIO or ratio < 1 / SCALE_BREAK_RATIO:
                breaks.append((ticker, day, round(ratio, 2)))

        for ticker, day, _ in breaks:
            conn.execute(text("""
                UPDATE stock_snapshots
                SET score_composite = MIN(score_composite, :cap),
                    grade = :grade
                WHERE ticker = :t AND snap_date = :d
            """), {"cap": SCALE_BREAK_CAP, "grade": SCALE_BREAK_GRADE,
                   "t": ticker, "d": day})
    return breaks


def refine_stages(raw_df: pd.DataFrame, meta: dict[str, dict]) -> None:
    """
    Upgrade the market-cap-heuristic stage of screen-sourced tickers using
    fetched fundamentals: real revenue → producer tiers; none → developer or
    explorer by market cap. Curated tickers keep their hand-assigned stage.
    """
    n = 0
    for tk, m in meta.items():
        if UNIVERSE_SRC.get(tk) != "screen" or tk not in raw_df.index:
            continue
        row = raw_df.loc[tk]
        rev  = row.get("totalRevenue")
        mcap = row.get("marketCap")
        rev_usd  = float(rev)  * FX_TO_USD if pd.notna(rev)  else 0.0
        mcap_usd = float(mcap) * FX_TO_USD if pd.notna(mcap) else 0.0
        if rev_usd >= 10e6:
            stage = ("Major Producer" if mcap_usd >= 5e9 else
                     "Mid-tier Producer" if mcap_usd >= 5e8 else "Producer")
        else:
            stage = "Developer" if mcap_usd >= 1e8 else "Explorer"
        if stage != m["stage"]:
            m["stage"] = stage
            n += 1
    if n:
        log.info(f"  Stage refinement: {n} screen-sourced tickers updated.")


def apply_ai_outlook() -> None:
    """
    Apply the AI-generated commodity outlook (ai_outlook.json, written daily
    by scripts/ai_overlay.py) on top of the static config multipliers.
    Values are clamped to [0.7, 1.3]; missing/stale file → static config wins.
    """
    import json
    from pathlib import Path
    import config as _cfg

    path = Path(__file__).resolve().parent.parent / "ai_outlook.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        n = 0
        for comm, item in payload.get("outlook", {}).items():
            if comm in _cfg.COMMODITY_OUTLOOK:
                mult = float(item["multiplier"])
                _cfg.COMMODITY_OUTLOOK[comm] = min(1.3, max(0.7, mult))
                n += 1
        if n:
            log.info(f"  AI commodity outlook applied ({n} commodities, "
                     f"generated {payload.get('generated_at', '?')})")
    except Exception as e:                                    # noqa: BLE001
        log.warning(f"  AI outlook skipped: {e}")


def run_daily_refresh():
    """Full pipeline: fetch → score → persist."""
    log.info("-" * 60)
    log.info("Daily refresh started")
    try:
        init_db()
        apply_ai_outlook()
        tickers = get_tickers()
        meta    = get_ticker_meta()

        # Fetch and store commodity spot prices
        spot_prices = fetch_spot_prices()
        if spot_prices:
            upsert_commodity_prices(spot_prices, date.today())

        raw_df  = fetch_all(tickers)
        if raw_df.empty:
            log.warning("No data fetched — skipping save.")
            return

        # Enrich with local SNL metrics so mining_score() can use them
        log.info("  Applying SNL local enrichment for scoring ...")
        raw_df = apply_snl_for_scoring(raw_df)

        # Refine heuristic stages using fetched revenue/market cap
        refine_stages(raw_df, meta)

        scored  = compute_scores(raw_df, meta)
        upsert_snapshot(scored, date.today())

        # Safety net: cap any micro-cap (<$10M) composite scores that slipped through
        # (e.g. Yahoo returned NaN for market_cap during bulk fetch due to rate limits)
        from data.database import _engine
        from sqlalchemy import text as _text
        with _engine().begin() as _conn:
            _r = _conn.execute(_text("""
                UPDATE stock_snapshots
                SET score_composite = 50.0
                WHERE snap_date = :d
                  AND market_cap IS NOT NULL
                  AND market_cap < 10000000
                  AND score_composite > 50.0
            """), {"d": str(date.today())})
            if _r.rowcount:
                log.info(f"  Micro-cap floor applied to {_r.rowcount} rows.")

        # Stale-price penalty: price frozen across last N snapshots = halt proxy
        _stale = apply_stale_price_penalty(date.today())
        if _stale:
            log.info(f"  Stale-price penalty applied to {len(_stale)} "
                     f"tickers: {', '.join(_stale)}")

        # Scale-break quarantine (#104): price jumped a magnitude vs previous
        # snapshot = unit change, not a return. Trailing sweep is idempotent.
        _breaks = apply_scale_break_quarantine()
        if _breaks:
            log.info(f"  Scale-break quarantine: {len(_breaks)} rows — "
                     + ", ".join(f"{t} {d} x{r}" for t, d, r in _breaks[:8]))

        log.info(f"Daily refresh complete. {len(scored)} stocks saved.")
    except Exception as e:
        log.exception(f"Daily refresh failed: {e}")
    log.info("-" * 60)


def start_scheduler() -> BackgroundScheduler:
    """Start APScheduler background scheduler and return it."""
    tz  = pytz.timezone(TIMEZONE)
    sched = BackgroundScheduler(timezone=tz)
    sched.add_job(
        run_daily_refresh,
        trigger=CronTrigger(hour=REFRESH_HOUR, minute=REFRESH_MINUTE, timezone=tz),
        id="daily_refresh",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    sched.start()
    log.info(
        f"Scheduler started — daily refresh at "
        f"{REFRESH_HOUR:02d}:{REFRESH_MINUTE:02d} {TIMEZONE}"
    )
    return sched
