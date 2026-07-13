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


def run_daily_refresh():
    """Full pipeline: fetch → score → persist."""
    log.info("-" * 60)
    log.info("Daily refresh started")
    try:
        init_db()
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
