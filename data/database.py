"""SQLite persistence layer — stores daily snapshots and exposes query helpers."""
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    create_engine, text,
    Column, String, Float, Date, DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Session

from config import DB_PATH

log = logging.getLogger(__name__)

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    return _ENGINE


# ── ORM model ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class StockSnapshot(Base):
    __tablename__ = "stock_snapshots"
    __table_args__ = (UniqueConstraint("ticker", "snap_date"),)

    id              = Column(String, primary_key=True)   # ticker|date
    ticker          = Column(String, nullable=False)
    snap_date       = Column(Date,   nullable=False)
    fetched_at      = Column(DateTime, default=datetime.utcnow)

    name            = Column(String)
    commodity       = Column(String)
    stage           = Column(String)
    currency        = Column(String)
    exchange        = Column(String)

    market_cap      = Column(Float)
    enterprise_value= Column(Float)
    price           = Column(Float)
    price_to_book   = Column(Float)
    ev_ebitda       = Column(Float)
    ev_revenue      = Column(Float)
    p_cf            = Column(Float)
    debt_to_equity  = Column(Float)
    current_ratio   = Column(Float)
    cash_pct_mcap   = Column(Float)
    net_debt_m      = Column(Float)
    operating_cf    = Column(Float)
    free_cf         = Column(Float)
    revenue         = Column(Float)

    rsi             = Column(Float)
    wk52_position   = Column(Float)
    pct_from_52hi   = Column(Float)
    pct_from_52lo   = Column(Float)
    price_vs_ma200  = Column(Float)

    score_valuation = Column(Float)
    score_health    = Column(Float)
    score_momentum  = Column(Float)
    score_commodity = Column(Float)
    score_stage     = Column(Float)
    score_composite = Column(Float)
    grade           = Column(String)

    # Bloomberg fields (nullable)
    bb_aisc         = Column(Float)
    bb_production   = Column(Float)
    bb_nav_per_shr  = Column(Float)
    bb_ev_to_nav    = Column(Float)
    bb_mine_life    = Column(Float)

    # S&P Capital IQ / SNL Mining fields (nullable)
    spg_p_nav       = Column(Float)   # Price / NAV per share
    spg_reserves_m  = Column(Float)   # Reserves in-situ value ($M)
    spg_resources_m = Column(Float)   # Resources in-situ value ($M)
    spg_aisc_per_oz    = Column(Float)   # AISC in $/oz  (Gold / Silver)
    spg_aisc_per_t     = Column(Float)   # AISC in $/t   (Copper / Zinc / Nickel)
    spg_aisc_per_lb    = Column(Float)   # AISC in $/lb  (Uranium)
    spg_aisc_margin    = Column(Float)   # AISC margin % vs relevant spot price
    spg_production_cost_t = Column(Float)  # Production cost $/t (SNL_PRODUCTION_COST_TONNE)
    spg_grade_gpt   = Column(Float)   # Primary grade g/t  (Au/Ag — SNL_PRIMARY_GRD_R_AND_R_G_PER_TONNE)
    spg_grade_pct   = Column(Float)   # Primary grade %    (Cu/Ni/Zn/U3O8/Li/Fe/K2O — SNL_PRIMARY_GRD_R_AND_R_PCT)
    score_mining    = Column(Float)   # Mining sub-score (0–100)

    # Cash cost (direct operating cost, below AISC)
    spg_cash_cost_oz  = Column(Float)   # $/oz — Gold / Silver
    spg_cash_cost_t   = Column(Float)   # $/t  — base metals
    spg_cash_cost_lb  = Column(Float)   # $/lb — Uranium
    # Production volume (attributable, most recent completed annual)
    spg_production_oz = Column(Float)  # oz  — Gold / Silver
    spg_production_t  = Column(Float)  # t   — Copper / Zinc / Nickel / Iron Ore
    spg_production_lb = Column(Float)  # lb  — Uranium
    # Average realized price vs spot
    spg_realized_price_oz = Column(Float)  # $/oz — Gold / Silver
    spg_realized_price_t  = Column(Float)  # $/t  — base metals
    spg_realized_price_lb = Column(Float)  # $/lb — Uranium
    # Primary contained metal in Proven+Probable reserves (for EV/oz Reserve)
    spg_contained_reserves_oz = Column(Float)  # oz — Gold / Silver
    spg_contained_reserves_lb = Column(Float)  # lb — Uranium / Copper
    # Reserve life index (computed: contained_reserves / annual_production, years)
    spg_reserve_life = Column(Float)

    # SNL local-table derived fields
    spg_mine_life   = Column(Float)   # Ownership-weighted mine life (years, from snl_mine_econ)
    spg_global_rank = Column(Float)   # Global production rank for primary commodity (SNL, 2024Y)
    analyst_upside  = Column(Float)   # Analyst consensus upside % vs current price

    # Peer-group ranking (computed in scorer.compute_scores)
    peer_group      = Column(String)  # e.g. "Gold · Producer" or "Royalty"
    peer_rank       = Column(Float)   # Rank within peer group (1 = best composite score)
    peer_n          = Column(Float)   # Number of companies in the peer group
    peer_pct        = Column(Float)   # Percentile within peer group (0–100, higher = better)

    dividend_yield  = Column(Float)   # Trailing dividend yield % (Yahoo Finance)
    return_1m       = Column(Float)   # 1-month price return %  (≈ 21 trading days)
    return_3m       = Column(Float)   # 3-month price return %  (≈ 63 trading days)
    avg_turnover    = Column(Float)   # ~20-day avg daily traded value (local ccy; avg volume × price)

    # Analyst consensus (Yahoo Finance)
    analyst_target_mean   = Column(Float)   # Consensus mean price target
    analyst_target_high   = Column(Float)   # High price target
    analyst_target_low    = Column(Float)   # Low price target
    analyst_count         = Column(Float)   # Number of analyst opinions
    analyst_rec_key       = Column(String)  # e.g. "buy", "hold", "sell"
    analyst_rec_mean      = Column(Float)   # 1=Strong Buy … 5=Strong Sell


def init_db():
    Base.metadata.create_all(_engine())
    # Safe migrations: add columns that may not exist in older DB files
    _migrations = [
        "ALTER TABLE stock_snapshots ADD COLUMN ev_revenue FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_p_nav FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_reserves_m FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_resources_m FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_aisc_per_oz FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_aisc_per_t FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_aisc_per_lb FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_aisc_margin FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN score_mining FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN dividend_yield FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN return_1m FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN return_3m FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_production_cost_t FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN analyst_target_mean FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN analyst_target_high FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN analyst_target_low FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN analyst_count FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN analyst_rec_key TEXT",
        "ALTER TABLE stock_snapshots ADD COLUMN analyst_rec_mean FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_grade_gpt FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_grade_pct FLOAT",
        # Production volume + realized price + contained reserves + RLI
        "ALTER TABLE stock_snapshots ADD COLUMN spg_cash_cost_oz FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_cash_cost_t FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_cash_cost_lb FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_production_oz FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_production_t FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_production_lb FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_realized_price_oz FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_realized_price_t FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_realized_price_lb FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_contained_reserves_oz FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_contained_reserves_lb FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_reserve_life FLOAT",
        # Profitability metrics (producer quality)
        "ALTER TABLE stock_snapshots ADD COLUMN return_on_equity FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN operating_margins FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN gross_margins FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN profit_margins FLOAT",
        # SNL local-table derived fields
        "ALTER TABLE stock_snapshots ADD COLUMN spg_mine_life FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN spg_global_rank FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN analyst_upside FLOAT",
        # Peer-group ranking
        "ALTER TABLE stock_snapshots ADD COLUMN peer_group TEXT",
        "ALTER TABLE stock_snapshots ADD COLUMN peer_rank FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN peer_n FLOAT",
        "ALTER TABLE stock_snapshots ADD COLUMN peer_pct FLOAT",
        # Liquidity: avg daily traded value (read by AU repo's risk _adv_dollars)
        "ALTER TABLE stock_snapshots ADD COLUMN avg_turnover FLOAT",
    ]
    with _engine().begin() as conn:
        for sql in _migrations:
            try:
                conn.execute(text(sql))
                col = sql.split("ADD COLUMN ")[1].split(" ")[0]
                log.info(f"Migrated: added {col}")
            except Exception:
                pass  # column already exists

        # Watchlist table — simple set of saved tickers with optional note
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker    TEXT PRIMARY KEY,
                added_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                note      TEXT
            )
        """))

        # User settings — generic key/value store (score weights, UI prefs, etc.)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """))

        # Commodity spot prices — one row per (date, commodity)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS commodity_prices (
                price_date  DATE    NOT NULL,
                commodity   TEXT    NOT NULL,
                price       FLOAT   NOT NULL,
                PRIMARY KEY (price_date, commodity)
            )
        """))

        # Portfolio positions — shares held + average cost per ticker
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS positions (
                ticker    TEXT PRIMARY KEY,
                shares    FLOAT   NOT NULL DEFAULT 0,
                avg_cost  FLOAT,          -- average cost per share (AUD as entered)
                currency  TEXT DEFAULT 'AUD',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Safe migration: add price_target to watchlist (older DBs won't have it)
        try:
            conn.execute(text("ALTER TABLE watchlist ADD COLUMN price_target FLOAT"))
        except Exception:
            pass  # column already exists

        # Transaction log — each buy/sell event per ticker
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                trans_date  DATE    NOT NULL,
                shares      FLOAT   NOT NULL,
                price       FLOAT   NOT NULL,
                trans_type  TEXT    NOT NULL DEFAULT 'buy',  -- 'buy' or 'sell'
                note        TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # ── SNL / Snowflake local cache tables (populated by snl_sync.py) ─────
        conn.execute(text("CREATE TABLE IF NOT EXISTS snl_sync_log (table_name TEXT PRIMARY KEY, last_sync_at TEXT, row_count INTEGER, status TEXT, message TEXT)"))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_company_production (
            snl_key TEXT, period TEXT, commodity TEXT,
            prod_oz REAL, prod_t REAL, prod_lb REAL,
            cash_cost_oz REAL, cash_cost_t REAL, cash_cost_lb REAL,
            aic_oz REAL, aisc_oz REAL, aisc_t REAL, aisc_lb REAL,
            realized_price_oz REAL, realized_price_t REAL, realized_price_lb REAL,
            revenue_m REAL, synced_at TEXT,
            PRIMARY KEY (snl_key, period, commodity))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_company_rr (
            snl_key TEXT, period TEXT, commodity TEXT,
            grade_gpt REAL, grade_pct REAL,
            contained_rr_oz REAL, contained_rr_lb REAL, contained_rr_t REAL,
            contained_reserves_oz REAL, contained_reserves_lb REAL, contained_reserves_t REAL,
            contained_total_resources_oz REAL, contained_mi_oz REAL, contained_inferred_oz REAL,
            ore_tonnes_rr REAL, ore_tonnes_reserves REAL, ore_tonnes_resources REAL,
            synced_at TEXT, PRIMARY KEY (snl_key, period, commodity))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_company_insitu (
            snl_key TEXT, period TEXT,
            insitu_rr_m REAL, insitu_reserves_m REAL, insitu_resources_m REAL,
            insitu_mi_m REAL, insitu_inferred_m REAL,
            synced_at TEXT, PRIMARY KEY (snl_key, period))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_company_ranking (
            snl_key TEXT, period TEXT, commodity TEXT, ownership_method TEXT,
            global_rank INTEGER, prod_oz REAL, prod_t REAL, prod_lb REAL,
            world_share_pct REAL, synced_at TEXT,
            PRIMARY KEY (snl_key, period, commodity, ownership_method))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_company_ranking_all (
            snl_key TEXT, period TEXT, ownership_method TEXT,
            global_rank_all INTEGER, global_prod_value_all_m REAL, world_share_all_pct REAL,
            synced_at TEXT, PRIMARY KEY (snl_key, period, ownership_method))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_company_projections (
            snl_key TEXT, estimate_period TEXT, description TEXT,
            prod_high_oz REAL, prod_low_oz REAL, prod_high_t REAL, prod_low_t REAL,
            prod_high_lb REAL, prod_low_lb REAL,
            aisc_high_oz REAL, aisc_low_oz REAL, aisc_high_t REAL, aisc_low_t REAL,
            cash_cost_high_oz REAL, cash_cost_low_oz REAL,
            synced_at TEXT, PRIMARY KEY (snl_key, estimate_period))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_general (
            property_id TEXT PRIMARY KEY, property_name TEXT, stage TEXT, status TEXT,
            primary_commodity TEXT, commodities TEXT, country TEXT, state_province TEXT,
            latitude REAL, longitude REAL, synced_at TEXT)"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_owner (
            property_id TEXT, property_name TEXT, snl_key TEXT,
            pct_own REAL, mkt_cap_m REAL, tev_m REAL, ticker TEXT, exchange TEXT,
            synced_at TEXT, PRIMARY KEY (property_id, snl_key))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_grade_oz (
            property_id TEXT, property_name TEXT, period TEXT, commodity TEXT,
            grade_reserves REAL, contained_reserves REAL,
            grade_mi_excl REAL, contained_mi_excl REAL,
            grade_inferred REAL, contained_inferred REAL,
            grade_rr REAL, contained_rr REAL,
            grade_mi_incl REAL, contained_mi_incl REAL,
            insitu_reserves_m REAL, insitu_mi_excl_m REAL,
            insitu_inferred_m REAL, insitu_rr_m REAL,
            synced_at TEXT, PRIMARY KEY (property_id, period, commodity))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_grade_lb (
            property_id TEXT, property_name TEXT, period TEXT, commodity TEXT,
            grade_reserves REAL, contained_reserves REAL,
            grade_mi_excl REAL, contained_mi_excl REAL,
            grade_inferred REAL, contained_inferred REAL,
            grade_rr REAL, contained_rr REAL,
            grade_mi_incl REAL, contained_mi_incl REAL,
            insitu_reserves_m REAL, insitu_mi_excl_m REAL,
            insitu_inferred_m REAL, insitu_rr_m REAL,
            synced_at TEXT, PRIMARY KEY (property_id, period, commodity))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_grade_t (
            property_id TEXT, property_name TEXT, period TEXT, commodity TEXT,
            grade_reserves REAL, contained_reserves REAL,
            grade_mi_excl REAL, contained_mi_excl REAL,
            grade_inferred REAL, contained_inferred REAL,
            grade_rr REAL, contained_rr REAL,
            grade_mi_incl REAL, contained_mi_incl REAL,
            insitu_reserves_m REAL, insitu_mi_excl_m REAL,
            insitu_inferred_m REAL, insitu_rr_m REAL,
            synced_at TEXT, PRIMARY KEY (property_id, period, commodity))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_studies (
            property_id TEXT, property_name TEXT, study_rank INTEGER,
            study_type TEXT, study_date TEXT, study_year INTEGER,
            mine_life_yrs REAL, initial_capex_m REAL, lom_sustaining_m REAL,
            pretax_npv_m REAL, posttax_npv_m REAL, npv_discount_pct REAL,
            pretax_irr_pct REAL, posttax_irr_pct REAL, payback_yrs REAL,
            currency TEXT, synced_at TEXT, PRIMARY KEY (property_id, study_rank))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_production (
            property_id TEXT, property_name TEXT, year INTEGER, primary_commodity TEXT,
            head_grade_gpt REAL, head_grade_pct REAL, recovery_rate REAL,
            prod_oz REAL, prod_t REAL, prod_lb REAL,
            aisc_oz REAL, aisc_t REAL, aisc_lb REAL,
            cash_cost_oz REAL, cash_cost_t REAL, cash_cost_lb REAL,
            synced_at TEXT, PRIMARY KEY (property_id, year, primary_commodity))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_property_capacity (
            property_id TEXT PRIMARY KEY, property_name TEXT,
            mill_capacity_tpd REAL, mill_capacity_tpy REAL, stripping_ratio REAL,
            mining_methods TEXT, processing_methods TEXT,
            actual_startup_year INTEGER, projected_startup_year INTEGER,
            actual_closure_year INTEGER, synced_at TEXT)"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_exploration_budget (
            snl_key TEXT, fiscal_year INTEGER,
            total_budget_m REAL, actual_spent_m REAL, commodity TEXT, company_class TEXT,
            synced_at TEXT, PRIMARY KEY (snl_key, fiscal_year))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_mine_econ_precious (
            property_id TEXT, property_name TEXT, period TEXT, commodity TEXT, basis TEXT,
            commodity_price REAL, mine_total_cost REAL, mill_total_cost REAL,
            byproduct_credits REAL, cash_op_cost REAL, total_cash_cost REAL,
            aisc_oz REAL, aic_oz REAL, sustaining_capex_oz REAL,
            total_prod_cost_oz REAL, cash_op_margin_oz REAL,
            synced_at TEXT, PRIMARY KEY (property_id, period, commodity))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS snl_mine_econ_base (
            property_id TEXT, property_name TEXT, period TEXT, commodity TEXT, basis TEXT,
            commodity_price REAL, mine_total_cost REAL, mill_total_cost REAL,
            byproduct_credits REAL, cash_op_cost REAL, total_cash_cost REAL,
            synced_at TEXT, PRIMARY KEY (property_id, period, commodity))"""))

    log.info(f"DB ready at {DB_PATH}")


# ── Write ──────────────────────────────────────────────────────────────────────

def upsert_snapshot(df: pd.DataFrame, snap_date: date | None = None):
    """Persist a scored DataFrame as a daily snapshot."""
    snap_date = snap_date or date.today()
    engine = _engine()

    def _g(row, col, default=None):
        v = row.get(col)
        try:
            f = float(v)
            import math
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return default

    rows = []
    for ticker, row in df.iterrows():
        rows.append({
            "id":               f"{ticker}|{snap_date}",
            "ticker":           ticker,
            "snap_date":        snap_date,
            "fetched_at":       datetime.utcnow(),
            "name":             str(row.get("name", ticker)),
            "commodity":        str(row.get("commodity", "")),
            "stage":            str(row.get("stage", "")),
            "currency":         str(row.get("currency", "AUD")),
            "exchange":         str(row.get("exchange", "")),
            "market_cap":       _g(row, "marketCap"),
            "enterprise_value": _g(row, "enterpriseValue"),
            "price":            _g(row, "regularMarketPrice"),
            "price_to_book":    _g(row, "priceToBook"),
            "ev_ebitda":        _g(row, "ev_ebitda"),
            "ev_revenue":       _g(row, "ev_revenue"),
            "p_cf":             _g(row, "p_cf"),
            "debt_to_equity":   _g(row, "debtToEquity"),
            "current_ratio":    _g(row, "currentRatio"),
            "cash_pct_mcap":    _g(row, "cash_pct_mcap"),
            "net_debt_m":       _g(row, "net_debt_m"),
            "operating_cf":     _g(row, "operatingCashflow"),
            "free_cf":          _g(row, "freeCashflow"),
            "revenue":          _g(row, "totalRevenue"),
            "rsi":              _g(row, "rsi"),
            "wk52_position":    _g(row, "wk52_position"),
            "pct_from_52hi":    _g(row, "pct_from_52hi"),
            "pct_from_52lo":    _g(row, "pct_from_52lo"),
            "price_vs_ma200":   _g(row, "price_vs_ma200"),
            "score_valuation":  _g(row, "score_valuation"),
            "score_health":     _g(row, "score_health"),
            "score_momentum":   _g(row, "score_momentum"),
            "score_commodity":  _g(row, "score_commodity"),
            "score_stage":      _g(row, "score_stage"),
            "score_composite":  _g(row, "score_composite"),
            "grade":            str(row.get("grade", "")),
            "bb_aisc":          _g(row, "bb_AISC_PER_OZ_GOLD_EQUIV"),
            "bb_production":    _g(row, "bb_GOLD_EQUIV_PRODUCTION_ANNUAL"),
            "bb_nav_per_shr":   _g(row, "bb_NET_ASSET_VALUE_PER_SHR"),
            "bb_ev_to_nav":     _g(row, "bb_EV_TO_NAV"),
            "bb_mine_life":     _g(row, "bb_MINE_LIFE_YRS"),
            # S&P / SNL Mining fields
            "spg_p_nav":        _g(row, "spg_p_nav"),
            "spg_reserves_m":   _g(row, "spg_reserves_m"),
            "spg_resources_m":  _g(row, "spg_resources_m"),
            "spg_aisc_per_oz":       _g(row, "spg_aisc_per_oz"),
            "spg_aisc_per_t":        _g(row, "spg_aisc_per_t"),
            "spg_aisc_per_lb":       _g(row, "spg_aisc_per_lb"),
            "spg_aisc_margin":       _g(row, "spg_aisc_margin"),
            "spg_production_cost_t": _g(row, "spg_production_cost_t"),
            "spg_grade_gpt":         _g(row, "spg_grade_gpt"),
            "spg_grade_pct":         _g(row, "spg_grade_pct"),
            "score_mining":          _g(row, "score_mining"),
            # Cash cost
            "spg_cash_cost_oz":      _g(row, "spg_cash_cost_oz"),
            "spg_cash_cost_t":       _g(row, "spg_cash_cost_t"),
            "spg_cash_cost_lb":      _g(row, "spg_cash_cost_lb"),
            # Production, realized price, contained reserves, reserve life
            "spg_production_oz":          _g(row, "spg_production_oz"),
            "spg_production_t":           _g(row, "spg_production_t"),
            "spg_production_lb":          _g(row, "spg_production_lb"),
            "spg_realized_price_oz":      _g(row, "spg_realized_price_oz"),
            "spg_realized_price_t":       _g(row, "spg_realized_price_t"),
            "spg_realized_price_lb":      _g(row, "spg_realized_price_lb"),
            "spg_contained_reserves_oz":  _g(row, "spg_contained_reserves_oz"),
            "spg_contained_reserves_lb":  _g(row, "spg_contained_reserves_lb"),
            "spg_reserve_life":           _g(row, "spg_reserve_life"),
            "return_1m":        _g(row, "return_1m"),
            "return_3m":        _g(row, "return_3m"),
            "avg_turnover":     _g(row, "avg_turnover"),
            # Dividend yield: Yahoo returns as decimal (0.034 = 3.4%) → store as pct
            "dividend_yield": (
                round(float(row.get("dividendYield", None) or
                            row.get("trailingAnnualDividendYield", None) or 0) * 100, 2)
                if (row.get("dividendYield") or row.get("trailingAnnualDividendYield")) else None
            ),
            "analyst_target_mean": _g(row, "targetMeanPrice"),
            "analyst_target_high": _g(row, "targetHighPrice"),
            "analyst_target_low":  _g(row, "targetLowPrice"),
            "analyst_count":       _g(row, "numberOfAnalystOpinions"),
            "analyst_rec_key":     str(row.get("recommendationKey", "") or ""),
            "analyst_rec_mean":    _g(row, "recommendationMean"),
            # Profitability (Yahoo financialData module — already in _INFO_FIELDS)
            "return_on_equity":    _g(row, "returnOnEquity"),
            "operating_margins":   _g(row, "operatingMargins"),
            "gross_margins":       _g(row, "grossMargins"),
            "profit_margins":      _g(row, "profitMargins"),
            # SNL local-derived
            "spg_mine_life":       _g(row, "spg_mine_life"),
            "spg_global_rank":     _g(row, "spg_global_rank"),
            "analyst_upside":      _g(row, "analyst_upside"),
            # Peer-group ranking
            "peer_group":  str(row.get("peer_group", "") or ""),
            "peer_rank":   _g(row, "peer_rank"),
            "peer_n":      _g(row, "peer_n"),
            "peer_pct":    _g(row, "peer_pct"),
        })

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT OR REPLACE INTO stock_snapshots
                (id,ticker,snap_date,fetched_at,name,commodity,stage,
                 currency,exchange,market_cap,enterprise_value,price,
                 price_to_book,ev_ebitda,ev_revenue,p_cf,debt_to_equity,current_ratio,
                 cash_pct_mcap,net_debt_m,operating_cf,free_cf,revenue,
                 rsi,wk52_position,pct_from_52hi,pct_from_52lo,price_vs_ma200,
                 score_valuation,score_health,score_momentum,score_mining,
                 score_commodity,score_stage,score_composite,grade,
                 bb_aisc,bb_production,bb_nav_per_shr,bb_ev_to_nav,bb_mine_life,
                 spg_p_nav,spg_reserves_m,spg_resources_m,
                 spg_aisc_per_oz,spg_aisc_per_t,spg_aisc_per_lb,spg_aisc_margin,
                 spg_production_cost_t,spg_grade_gpt,spg_grade_pct,
                 spg_cash_cost_oz,spg_cash_cost_t,spg_cash_cost_lb,
                 spg_production_oz,spg_production_t,spg_production_lb,
                 spg_realized_price_oz,spg_realized_price_t,spg_realized_price_lb,
                 spg_contained_reserves_oz,spg_contained_reserves_lb,spg_reserve_life,
                 dividend_yield,return_1m,return_3m,avg_turnover,
                 analyst_target_mean,analyst_target_high,analyst_target_low,
                 analyst_count,analyst_rec_key,analyst_rec_mean,
                 return_on_equity,operating_margins,gross_margins,profit_margins,
                 spg_mine_life,spg_global_rank,analyst_upside,
                 peer_group,peer_rank,peer_n,peer_pct)
                VALUES
                (:id,:ticker,:snap_date,:fetched_at,:name,:commodity,:stage,
                 :currency,:exchange,:market_cap,:enterprise_value,:price,
                 :price_to_book,:ev_ebitda,:ev_revenue,:p_cf,:debt_to_equity,:current_ratio,
                 :cash_pct_mcap,:net_debt_m,:operating_cf,:free_cf,:revenue,
                 :rsi,:wk52_position,:pct_from_52hi,:pct_from_52lo,:price_vs_ma200,
                 :score_valuation,:score_health,:score_momentum,:score_mining,
                 :score_commodity,:score_stage,:score_composite,:grade,
                 :bb_aisc,:bb_production,:bb_nav_per_shr,:bb_ev_to_nav,:bb_mine_life,
                 :spg_p_nav,:spg_reserves_m,:spg_resources_m,
                 :spg_aisc_per_oz,:spg_aisc_per_t,:spg_aisc_per_lb,:spg_aisc_margin,
                 :spg_production_cost_t,:spg_grade_gpt,:spg_grade_pct,
                 :spg_cash_cost_oz,:spg_cash_cost_t,:spg_cash_cost_lb,
                 :spg_production_oz,:spg_production_t,:spg_production_lb,
                 :spg_realized_price_oz,:spg_realized_price_t,:spg_realized_price_lb,
                 :spg_contained_reserves_oz,:spg_contained_reserves_lb,:spg_reserve_life,
                 :dividend_yield,:return_1m,:return_3m,:avg_turnover,
                 :analyst_target_mean,:analyst_target_high,:analyst_target_low,
                 :analyst_count,:analyst_rec_key,:analyst_rec_mean,
                 :return_on_equity,:operating_margins,:gross_margins,:profit_margins,
                 :spg_mine_life,:spg_global_rank,:analyst_upside,
                 :peer_group,:peer_rank,:peer_n,:peer_pct)
            """),
            rows,
        )
    log.info(f"Saved {len(rows)} rows for {snap_date}")


# ── Read ───────────────────────────────────────────────────────────────────────

def load_latest() -> pd.DataFrame:
    """Load the most recent snapshot for each ticker."""
    engine = _engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM stock_snapshots
            WHERE snap_date = (
                SELECT MAX(snap_date) FROM stock_snapshots
            )
            ORDER BY score_composite DESC
        """))
        rows = result.fetchall()
        cols = result.keys()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=list(cols))


def load_sector_trends() -> pd.DataFrame:
    """
    Load average composite score per commodity group per snapshot date.
    Used for the sector trend chart in the Charts tab.
    Returns DataFrame: (snap_date, commodity_group, avg_score, n_companies).
    """
    engine = _engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    snap_date,
                    CASE
                        WHEN commodity LIKE 'Gold%'      THEN 'Gold'
                        WHEN commodity LIKE 'Silver%'    THEN 'Silver'
                        WHEN commodity LIKE 'Copper%'    THEN 'Copper'
                        WHEN commodity LIKE 'Uranium%'   THEN 'Uranium'
                        WHEN commodity LIKE 'Nickel%'    THEN 'Nickel'
                        WHEN commodity LIKE 'Lithium%'   THEN 'Lithium'
                        WHEN commodity LIKE 'Iron%'      THEN 'Iron Ore'
                        WHEN commodity LIKE 'Zinc%'      THEN 'Zinc'
                        ELSE 'Other'
                    END AS commodity_group,
                    ROUND(AVG(score_composite), 1) AS avg_score,
                    COUNT(*) AS n_companies
                FROM stock_snapshots
                WHERE score_composite IS NOT NULL
                GROUP BY snap_date, commodity_group
                ORDER BY snap_date ASC
            """))
            rows = result.fetchall()
            cols = list(result.keys())
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()


def load_prev_scores() -> pd.DataFrame:
    """
    Load composite scores and grade from the second-most-recent snapshot date.
    Used to compute score Δ and detect grade transitions since last refresh.
    Returns DataFrame with columns [ticker, score_prev, grade_prev].
    """
    engine = _engine()
    try:
        with engine.connect() as conn:
            dates = conn.execute(text("""
                SELECT DISTINCT snap_date FROM stock_snapshots
                ORDER BY snap_date DESC LIMIT 2
            """)).fetchall()
        if len(dates) < 2:
            return pd.DataFrame(columns=["ticker", "score_prev", "grade_prev"])
        prev_date = dates[1][0]
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT ticker, score_composite, grade FROM stock_snapshots WHERE snap_date = :d"),
                {"d": prev_date},
            )
            rows = result.fetchall()
        df = pd.DataFrame(rows, columns=["ticker", "score_prev", "grade_prev"])
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "score_prev", "grade_prev"])


def load_backtest_data() -> pd.DataFrame:
    """
    Load all (ticker, snap_date, name, commodity, score_composite, price, grade)
    across every snapshot date, sorted chronologically.
    Used to backtest whether high scores predicted higher subsequent returns.
    Returns DataFrame with one row per (ticker, snap_date).
    """
    engine = _engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT ticker, snap_date, name, commodity,
                       score_composite, price, grade
                FROM stock_snapshots
                WHERE score_composite IS NOT NULL
                  AND price IS NOT NULL
                  AND price > 0
                ORDER BY ticker, snap_date ASC
            """))
            rows = result.fetchall()
            cols = ["ticker", "snap_date", "name", "commodity",
                    "score_composite", "price", "grade"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows, columns=cols)
        df["snap_date"] = pd.to_datetime(df["snap_date"])
        return df
    except Exception:
        return pd.DataFrame()


def load_return_matrix(days: int = 30) -> pd.DataFrame:
    """
    Load daily return_3m for every ticker across the last `days` snapshot dates.
    Returns a DataFrame shaped (dates × tickers) suitable for pairwise correlation.
    Only tickers with ≥2 non-null observations are included.
    """
    engine = _engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT snap_date, ticker, return_3m
                    FROM stock_snapshots
                    WHERE snap_date >= date('now', :offset)
                      AND return_3m IS NOT NULL
                    ORDER BY snap_date ASC
                """),
                {"offset": f"-{days} days"},
            )
            rows = result.fetchall()
        if not rows:
            return pd.DataFrame()
        tmp = pd.DataFrame(rows, columns=["snap_date", "ticker", "return_3m"])
        pivot = tmp.pivot(index="snap_date", columns="ticker", values="return_3m")
        # Drop tickers that are almost always null across the window
        pivot = pivot.loc[:, pivot.notna().sum() >= 2]
        return pivot
    except Exception:
        return pd.DataFrame()


def load_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """Load score + price history for a single ticker."""
    engine = _engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT snap_date, price, score_composite, score_valuation,
                       score_health, score_momentum, score_mining,
                       score_commodity, score_stage, rsi, wk52_position
                FROM stock_snapshots
                WHERE ticker = :tk
                ORDER BY snap_date DESC
                LIMIT :days
            """),
            {"tk": ticker, "days": days},
        )
        rows  = result.fetchall()
        cols  = result.keys()
    return pd.DataFrame(rows, columns=list(cols))


def last_refresh() -> str:
    """Return the timestamp of the most recent fetch."""
    engine = _engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT MAX(fetched_at) FROM stock_snapshots")
            ).fetchone()
        return str(row[0]) if row and row[0] else "Never"
    except Exception:
        return "Never"


# ── Commodity spot price history ──────────────────────────────────────────────

def upsert_commodity_prices(prices: dict[str, float], price_date=None) -> None:
    """Store daily commodity spot prices. Call after fetch_spot_prices()."""
    price_date = price_date or date.today()
    engine = _engine()
    with engine.begin() as conn:
        for commodity, price in prices.items():
            try:
                conn.execute(
                    text("""
                        INSERT OR REPLACE INTO commodity_prices (price_date, commodity, price)
                        VALUES (:d, :c, :p)
                    """),
                    {"d": price_date, "c": commodity, "p": float(price)},
                )
            except Exception:
                pass
    log.info(f"Commodity prices saved for {price_date}: {list(prices.keys())}")


def load_commodity_price_history(days: int = 180) -> pd.DataFrame:
    """
    Load commodity spot price history for trend charts.
    Returns DataFrame with columns [price_date, commodity, price].
    """
    engine = _engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT price_date, commodity, price
                    FROM commodity_prices
                    WHERE price_date >= date('now', :offset)
                    ORDER BY price_date ASC
                """),
                {"offset": f"-{days} days"},
            )
            rows = result.fetchall()
            cols = list(result.keys())
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()


# ── Watchlist ──────────────────────────────────────────────────────────────────

def get_watchlist() -> set[str]:
    """Return the set of watchlisted tickers."""
    try:
        with _engine().connect() as conn:
            rows = conn.execute(text("SELECT ticker FROM watchlist")).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def add_to_watchlist(ticker: str, note: str = "") -> None:
    with _engine().begin() as conn:
        conn.execute(
            text("INSERT OR IGNORE INTO watchlist (ticker, note) VALUES (:t, :n)"),
            {"t": ticker, "n": note},
        )


def remove_from_watchlist(ticker: str) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text("DELETE FROM watchlist WHERE ticker = :t"),
            {"t": ticker},
        )


def get_watchlist_note(ticker: str) -> str:
    try:
        with _engine().connect() as conn:
            row = conn.execute(
                text("SELECT note FROM watchlist WHERE ticker = :t"),
                {"t": ticker},
            ).fetchone()
        return row[0] or "" if row else ""
    except Exception:
        return ""


def update_watchlist_note(ticker: str, note: str) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text("UPDATE watchlist SET note = :n WHERE ticker = :t"),
            {"n": note, "t": ticker},
        )


# ── User settings ──────────────────────────────────────────────────────────────

def get_setting(key: str, default=None):
    try:
        with _engine().connect() as conn:
            row = conn.execute(
                text("SELECT value FROM user_settings WHERE key = :k"),
                {"k": key},
            ).fetchone()
        return row[0] if row else default
    except Exception:
        return default


def set_setting(key: str, value) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text("INSERT OR REPLACE INTO user_settings (key, value) VALUES (:k, :v)"),
            {"k": key, "v": str(value)},
        )


def get_score_weights() -> dict:
    """Load persisted score weights; returns defaults if never saved."""
    defaults = {"valuation": 30, "health": 20, "momentum": 15,
                "mining": 25, "commodity": 5, "stage": 5}
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("SELECT key, value FROM user_settings WHERE key LIKE 'weight_%'")
            ).fetchall()
        saved = {r[0].replace("weight_", ""): int(r[1]) for r in rows}
        return {**defaults, **saved} if saved else defaults
    except Exception:
        return defaults


def save_score_weights(weights: dict) -> None:
    with _engine().begin() as conn:
        for k, v in weights.items():
            conn.execute(
                text("INSERT OR REPLACE INTO user_settings (key, value) VALUES (:k, :v)"),
                {"k": f"weight_{k}", "v": str(int(v))},
            )


# ── Portfolio positions ────────────────────────────────────────────────────────

def get_positions() -> pd.DataFrame:
    """
    Return all positions as a DataFrame with columns:
    [ticker, shares, avg_cost, currency, updated_at].
    """
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("SELECT ticker, shares, avg_cost, currency, updated_at FROM positions")
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["ticker", "shares", "avg_cost", "currency", "updated_at"])
        return pd.DataFrame(rows, columns=["ticker", "shares", "avg_cost", "currency", "updated_at"])
    except Exception:
        return pd.DataFrame(columns=["ticker", "shares", "avg_cost", "currency", "updated_at"])


def upsert_position(ticker: str, shares: float, avg_cost: float | None,
                    currency: str = "AUD") -> None:
    """Save or update a position. Pass shares=0 to zero out (keeps row for history)."""
    with _engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO positions (ticker, shares, avg_cost, currency, updated_at)
                VALUES (:t, :s, :c, :cur, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    shares     = excluded.shares,
                    avg_cost   = excluded.avg_cost,
                    currency   = excluded.currency,
                    updated_at = CURRENT_TIMESTAMP
            """),
            {"t": ticker, "s": float(shares),
             "c": float(avg_cost) if avg_cost is not None else None,
             "cur": currency},
        )


def delete_position(ticker: str) -> None:
    with _engine().begin() as conn:
        conn.execute(text("DELETE FROM positions WHERE ticker = :t"), {"t": ticker})


# ── Price targets ──────────────────────────────────────────────────────────────

def get_price_target(ticker: str) -> float | None:
    """Return the stored price target for a watchlisted ticker, or None."""
    try:
        with _engine().connect() as conn:
            row = conn.execute(
                text("SELECT price_target FROM watchlist WHERE ticker = :t"),
                {"t": ticker},
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def set_price_target(ticker: str, target: float | None) -> None:
    """Persist (or clear) a price target for a watchlisted ticker."""
    with _engine().begin() as conn:
        conn.execute(
            text("UPDATE watchlist SET price_target = :v WHERE ticker = :t"),
            {"v": float(target) if target is not None else None, "t": ticker},
        )


def get_all_price_targets() -> dict[str, float]:
    """Return {ticker: price_target} for all watchlisted tickers that have a target set."""
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("SELECT ticker, price_target FROM watchlist WHERE price_target IS NOT NULL")
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows}
    except Exception:
        return {}


# ── Filter presets ─────────────────────────────────────────────────────────────

import json as _json

_PRESET_PREFIX = "filter_preset__"


def save_filter_preset(name: str, filters: dict) -> None:
    """Persist a named filter preset to user_settings."""
    with _engine().begin() as conn:
        conn.execute(
            text("INSERT OR REPLACE INTO user_settings (key, value) VALUES (:k, :v)"),
            {"k": f"{_PRESET_PREFIX}{name}", "v": _json.dumps(filters)},
        )


def load_filter_presets() -> dict[str, dict]:
    """Return all saved presets as {name: filters_dict}."""
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("SELECT key, value FROM user_settings WHERE key LIKE :p"),
                {"p": f"{_PRESET_PREFIX}%"},
            ).fetchall()
        return {
            r[0].removeprefix(_PRESET_PREFIX): _json.loads(r[1])
            for r in rows
        }
    except Exception:
        return {}


def delete_filter_preset(name: str) -> None:
    """Remove a named filter preset."""
    with _engine().begin() as conn:
        conn.execute(
            text("DELETE FROM user_settings WHERE key = :k"),
            {"k": f"{_PRESET_PREFIX}{name}"},
        )


# ── Transaction log ────────────────────────────────────────────────────────────

def add_transaction(ticker: str, trans_date, shares: float, price: float,
                    trans_type: str = "buy", note: str = "") -> None:
    """Record a buy or sell transaction."""
    with _engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO transactions (ticker, trans_date, shares, price, trans_type, note)
                VALUES (:tk, :d, :s, :p, :t, :n)
            """),
            {"tk": ticker, "d": str(trans_date), "s": float(shares),
             "p": float(price), "t": trans_type, "n": note or ""},
        )


def get_transactions(ticker: str | None = None) -> pd.DataFrame:
    """Return all transactions, optionally filtered by ticker."""
    try:
        with _engine().connect() as conn:
            if ticker:
                rows = conn.execute(
                    text("SELECT * FROM transactions WHERE ticker = :tk ORDER BY trans_date DESC, id DESC"),
                    {"tk": ticker},
                ).fetchall()
            else:
                rows = conn.execute(
                    text("SELECT * FROM transactions ORDER BY trans_date DESC, id DESC")
                ).fetchall()
            cols = ["id", "ticker", "trans_date", "shares", "price",
                    "trans_type", "note", "created_at"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    except Exception:
        return pd.DataFrame(columns=["id", "ticker", "trans_date", "shares",
                                     "price", "trans_type", "note", "created_at"])


def delete_transaction(trans_id: int) -> None:
    """Remove a single transaction by its id."""
    with _engine().begin() as conn:
        conn.execute(text("DELETE FROM transactions WHERE id = :i"), {"i": trans_id})
