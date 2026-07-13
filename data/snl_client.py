"""
SNL Metals & Mining — Snowflake live query client.

Full paid subscription — local caching is permitted.
Local SQLite cache is maintained by snl_sync.py (run manually or via scheduler).
This module provides live Snowflake queries for on-demand lookups;
bulk/historical data is read from the local snl_* tables in mining_screener.db.

Connection is cached per Streamlit session via @st.cache_resource.
Outside Streamlit, use get_connection() directly.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ── ticker → SNL institution key mapping ──────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent.parent / "_asx_snl_ticker_mapping.json"


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, dict]:
    """Load ticker→SNL key mapping from JSON (built during exploration)."""
    if _MAPPING_FILE.exists():
        with open(_MAPPING_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_snl_key(ticker: str) -> str | None:
    """Return SNL institution key for a ticker (bare or .AX suffix, with/without class suffix)."""
    bare = ticker.replace(".AX", "").replace(".ASX", "")
    mapping = _load_mapping()
    # Try exact match first, then strip share-class suffix (.A, .B, -A, -B)
    for candidate in (bare, ticker):
        if candidate in mapping:
            return str(mapping[candidate]["snl_key"])
    # Strip trailing class suffix: TECK-B → TECK, BRK-A → BRK
    import re
    stripped = re.sub(r"[-.]?[AB]$", "", bare)
    if stripped != bare and stripped in mapping:
        return str(mapping[stripped]["snl_key"])
    return None


# ── Snowflake connection ───────────────────────────────────────────────────────

def _load_env() -> None:
    """Load .env once if not already loaded."""
    try:
        from dotenv import load_dotenv
        from pathlib import Path as _Path
        _env = _Path(__file__).parent.parent / ".env"
        if _env.exists():
            load_dotenv(_env, override=False)
    except Exception:
        pass


_load_env()


def _snowflake_params() -> dict[str, str]:
    return {
        "account":   os.getenv("SNOWFLAKE_ACCOUNT", ""),
        "user":      os.getenv("SNOWFLAKE_USER", ""),
        "password":  os.getenv("SNOWFLAKE_PASSWORD", ""),
        "database":  "MI_XPRESSCLOUD",
        "schema":    "XPRESSFEED",
        "warehouse": "XF_READER_LINGBAO_WH",
    }


def is_configured() -> bool:
    # Master switch — set USE_SNOWFLAKE=false in .env to disable all live
    # Snowflake calls (credentials stay in place for later re-enabling).
    if os.getenv("USE_SNOWFLAKE", "true").strip().lower() != "true":
        return False
    p = _snowflake_params()
    return bool(p["account"] and p["user"] and p["password"])


def get_connection():
    """Return a live Snowflake connection. Caller is responsible for closing."""
    import snowflake.connector
    p = _snowflake_params()
    return snowflake.connector.connect(**p, login_timeout=20)


def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Execute SQL and return list of row dicts. Returns [] if table is unavailable."""
    import snowflake.connector.errors as _sf_errors
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except _sf_errors.ProgrammingError as e:
        log.warning(f"[SNL] query unavailable (subscription?): {e}")
        return []
    finally:
        conn.close()


# ── public API ────────────────────────────────────────────────────────────────

def get_company_data(ticker: str) -> dict[str, Any]:
    """
    Return latest-year SNL data for a company.
    Keys mirror the spg_ field names expected by scorer.py and app.py.
    Returns empty dict if ticker not in mapping or Snowflake unavailable.
    """
    if not is_configured():
        return {}
    snl_key = get_snl_key(ticker)
    if not snl_key:
        return {}

    # ── Production + AISC (company level, most recent available year) ─────────
    rows = _query("""
        SELECT
            INSTITUTIONNAME_1   AS company,
            SNLDATASOURCEPERIOD_3 AS period,
            COMMODITY_4          AS commodity,
            ATTRIBUTABLEPRODUCTIONOZ_5   AS prod_oz,
            ATTRIBUTABLEPRODUCTIONTONNE_6 AS prod_t,
            ATTRIBUTABLEPRODUCTIONLB_7    AS prod_lb,
            CASHCOSTOZ_13                 AS cash_cost_oz,
            CASHCOSTTONNE_14              AS cash_cost_t,
            CASHCOSTLB_15                 AS cash_cost_lb,
            ALLINCOSTOZ_21                AS aic_oz,
            ALLINSUSTAININGCOSTOZ_34      AS aisc_oz,
            ALLINSUSTAININGCOSTTONNE_35   AS aisc_t,
            ALLINSUSTAININGCOSTLB_36      AS aisc_lb,
            AVERAGEPRICEREALIZEDOZ_29     AS realized_price_oz,
            AVERAGEPRICEREALIZEDTONNE_30  AS realized_price_t,
            AVERAGEPRICEREALIZEDLB_31     AS realized_price_lb,
            REVENUEPERCOMMODITY_33        AS revenue
        FROM SNL_MMCOMPANIES_PRODUCTIONCOSTSALES_C
        WHERE CAST(SNLINSTITUTIONKEY_2 AS VARCHAR) = %s
          AND SNLDATASOURCEPERIOD_3 IN ('2024Y', '2023Y', '2022Y')
        ORDER BY period DESC, commodity
    """, (snl_key,))

    if not rows:
        return {}

    # ── Global production ranking ──────────────────────────────────────────────
    rank_rows = _query("""
        SELECT
            SNLDATASOURCEPERIOD_4 AS period,
            COMMODITY_6           AS commodity,
            GLOBALRANKBYCOMMODITY_7       AS global_rank,
            PRODUCTIONAMOUNTBYCOMMODITYOZ_8  AS prod_oz,
            PRODUCTIONAMOUNTBYCOMMODITYTONNE_9 AS prod_t,
            PRODUCTIONAMOUNTBYCOMMODITYLB_11  AS prod_lb,
            SHAREOFWORLDBYCOMMODITY_13     AS world_share_pct
        FROM SNL_MMCOMPANIES_SNLAGGREGATEDPRODBYCOMMODITY
        WHERE CAST(SNLINSTITUTIONKEY_1 AS VARCHAR) = %s
          AND SNLDATASOURCEPERIOD_4 IN ('2024Y', '2023Y')
        ORDER BY period DESC, commodity
    """, (snl_key,))

    return {
        "snl_key": snl_key,
        "found":   True,
        "production_costs": rows,      # full list, all commodities × years
        "global_rank":      rank_rows,
    }


def get_company_aisc_history(ticker: str) -> list[dict]:
    """
    Return 3-year AISC history for charting (2022–2024).
    Returns list of {period, commodity, aisc_oz, cash_cost_oz, realized_price_oz}.
    """
    if not is_configured():
        return []
    snl_key = get_snl_key(ticker)
    if not snl_key:
        return []

    rows = _query("""
        SELECT
            SNLDATASOURCEPERIOD_3 AS period,
            COMMODITY_4           AS commodity,
            CASHCOSTOZ_13         AS cash_cost_oz,
            ALLINCOSTOZ_21        AS aic_oz,
            ALLINSUSTAININGCOSTOZ_34 AS aisc_oz,
            AVERAGEPRICEREALIZEDOZ_29 AS realized_price_oz,
            ATTRIBUTABLEPRODUCTIONOZ_5 AS prod_oz
        FROM SNL_MMCOMPANIES_PRODUCTIONCOSTSALES_C
        WHERE CAST(SNLINSTITUTIONKEY_2 AS VARCHAR) = %s
          AND SNLDATASOURCEPERIOD_3 IN ('2022Y', '2023Y', '2024Y')
          AND COMMODITY_4 IN ('Gold', 'Silver', 'Copper', 'Zinc',
                              'Nickel', 'Uranium', 'PGM', 'Iron Ore')
        ORDER BY commodity, period
    """, (snl_key,))
    return rows


def get_property_reserves(ticker: str) -> list[dict]:
    """
    Return top properties owned by this company with reserve data.
    """
    if not is_configured():
        return []
    snl_key = get_snl_key(ticker)
    if not snl_key:
        return []

    # Get property IDs owned by this company
    owner_rows = _query("""
        SELECT
            PROPERTYID_1,
            PROPERTYNAME_2,
            CURRENTEQUITYOWNERSHIPPERCENT_11
        FROM SNL_MMPROPERTIES_OWNER
        WHERE CAST(OWNERSNLINSTITUTIONKEY_9 AS VARCHAR) = %s
        ORDER BY CURRENTEQUITYOWNERSHIPPERCENT_11 DESC NULLS LAST
        LIMIT 20
    """, (snl_key,))

    if not owner_rows:
        return []

    prop_ids = [str(r["PROPERTYID_1"]) for r in owner_rows]
    id_sql   = ", ".join(f"'{p}'" for p in prop_ids)
    pct_map  = {str(r["PROPERTYID_1"]): r["CURRENTEQUITYOWNERSHIPPERCENT_11"] for r in owner_rows}

    # Get general info for those properties
    gen_rows = _query(f"""
        SELECT
            PROPERTYID_1,
            PROPERTYNAME_2,
            DEVELOPMENTSTAGE_4,
            ACTIVITYSTATUS_5,
            PRIMARYCOMMODITY_31,
            COUNTRYNAME_22,
            STATEPERPROVINCE_21
        FROM SNL_MMPROPERTIES_GENERAL
        WHERE PROPERTYID_1 IN ({id_sql})
        ORDER BY PROPERTYID_1
    """)

    # Merge ownership % in with readable names
    result = []
    for r in gen_rows:
        result.append({
            "PROPERTY_ID":       r.get("PROPERTYID_1"),
            "PROPERTY_NAME":     r.get("PROPERTYNAME_2"),
            "STAGE":             r.get("DEVELOPMENTSTAGE_4"),
            "STATUS":            r.get("ACTIVITYSTATUS_5"),
            "PRIMARY_COMMODITY": r.get("PRIMARYCOMMODITY_31"),
            "COUNTRY":           r.get("COUNTRYNAME_22"),
            "STATE_PROVINCE":    r.get("STATEPERPROVINCE_21"),
            "PCT_OWN":           pct_map.get(str(r.get("PROPERTYID_1"))),
        })
    return result


def get_exploration_budget(ticker: str) -> list[dict]:
    """
    Return exploration budget data for Canada/US/Australia.
    """
    if not is_configured():
        return []
    snl_key = get_snl_key(ticker)
    if not snl_key:
        return []

    rows = _query("""
        SELECT *
        FROM SNL_EXPLORATIONBUDGET_USCANADAAU
        WHERE CAST(SNLINSTITUTIONKEY AS VARCHAR) = %s
        ORDER BY YEAR DESC
        LIMIT 10
    """, (snl_key,))
    return rows


def get_batch_metrics(year: str = "2024Y") -> dict[str, dict]:
    """
    Return production/cost metrics for ALL tickers in our mapping for a given year.
    Returns dict keyed by SNL institution key string → row dict with commodity-keyed data.

    Used for runtime overlay: fills spg_ columns in memory without writing to SQLite.
    Live query only — trial-compliant (§3.2.2).
    """
    if not is_configured():
        return {}

    mapping = _load_mapping()
    if not mapping:
        return {}

    snl_keys = list({str(v["snl_key"]) for v in mapping.values()})
    key_sql   = ", ".join(f"'{k}'" for k in snl_keys)

    rows = _query(f"""
        SELECT
            CAST(c.SNLINSTITUTIONKEY_2 AS VARCHAR) AS snl_key,
            c.COMMODITY_4                          AS commodity,
            c.ATTRIBUTABLEPRODUCTIONOZ_5           AS prod_oz,
            c.ATTRIBUTABLEPRODUCTIONTONNE_6        AS prod_t,
            c.ATTRIBUTABLEPRODUCTIONLB_7           AS prod_lb,
            c.CASHCOSTOZ_13                        AS cash_cost_oz,
            c.CASHCOSTTONNE_14                     AS cash_cost_t,
            c.CASHCOSTLB_15                        AS cash_cost_lb,
            c.ALLINSUSTAININGCOSTOZ_34             AS aisc_oz,
            c.ALLINSUSTAININGCOSTTONNE_35          AS aisc_t,
            c.ALLINSUSTAININGCOSTLB_36             AS aisc_lb,
            c.AVERAGEPRICEREALIZEDOZ_29            AS realized_price_oz,
            c.AVERAGEPRICEREALIZEDTONNE_30         AS realized_price_t,
            c.AVERAGEPRICEREALIZEDLB_31            AS realized_price_lb
        FROM SNL_MMCOMPANIES_PRODUCTIONCOSTSALES_C c
        WHERE c.SNLINSTITUTIONKEY_2 IN ({key_sql})
          AND c.SNLDATASOURCEPERIOD_3 = '{year}'
          AND c.COMMODITY_4 IN ('Gold','Silver','Copper','Zinc',
                                'Nickel','Uranium','PGM','Iron Ore')
    """)

    # Build: snl_key → {commodity → row}
    result: dict[str, dict] = {}
    for r in rows:
        key  = str(r.get("SNL_KEY", ""))
        comm = str(r.get("COMMODITY", ""))
        if key not in result:
            result[key] = {}
        result[key][comm] = r
    return result


# ── Column-discovery helpers ──────────────────────────────────────────────────

def _find_col(cols: list[str], *patterns: str) -> str | None:
    """Return first column name whose uppercase text contains any pattern."""
    for pat in patterns:
        pat_u = pat.upper()
        for c in cols:
            if pat_u in c.upper():
                return c
    return None


@lru_cache(maxsize=10)
def _discover_key_col(table: str) -> str | None:
    """
    Probe a table with LIMIT 1 to find the SNLINSTITUTIONKEY column.
    Result is cached so the extra round-trip happens only once per session.
    """
    try:
        rows = _query(f"SELECT * FROM {table} LIMIT 1")
        if not rows:
            return None
        col = _find_col(list(rows[0].keys()), "INSTITUTIONKEY")
        if col:
            log.info(f"[SNL discover] {table}: institution key column = {col}")
        return col
    except Exception as e:
        log.warning(f"[SNL discover] {table}: {e}")
        return None


# ── Batch reserves (in-situ value) ───────────────────────────────────────────

def get_batch_reserves(period: str = "4Q2024") -> dict[str, dict]:
    """
    Return in-situ reserve/resource values ($M) for all mapped tickers.
    Queries SNL_MMCOMPANIES_INSITUVALUE (wide format — one row per company).

    Column layout confirmed 2026-04-27:
      INSITUVALUERESERVES_5          — proven + probable reserves only
      INSITUVALUETOTALRESOURCES_6    — total resources (M+I+Inferred)
      INSITUVALUERESERVESRESOURCES_4 — combined R&R (fallback)

    Returns {snl_key: {reserves_m?, resources_m?}}.
    Live query — not stored (trial §3.2.2).
    """
    if not is_configured():
        return {}
    mapping = _load_mapping()
    if not mapping:
        return {}

    TABLE   = "SNL_MMCOMPANIES_INSITUVALUE"
    key_col = _discover_key_col(TABLE)
    if not key_col:
        return {}

    snl_keys = list({str(v["snl_key"]) for v in mapping.values()})
    key_sql  = ", ".join(f"'{k}'" for k in snl_keys)

    try:
        rows = _query(f"""
            SELECT
                CAST({key_col} AS VARCHAR)      AS snl_key,
                SNLDATASOURCEPERIOD_3            AS period,
                INSITUVALUERESERVES_5            AS reserves_m,
                INSITUVALUETOTALRESOURCES_6      AS resources_m,
                INSITUVALUERESERVESRESOURCES_4   AS rr_m
            FROM {TABLE}
            WHERE CAST({key_col} AS VARCHAR) IN ({key_sql})
            ORDER BY {key_col}, SNLDATASOURCEPERIOD_3 DESC
        """)
    except Exception as e:
        log.warning(f"[SNL reserves] query failed: {e}")
        return {}

    def _f(v) -> float | None:
        try:
            x = float(v)
            return x if np.isfinite(x) and x >= 0 else None
        except (TypeError, ValueError):
            return None

    result: dict[str, dict] = {}

    for r in rows:
        key = str(r.get("SNL_KEY", ""))
        if not key or key in result:   # keep most-recent period (rows ordered DESC)
            continue

        bucket: dict = {}
        rv = _f(r.get("RESERVES_M"))
        rc = _f(r.get("RESOURCES_M"))
        rr = _f(r.get("RR_M"))

        if rv is not None and rv > 0:
            bucket["reserves_m"] = round(rv, 2)
        elif rr is not None and rr > 0:
            # No explicit reserves column → use combined R&R as proxy
            bucket["reserves_m"] = round(rr, 2)

        if rc is not None and rc > 0:
            bucket["resources_m"] = round(rc, 2)

        if bucket:
            result[key] = bucket

    log.info(f"[SNL reserves] loaded for {len(result)} companies")
    return result


# ── Batch grade and contained metals ─────────────────────────────────────────

def get_batch_grade() -> dict[str, dict]:
    """
    Return aggregate grade and contained R&R metal for all mapped tickers.
    Queries SNL_MMCOMPANIES_AGGREGATEATTRIBUTABLERR.

    Column layout confirmed 2026-04-27:
      PRIMARYGRADERESERVESRESOURCESGPERTONNE_59  — combined R&R grade (g/t)
      PRIMARYGRADERESERVESRESOURCESPC_65         — combined R&R grade (%)
      PRIMARYCONTAINEDRESERVESRESOURCESOZ_33     — combined R&R contained (oz)
      PRIMARYCONTAINEDRESERVESRESOURCESLB_45     — combined R&R contained (lb)
      PRIMARYCONTAINEDRESERVESRESOURCESTONNE_51  — combined R&R contained (tonne)

    Returns {snl_key: {grade_gpt?, grade_pct?,
                        contained_reserves_oz?, contained_reserves_lb?,
                        contained_reserves_t?}}.
    Live query — not stored (trial §3.2.2).
    """
    if not is_configured():
        return {}
    mapping = _load_mapping()
    if not mapping:
        return {}

    TABLE   = "SNL_MMCOMPANIES_AGGREGATEATTRIBUTABLERR"
    key_col = _discover_key_col(TABLE)
    if not key_col:
        return {}

    snl_keys = list({str(v["snl_key"]) for v in mapping.values()})
    key_sql  = ", ".join(f"'{k}'" for k in snl_keys)

    try:
        rows = _query(f"""
            SELECT
                CAST({key_col} AS VARCHAR)                  AS snl_key,
                SNLDATASOURCEPERIOD_4                        AS period,
                PRIMARYGRADERESERVESRESOURCESGPERTONNE_59   AS grade_gpt,
                PRIMARYGRADERESERVESRESOURCESPC_65          AS grade_pct,
                PRIMARYCONTAINEDRESERVESRESOURCESOZ_33      AS contained_oz,
                PRIMARYCONTAINEDRESERVESRESOURCESLB_45      AS contained_lb,
                PRIMARYCONTAINEDRESERVESRESOURCESTONNE_51   AS contained_t
            FROM {TABLE}
            WHERE CAST({key_col} AS VARCHAR) IN ({key_sql})
            ORDER BY {key_col}, SNLDATASOURCEPERIOD_4 DESC
        """)
    except Exception as e:
        log.warning(f"[SNL grade] query failed: {e}")
        return {}

    def _f(v) -> float | None:
        try:
            x = float(v)
            return x if np.isfinite(x) and x > 0 else None
        except (TypeError, ValueError):
            return None

    result: dict[str, dict] = {}

    for r in rows:
        key = str(r.get("SNL_KEY", ""))
        if not key or key in result:   # keep most-recent period
            continue

        row: dict = {}
        gpt = _f(r.get("GRADE_GPT"))
        pct = _f(r.get("GRADE_PCT"))
        oz  = _f(r.get("CONTAINED_OZ"))
        lb  = _f(r.get("CONTAINED_LB"))
        t   = _f(r.get("CONTAINED_T"))

        if gpt: row["grade_gpt"]             = round(gpt, 4)
        if pct: row["grade_pct"]             = round(pct, 4)
        if oz:  row["contained_reserves_oz"] = round(oz, 0)
        if lb:  row["contained_reserves_lb"] = round(lb, 0)
        if t:   row["contained_reserves_t"]  = round(t, 0)

        if row:
            result[key] = row

    log.info(f"[SNL grade] loaded for {len(result)} companies")
    return result


def get_batch_mine_life_local() -> dict[str, float]:
    """
    Compute ownership-weighted mine life (years from today) for each mapped company
    using the locally-cached snl_mine_econ_precious + snl_mine_econ_base tables.

    Mine life = weighted-average of (last_production_year - current_year) across all
    properties owned, weighted by equity ownership %.

    Returns {snl_key: mine_life_years}.  No Snowflake round-trip required.
    """
    import sqlite3
    from pathlib import Path as _Path
    try:
        from config import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("""
            WITH max_year AS (
                SELECT property_id,
                       MAX(CAST(SUBSTR(period, 1, 4) AS INTEGER)) AS last_yr
                FROM snl_mine_econ_precious
                WHERE period LIKE '____Y'
                GROUP BY property_id
                UNION ALL
                SELECT property_id,
                       MAX(CAST(SUBSTR(period, 1, 4) AS INTEGER)) AS last_yr
                FROM snl_mine_econ_base
                WHERE period LIKE '____Y'
                GROUP BY property_id
            ),
            prop_life AS (
                SELECT property_id,
                       MAX(last_yr) - CAST(strftime('%Y','now') AS INTEGER) AS mine_life
                FROM max_year
                GROUP BY property_id
                HAVING mine_life > 0
            ),
            co_life AS (
                SELECT o.snl_key,
                       SUM(pl.mine_life * o.pct_own / 100.0)
                           / NULLIF(SUM(o.pct_own / 100.0), 0) AS weighted_life
                FROM snl_property_owner o
                JOIN prop_life pl ON o.property_id = pl.property_id
                WHERE o.pct_own > 0
                GROUP BY o.snl_key
            )
            SELECT snl_key, ROUND(weighted_life, 1) AS mine_life
            FROM co_life
            WHERE mine_life > 0 AND mine_life < 100
        """).fetchall()
        conn.close()
        result = {row[0]: row[1] for row in rows if row[1] is not None}
        log.info(f"[SNL local] mine_life loaded for {len(result)} companies")
        return result
    except Exception as e:
        log.warning(f"[SNL local] mine_life query failed: {e}")
        return {}


def get_batch_global_rank_local() -> dict[str, dict]:
    """
    Load global production rank from the locally-cached snl_company_ranking table.
    Returns {snl_key: {commodity: global_rank}}.  No Snowflake round-trip.
    """
    import sqlite3
    try:
        from config import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT snl_key, commodity, global_rank "
            "FROM snl_company_ranking "
            "WHERE period = '2024Y' "
            "  AND ownership_method = 'Equity' "
            "  AND global_rank IS NOT NULL "
            "ORDER BY snl_key, global_rank"
        ).fetchall()
        conn.close()
        result: dict[str, dict] = {}
        for snl_key, commodity, rank in rows:
            result.setdefault(snl_key, {})[commodity] = int(rank)
        log.info(f"[SNL local] global_rank loaded for {len(result)} companies")
        return result
    except Exception as e:
        log.warning(f"[SNL local] global_rank query failed: {e}")
        return {}


def get_all_companies_aisc(year: str = "2024Y",
                           commodity: str = "Gold") -> list[dict]:
    """
    Return AISC for all companies for a given year/commodity.
    Used for cross-sectional ranking in the screener.
    Live query — not stored.
    """
    if not is_configured():
        return []
    rows = _query("""
        SELECT
            p.TICKER_3              AS ticker,
            p.INSTITUTIONNAME_2     AS company,
            c.ALLINSUSTAININGCOSTOZ_34 AS aisc_oz,
            c.CASHCOSTOZ_13            AS cash_cost_oz,
            c.ATTRIBUTABLEPRODUCTIONOZ_5 AS prod_oz,
            c.AVERAGEPRICEREALIZEDOZ_29  AS realized_price_oz,
            p.GLOBALRANKBYCOMMODITY_7    AS global_rank,
            p.SHAREOFWORLDBYCOMMODITY_13 AS world_share_pct
        FROM SNL_MMCOMPANIES_SNLAGGREGATEDPRODBYCOMMODITY p
        JOIN SNL_MMCOMPANIES_PRODUCTIONCOSTSALES_C c
          ON CAST(p.SNLINSTITUTIONKEY_1 AS VARCHAR) =
             CAST(c.SNLINSTITUTIONKEY_2 AS VARCHAR)
         AND p.SNLDATASOURCEPERIOD_4 = c.SNLDATASOURCEPERIOD_3
         AND p.COMMODITY_6           = c.COMMODITY_4
        WHERE p.SNLDATASOURCEPERIOD_4 = %s
          AND p.COMMODITY_6           = %s
          AND c.ALLINSUSTAININGCOSTOZ_34 IS NOT NULL
        ORDER BY c.ALLINSUSTAININGCOSTOZ_34
    """, (year, commodity))
    return rows
