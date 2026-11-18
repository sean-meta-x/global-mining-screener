"""Gold base price and scenario sensitivity for gold miners.

data/gold_consensus.json holds single derived display values exported from the fund's local Bloomberg
store (gold.export_consensus): next year's analyst consensus median (METFGD <YY> Index) and the latest
gold spot (xau_spot), each with its as-of date - never a series. Without the file the base falls back
to the gold spot in config.

Base price for valuation (AISC margin): next year's consensus.
Scenarios: what-ifs, not forecasts - spot, US$3,000 / 3,500 / 4,000 and the consensus. Share prices
today reflect spot, so every scenario is measured from spot: move = (scenario / spot) ** beta - 1, where
beta is the miner's historical elasticity to gold (daily log returns on the gold spot over the stored
snapshots). AISC is not available on these stations, so a margin-based sensitivity cannot be computed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONSENSUS_FILE = Path(__file__).with_name("gold_consensus.json")
SCENARIOS = (3000.0, 3500.0, 4000.0)
MIN_DAYS = 20                      # fewer common trading days than this: no beta at all
SOLID_DAYS = 60                    # fewer than this, or |t| < SOLID_T: shown "仅供参考"
SOLID_T = 2.0


def load_consensus(path: Path = CONSENSUS_FILE) -> dict | None:
    """{"year", "median", "asof", "spot", "spot_asof", "source"} or None when missing or malformed."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if float(d["median"]) > 0 and int(d["year"]) else None
    except Exception:
        return None


def base_gold(fallback: float) -> float:
    """Valuation base: next year's consensus median."""
    c = load_consensus()
    return float(c["median"]) if c else float(fallback)


def spot_gold(fallback: float) -> float:
    """Scenario base: the latest exported gold spot."""
    c = load_consensus() or {}
    try:
        return float(c["spot"]) if float(c["spot"]) > 0 else float(fallback)
    except (KeyError, TypeError, ValueError):
        return float(fallback)


def gold_beta(stock: pd.Series, gold: pd.Series) -> tuple[float, int, float]:
    """(beta, n, t) of the stock's daily log returns on gold's over the dates both have; beta nan if n < MIN_DAYS."""
    s = pd.Series(stock.values, index=pd.to_datetime(stock.index)).astype(float)
    g = pd.Series(gold.values, index=pd.to_datetime(gold.index)).astype(float)
    s, g = s[s > 0].sort_index(), g[g > 0].sort_index()
    both = pd.concat([np.log(s).diff(), np.log(g).diff()], axis=1, join="inner").dropna()
    n = len(both)
    y, x = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    sxx = ((x - x.mean()) ** 2).sum() if n else 0.0
    if n < MIN_DAYS or sxx == 0:
        return float("nan"), n, float("nan")
    b = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    resid = y - y.mean() - b * (x - x.mean())
    se = np.sqrt((resid ** 2).sum() / (n - 2) / sxx)
    return float(b), n, float(b / se) if se > 0 else float("inf")


def is_solid(n: int, t: float) -> bool:
    return n >= SOLID_DAYS and np.isfinite(t) and abs(t) >= SOLID_T


def scenario_table(beta: float, spot: float, consensus: float | None, scenarios=SCENARIOS) -> pd.DataFrame:
    """Five columns - spot, the three scenarios, the consensus - and three rows: gold price, gold vs spot,
    implied share-price move. Rows hold numbers; formatting is the page's job."""
    cols = [("现货", spot)] + [(f"US${p:,.0f}", p) for p in scenarios]
    if consensus:
        cols.append(("一致预期", float(consensus)))
    price = {k: v for k, v in cols}
    vs = {k: v / spot - 1 for k, v in cols}
    move = {k: ((v / spot) ** beta - 1) if np.isfinite(beta) else float("nan") for k, v in cols}
    return pd.DataFrame([{"": "金价 US$/oz", **price}, {"": "相对现货", **vs}, {"": "按历史弹性的股价变动", **move}])
