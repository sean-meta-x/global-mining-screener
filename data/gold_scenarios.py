"""Gold base price and scenario sensitivity for gold miners.

Base price: the analyst consensus median for next year's average gold price, read from
data/gold_consensus.json — one derived display value (year, median, as-of date), never a series.
The file is written from the fund's local Bloomberg store (consensus METFGD <YY> Index); this repo
holds only that single number. Without the file the base falls back to the gold spot in config.

Scenarios: US$3,000 / 3,500 / 4,000 against the base. They are what-ifs, not forecasts. A miner's
sensitivity is its historical elasticity to gold (beta of daily log returns on the gold spot over the
stored snapshots); the scenario move is (scenario / base) ** beta - 1, the same
elasticity applied in log terms so large moves are not overstated. No cost data is needed — AISC is
not available on these stations, so a margin-based sensitivity cannot be computed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONSENSUS_FILE = Path(__file__).with_name("gold_consensus.json")
SCENARIOS = (3000.0, 3500.0, 4000.0)
MIN_DAYS = 60                      # fewer common trading days than this: no beta


def load_consensus(path: Path = CONSENSUS_FILE) -> dict | None:
    """{"year", "median", "asof", "source"} or None when the file is missing or malformed."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if float(d["median"]) > 0 and int(d["year"]) else None
    except Exception:
        return None


def base_gold(fallback: float) -> float:
    c = load_consensus()
    return float(c["median"]) if c else float(fallback)


def gold_beta(stock: pd.Series, gold: pd.Series) -> tuple[float, int]:
    """Beta of the stock's daily log returns on gold's, over the dates both have. (nan, n) if n < MIN_DAYS."""
    s = pd.Series(stock.values, index=pd.to_datetime(stock.index)).astype(float)
    g = pd.Series(gold.values, index=pd.to_datetime(gold.index)).astype(float)
    s, g = s[s > 0].sort_index(), g[g > 0].sort_index()
    both = pd.concat([np.log(s).diff(), np.log(g).diff()], axis=1, join="inner").dropna()
    n = len(both)
    if n < MIN_DAYS or both.iloc[:, 1].var() == 0:
        return float("nan"), n
    b = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
    return float(b), n


def scenario_table(beta: float, base: float, scenarios=SCENARIOS) -> pd.DataFrame:
    rows = []
    for p in scenarios:
        chg = p / base - 1
        rows.append({"金价情景": f"US${p:,.0f}", "相对基准": chg,
                     "按历史弹性的股价变动": (p / base) ** beta - 1 if np.isfinite(beta) else float("nan")})
    return pd.DataFrame(rows)
