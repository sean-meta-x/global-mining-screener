"""data/ic_gate.py — weekly IC of the trading score and the gate on decision cards (Sean 2026-11-18)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import data.ic_gate as g
from data.scorer import valuation_score
from config import SCORE_WEIGHTS


def snaps(n_days=60, n_names=40, signal=1.0, seed=0, start="2026-11-19"):
    """Daily snapshots where score_composite predicts the next 20-day return with strength `signal`."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days).strftime("%Y-%m-%d")
    score = rng.uniform(0, 100, n_names)
    drift = (score - 50) / 50 * 0.01 * signal                     # per day
    px = 10 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, (n_days, n_names)), axis=0))
    rows = []
    for i, d in enumerate(dates):
        for j in range(n_names):
            rows.append(dict(ticker=f"T{j}", snap_date=d, price=px[i, j], market_cap=1e9, avg_turnover=1e6,
                             grade="🔵 Buy" if score[j] >= 70 else "🟡 Watch",
                             **{f: score[j] for f in g.FACTORS}))
    return pd.DataFrame(rows)


def test_weights_are_valuation_and_health_only():
    assert SCORE_WEIGHTS["valuation"] == SCORE_WEIGHTS["health"] == 0.5
    assert all(SCORE_WEIGHTS[k] == 0 for k in ("momentum", "mining", "commodity", "stage"))


def test_analyst_upside_no_longer_moves_the_valuation_score():
    peers = pd.DataFrame({"priceToBook": [1.0, 2.0, 3.0], "enterpriseValue": [1, 1, 1],
                          "ebitda": [np.nan] * 3, "marketCap": [1, 1, 1], "operatingCashflow": [np.nan] * 3})
    low = pd.Series({"priceToBook": 1.2, "analyst_count": 5, "analyst_upside": 0.0})
    high = pd.Series({"priceToBook": 1.2, "analyst_count": 5, "analyst_upside": 80.0})
    assert valuation_score(low, peers) == valuation_score(high, peers)


def test_a_score_that_predicts_returns_passes_once_there_are_enough_windows():
    recs = []
    for m in ("ca", "us"):
        recs += g.windows(snaps(seed=1 if m == "ca" else 2), 1.0, m)
    v = g.verdict(recs)
    assert v["windows"] == 4 and not v["validated"] and "样本不够" in v["reason"]
    recs = []
    for k, m in enumerate(("ca", "us", "uk")):
        recs += g.windows(snaps(n_days=70, seed=k), 1.0, m)
    v = g.verdict(recs)
    assert v["windows"] >= g.MIN_WINDOWS and v["starts"] >= g.MIN_STARTS
    assert v["validated"] and v["ic"] > 0.3 and v["buy_excess"] > 0


def test_a_score_with_no_signal_is_blocked():
    recs = []
    for k, m in enumerate(("ca", "us", "uk")):
        recs += g.windows(snaps(n_days=70, signal=0.0, seed=10 + k), 1.0, m)
    v = g.verdict(recs)
    assert v["windows"] >= g.MIN_WINDOWS and not v["validated"]


def test_windows_before_the_new_score_do_not_count():
    recs = g.windows(snaps(n_days=70, start="2026-07-13"), 1.0, "ca")
    assert recs and g.verdict(recs)["windows"] == 0


def test_illiquid_stale_and_unlisted_names_are_left_out():
    s = snaps(n_days=25)
    s.loc[s.ticker == "T0", "market_cap"] = 1e6
    s.loc[s.ticker == "T1", "grade"] = "⏸️ Stale price"
    s.loc[s.ticker == "T2", "grade"] = "⚪ 副上市"
    (r,) = g.windows(s, 1.0, "ca")
    assert r["n"] == 37


def test_non_overlapping_windows_start_after_the_previous_end():
    recs = g.windows(snaps(n_days=70), 1.0, "ca")
    picked = g.non_overlapping(recs)
    assert len(recs) == 10 and len(picked) == 3
    assert all(b["d0"] >= a["d1"] for a, b in zip(picked, picked[1:]))
