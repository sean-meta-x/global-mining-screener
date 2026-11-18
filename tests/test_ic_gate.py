"""data/ic_gate.py v2 — group IC net of metal beta, delisted names kept, >= 12 starts, Q5 vs Q1 (2026-11-18)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import data.ic_gate as g
from data.scorer import valuation_score
from config import SCORE_WEIGHTS


def snaps(n_days=60, n_names=40, signal=1.0, seed=0, start="2026-11-19", beta_score=False, metal_drift=0.0):
    """Daily snapshots (weekends included) scored valuation 50 / health 50, half Gold, half Copper.
    `signal`: score predicts idiosyncratic return. `beta_score`: score = the name's metal beta instead."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days).strftime("%Y-%m-%d")
    metal_r = rng.normal(metal_drift, 0.01, n_days)
    betas = rng.uniform(0.5, 2.5, n_names)
    score = (betas - 0.5) / 2 * 100 if beta_score else rng.uniform(0, 100, n_names)
    idio = (score - 50) / 50 * 0.01 * signal * (0 if beta_score else 1)
    r = betas * metal_r[:, None] + idio + rng.normal(0, 0.01, (n_days, n_names))
    px = 10 * np.exp(np.cumsum(r, axis=0))
    metal = 100 * np.exp(np.cumsum(metal_r))
    rows = []
    for i, d in enumerate(dates):
        for j in range(n_names):
            rows.append(dict(ticker=f"T{j}", snap_date=d, price=px[i, j], market_cap=1e9, avg_turnover=1e6,
                             grade="🔵 Buy" if score[j] >= 70 else "🟡 Watch",
                             commodity="Gold" if j % 2 == 0 else "Copper",
                             **{f: score[j] for f in g.FACTORS}))
    m = {"Gold": pd.Series(metal, index=dates), "Copper": pd.Series(metal, index=dates)}
    return pd.DataFrame(rows), m


def test_weights_are_valuation_and_health_only():
    assert SCORE_WEIGHTS["valuation"] == SCORE_WEIGHTS["health"] == 0.5
    assert all(SCORE_WEIGHTS[k] == 0 for k in ("momentum", "mining", "commodity", "stage"))


def test_analyst_upside_no_longer_moves_the_valuation_score():
    peers = pd.DataFrame({"priceToBook": [1.0, 2.0, 3.0], "enterpriseValue": [1, 1, 1],
                          "ebitda": [np.nan] * 3, "marketCap": [1, 1, 1], "operatingCashflow": [np.nan] * 3})
    low = pd.Series({"priceToBook": 1.2, "analyst_count": 5, "analyst_upside": 0.0})
    high = pd.Series({"priceToBook": 1.2, "analyst_count": 5, "analyst_upside": 80.0})
    assert valuation_score(low, peers) == valuation_score(high, peers)


def _recs(n_days, markets=("ca", "us", "uk"), **kw):
    out = []
    for k, m in enumerate(markets):
        s, metal = snaps(n_days=n_days, seed=k, **kw)
        out += g.windows(s, 1.0, m, SCORE_WEIGHTS, metal)
    return out


def test_a_predictive_score_passes_only_with_twelve_start_dates():
    v = g.verdict(_recs(200))
    assert v["starts"] < g.MIN_STARTS and not v["validated"] and "样本不够" in v["reason"]
    v = g.verdict(_recs(370))
    assert v["starts"] >= g.MIN_STARTS and v["validated"] and v["ic"] > 0.2 and v["q5q1"] > 0


def test_a_score_with_no_signal_is_blocked():
    v = g.verdict(_recs(370, signal=0.0))
    assert v["starts"] >= g.MIN_STARTS and not v["validated"]


def test_a_score_that_only_picks_metal_beta_has_no_ic_after_the_beta_is_taken_out():
    adj = g.verdict(_recs(370, beta_score=True, metal_drift=0.004))
    raw = []                                            # same data, no metal prices: no beta adjustment
    for k, m in enumerate(("ca", "us", "uk")):
        s, _ = snaps(n_days=370, seed=k, beta_score=True, metal_drift=0.004)
        raw += g.windows(s, 1.0, m, SCORE_WEIGHTS, {})
    raw = g.verdict(raw)
    # most of it goes; what is left is beta estimation error (60 snapshots) under an extreme +0.4 %/day metal
    assert raw["ic"] > 0.5 and adj["ic"] < raw["ic"] / 4


def test_a_reversed_score_fails_on_q5_vs_q1():
    v = g.verdict(_recs(370, signal=-1.0))
    assert not v["validated"] and v["q5q1"] < 0


def test_snapshots_from_another_scorer_or_before_the_date_do_not_count():
    s, metal = snaps(n_days=120, start="2026-07-13")
    assert g.verdict(g.windows(s, 1.0, "ca", SCORE_WEIGHTS, metal))["windows"] == 0
    s, metal = snaps(n_days=120)
    s["score_composite"] = s["score_composite"] * 0.9 + 3
    recs = g.windows(s, 1.0, "ca", SCORE_WEIGHTS, metal)
    assert recs and not any(r["live"] for r in recs)


def test_a_name_that_leaves_inside_the_window_stays_at_its_last_price():
    s, metal = snaps(n_days=40)
    s = s[~((s.ticker == "T3") & (s.snap_date > "2026-11-25"))]          # delisted / acquired on day 7
    (r,) = g.windows(s, 1.0, "ca", SCORE_WEIGHTS, metal)[:1]
    assert r["n"] == 40 and r["left_in_window"] == 1


def test_illiquid_stale_and_unlisted_names_are_left_out():
    s, metal = snaps(n_days=30)
    s.loc[s.ticker == "T0", "market_cap"] = 1e6
    s.loc[s.ticker == "T1", "grade"] = "⏸️ Stale price"
    s.loc[s.ticker == "T2", "grade"] = "⚪ 副上市"
    (r,) = g.windows(s, 1.0, "ca", SCORE_WEIGHTS, metal)[:1]
    assert r["n"] == 37


def test_windows_are_four_calendar_weeks_and_one_week_apart():
    s, metal = snaps(n_days=60)
    recs = g.windows(s, 1.0, "ca", SCORE_WEIGHTS, metal)
    assert [r["d0"] for r in recs[:2]] == ["2026-11-19", "2026-11-26"] and recs[0]["d1"] == "2026-12-17"
    assert all(b["d0"] >= a["d1"] for a, b in zip(g.non_overlapping(recs), g.non_overlapping(recs)[1:]))


def test_newey_west_t_is_smaller_than_naive_t_on_overlapping_series():
    rng = np.random.default_rng(3)
    e = rng.normal(0.02, 0.05, 80)
    x = pd.Series(e).rolling(4).mean().dropna()                         # overlapping 4-period windows
    naive = x.mean() / x.std() * np.sqrt(len(x))
    assert g.newey_west_t(x) < naive
