"""Under-covered scoring factors are left out and their weight redistributed (2026-11-17)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.scorer import effective_weights, factor_coverage

W = {"valuation": 0.30, "health": 0.20, "momentum": 0.15, "mining": 0.25, "commodity": 0.05, "stage": 0.05}


def test_full_coverage_keeps_the_weights():
    cov = {f: 1.0 for f in W}
    assert effective_weights(W, cov) == (W, [])


def test_an_uncovered_factor_is_dropped_and_the_rest_scaled_to_the_same_total():
    cov = {f: 1.0 for f in W} | {"mining": 0.0}
    w, excluded = effective_weights(W, cov)
    assert excluded == ["mining"] and w["mining"] == 0
    assert abs(sum(w.values()) - sum(W.values())) < 1e-9
    assert abs(w["valuation"] / w["health"] - 0.30 / 0.20) < 1e-9


def test_nothing_is_dropped_when_nothing_would_be_left():
    cov = {f: 0.0 for f in W}
    assert effective_weights(W, cov) == (W, [])


def test_factor_coverage_counts_rows_with_any_input():
    df = pd.DataFrame({"spg_p_nav": [np.nan, np.nan, 1.0, np.nan], "snl_p_insitu": [np.nan, 2.0, np.nan, np.nan],
                       "rsi": [30, 40, 50, 60], "commodity": ["Gold"] * 4, "stage": ["Producer"] * 4})
    cov = factor_coverage(df)
    assert cov["mining"] == 0.5 and cov["momentum"] == 1.0 and cov["health"] == 0.0


# ── grade calibration (2026-11-17: leaving the mining factor out took AU's Strong Buy from 11 to 42) ──
from data.scorer import calibrated_cutoffs, grade_label  # noqa: E402


def test_calibrated_cutoffs_keep_each_grades_count():
    reference = pd.Series([80, 76, 70, 62, 50, 40, 20])          # 2 Strong Buy, 2 Buy, 1 Watch, 1 Neutral, 1 Avoid
    shifted = pd.Series([90, 88, 86, 80, 70, 60, 45])             # same order, everything higher
    cuts = calibrated_cutoffs(reference, shifted)
    grades = [grade_label(s, cuts) for s in shifted]
    assert [g.split(" ", 1)[1] for g in grades] == ["Strong Buy", "Strong Buy", "Buy", "Buy", "Watch", "Neutral", "Avoid"]
    assert [grade_label(s) for s in shifted].count("🟢 Strong Buy") == 4   # fixed cutoffs would inflate


def test_a_grade_nobody_reached_stays_empty_and_fixed_cutoffs_are_the_default():
    cuts = calibrated_cutoffs(pd.Series([70, 50]), pd.Series([95, 60]))
    assert cuts[0] == float("inf") and grade_label(95, cuts) == "🔵 Buy"
    assert grade_label(75) == "🟢 Strong Buy" and grade_label(29.9) == "🔴 Avoid"


# ── cutoffs from tradable names only (Sean 2026-11-17) ──
from data.scorer import tradable_mask  # noqa: E402


def test_tradable_mask_needs_both_size_and_trading():
    df = pd.DataFrame({"market_cap": [50e6, 50e6, 5e6, None], "avg_turnover": [100e3, 1e3, 100e3, 100e3]})
    assert tradable_mask(df, fx=1.0).tolist() == [True, False, False, False]      # missing data = not tradable


def test_micro_caps_do_not_move_the_cutoffs():
    tradable = pd.Series([True, True, True, True, False, False])
    reference = pd.Series([80, 62, 50, 40, 90, 90])            # two hidden micro caps would be Strong Buy
    shifted = pd.Series([86, 70, 58, 48, 99, 98])
    cuts = calibrated_cutoffs(reference[tradable], shifted[tradable])
    assert [grade_label(s, cuts).split(" ", 1)[1] for s in shifted[tradable]] == ["Strong Buy", "Buy", "Watch", "Neutral"]
    assert grade_label(99, cuts).endswith("Strong Buy")       # hidden names are still graded on the same scale


# ── investor review v2 (2026-11-18) ──
from data.scorer import apply_grade_rules, NOT_GRADED, PHYSICAL_GRADE, SECONDARY_GRADE  # noqa: E402


def test_a_micro_cap_tops_out_at_buy_and_is_flagged():
    df = pd.DataFrame({"market_cap": [30e6, 300e6], "grade": ["🟢 Strong Buy", "🟢 Strong Buy"]})
    out = apply_grade_rules(df, ["SMALL", "BIG"], fx=1.0)
    assert out["grade"].tolist() == ["🔵 Buy", "🟢 Strong Buy"]
    assert out["micro_cap"].tolist() == [True, False]


def test_not_graded_names_get_their_label_instead_of_a_grade(monkeypatch):
    import data.scorer as sc
    monkeypatch.setattr(sc, "NOT_GRADED", {"YCA": ("实物持有", None), "AAUC": ("副上市 · 主上市 AAUC.TO（加拿大站）", "https://x")})
    df = pd.DataFrame({"market_cap": [1e9, 1e9, 1e9], "grade": ["🔵 Buy", "🟢 Strong Buy", "🟡 Watch"]})
    out = sc.apply_grade_rules(df, ["YCA", "AAUC", "OTHER"], fx=1.0)
    assert out["grade"].tolist() == [PHYSICAL_GRADE, SECONDARY_GRADE, "🟡 Watch"]
