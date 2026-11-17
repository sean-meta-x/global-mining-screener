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
