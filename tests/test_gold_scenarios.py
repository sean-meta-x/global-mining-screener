"""data/gold_scenarios.py — consensus base price and scenario sensitivity."""
import json

import numpy as np
import pandas as pd

from data import gold_scenarios as gs


def test_the_base_is_the_consensus_median_and_falls_back_to_spot(tmp_path, monkeypatch):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"year": 2027, "median": 4500.0, "asof": "2026-11-16", "source": "x"}), encoding="utf-8")
    monkeypatch.setattr(gs, "CONSENSUS_FILE", f)
    assert gs.load_consensus(f)["year"] == 2027
    assert gs.base_gold(4000.0) == 4500.0 if gs.load_consensus(f) else False
    monkeypatch.setattr(gs, "load_consensus", lambda path=None: None)
    assert gs.base_gold(4000.0) == 4000.0


def test_the_committed_consensus_file_is_one_number_not_a_series():
    d = json.loads(gs.CONSENSUS_FILE.read_text(encoding="utf-8"))
    assert set(d) == {"year", "median", "asof", "source"}
    assert isinstance(d["median"], float)


def test_beta_recovers_a_known_elasticity():
    idx = pd.bdate_range("2026-01-01", periods=120)
    rng = np.random.default_rng(0)
    g = pd.Series(4000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    s = pd.Series(10 * np.exp(1.8 * np.log(g / 4000)), index=idx)          # stock = gold ^ 1.8
    beta, n = gs.gold_beta(s, g)
    assert n == 119 and abs(beta - 1.8) < 1e-9


def test_no_beta_with_too_little_history():
    idx = pd.bdate_range("2026-01-01", periods=30)
    beta, n = gs.gold_beta(pd.Series(range(1, 31), index=idx), pd.Series(range(100, 130), index=idx))
    assert np.isnan(beta) and n == 29


def test_scenarios_are_relative_to_the_base():
    t = gs.scenario_table(2.0, 4000.0)
    assert list(t["金价情景"]) == ["US$3,000", "US$3,500", "US$4,000"]
    assert np.allclose(t["相对基准"], [-0.25, -0.125, 0.0])
    assert np.allclose(t["按历史弹性的股价变动"], [0.75 ** 2 - 1, 0.875 ** 2 - 1, 0.0])
