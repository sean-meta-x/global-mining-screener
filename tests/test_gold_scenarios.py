"""data/gold_scenarios.py — consensus base, spot-based scenarios, beta and its reliability flag."""
import json

import numpy as np
import pandas as pd

from data import gold_scenarios as gs


def test_no_bloomberg_consensus_in_this_public_repo():
    root = gs.CONSENSUS_FILE.parent.parent
    assert "data/gold_consensus.json" in (root / ".gitignore").read_text(encoding="utf-8")
    assert not gs.CONSENSUS_FILE.exists() and gs.base_gold(3900.0) == 3900.0     # no file: the spot is the base


def test_spot_and_consensus_bases(tmp_path, monkeypatch):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"year": 2027, "median": 4500.0, "asof": "2026-11-16", "spot": 3854.0,
                             "spot_asof": "2026-11-17", "source": "x"}), encoding="utf-8")
    monkeypatch.setattr(gs, "CONSENSUS_FILE", f)
    load = gs.load_consensus
    monkeypatch.setattr(gs, "load_consensus", lambda path=f: load(path))
    assert gs.base_gold(1.0) == 4500.0 and gs.spot_gold(1.0) == 3854.0
    monkeypatch.setattr(gs, "load_consensus", lambda path=None: None)
    assert gs.base_gold(4000.0) == 4000.0 and gs.spot_gold(3900.0) == 3900.0


def test_beta_recovers_a_known_elasticity_and_is_solid():
    idx = pd.bdate_range("2026-01-01", periods=120)
    rng = np.random.default_rng(0)
    g = pd.Series(4000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    s = pd.Series(10 * np.exp(1.8 * np.log(g / 4000) + rng.normal(0, 0.005, len(idx))), index=idx)
    beta, n, t = gs.gold_beta(s, g)
    assert n == 119 and abs(beta - 1.8) < 0.1 and gs.is_solid(n, t)


def test_short_or_noisy_history_is_only_for_reference():
    idx = pd.bdate_range("2026-01-01", periods=40)
    rng = np.random.default_rng(1)
    g = pd.Series(4000 * np.exp(np.cumsum(rng.normal(0, 0.01, 40))), index=idx)
    s = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.03, 40))), index=idx)      # unrelated to gold
    beta, n, t = gs.gold_beta(s, g)
    assert np.isfinite(beta) and n == 39 and not gs.is_solid(n, t)
    beta, n, t = gs.gold_beta(s.iloc[:10], g.iloc[:10])
    assert np.isnan(beta) and n == 9


def test_scenarios_are_measured_from_spot_with_the_consensus_as_a_column():
    t = gs.scenario_table(2.0, 4000.0, 4500.0)
    assert list(t.columns) == ["", "现货", "US$3,000", "US$3,500", "US$4,000", "一致预期"]
    vs, move = t.iloc[1, 1:].astype(float), t.iloc[2, 1:].astype(float)
    assert np.allclose(vs, [0.0, -0.25, -0.125, 0.0, 0.125])
    assert np.allclose(move, [0.0, 0.75 ** 2 - 1, 0.875 ** 2 - 1, 0.0, 1.125 ** 2 - 1])
    assert "一致预期" not in gs.scenario_table(2.0, 4000.0, None).columns
