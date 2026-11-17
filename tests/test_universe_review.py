"""Universe review 2026-11-17 (data/universe_review.py): scope = metal ores, coal, uranium."""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _meta(market):
    os.environ["MARKET"] = market
    import config
    importlib.reload(config)
    import data.universe as u
    return importlib.reload(u).get_ticker_meta()


def test_each_market_drops_what_the_review_removed_and_gains_what_it_added():
    from data import universe_review as r
    for m in ("ca", "us", "uk", "za", "id", "cn"):
        meta = _meta(m)
        assert not set(r.REMOVE[m]) & set(meta), m
        assert all(t in meta for t, *_ in r.ADD.get(m, [])), m
    os.environ["MARKET"] = "hk"


def test_tags_and_the_names_that_changed_hands():
    uk, us = _meta("uk"), _meta("us")
    assert "实物铀持有" in uk["YCA.L"]["name"] and "二次上市" in uk["KAP.L"]["name"]
    assert "GOLD" not in us and "B" in us            # GOLD is Gold.com now; Barrick trades as B
    assert "USAU" in us and "USGO" in us             # two different companies, both kept
    assert "CNR" in us and "CEIX" not in us and "ARCH" not in us
    os.environ["MARKET"] = "hk"
