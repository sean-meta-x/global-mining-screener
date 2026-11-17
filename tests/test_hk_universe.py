"""HK universe review (2026-11-17): checked against etnet's Hang Seng industry lists and each
company's own business description. Sean could not find 灵宝黄金 / 香港黄金产业 / 集海黄金."""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _hk():
    os.environ["MARKET"] = "hk"
    import config
    importlib.reload(config)
    import data.universe as u
    return importlib.reload(u)


def test_non_miners_are_out_and_verified_miners_are_in():
    u = _hk()
    meta = u.get_ticker_meta()
    for t in ("0061.HK", "0098.HK", "1051.HK", "8095.HK", "8250.HK", "1733.HK"):
        assert t not in meta
    for t in ("2546.HK", "0893.HK", "1370.HK", "0159.HK", "0274.HK"):
        assert t in meta
    assert meta["2546.HK"]["commodity"] == "Zinc/Lead"


def test_the_three_names_sean_looked_for_are_findable_in_chinese():
    meta = _hk().get_ticker_meta()
    names = " ".join(v["name"] for v in meta.values())
    for zh in ("灵宝黄金", "靈寶黃金", "香港黄金产业", "集海黄金"):
        assert zh in names
    assert meta["2623.HK"]["commodity"] == "Iron Ore"          # Hong Kong Gold Industry mines iron and ilmenite
    assert "-500" not in meta["2623.HK"]["name"]


def test_other_markets_are_untouched():
    os.environ["MARKET"] = "ca"
    import config
    importlib.reload(config)
    import data.universe as u
    u = importlib.reload(u)
    assert not any(t.endswith(".HK") for t in u.get_tickers())
    os.environ["MARKET"] = "hk"
