"""Tests for quote-currency normalization in data.fetcher.

LSE quotes in pence (GBp) and JSE in ZAR cents (ZAc); Yahoo's book value,
market cap and enterprise value are in the major unit, and financial-statement
fields come in financialCurrency. Sample values mirror real quoteSummary
payloads captured 2026-07-14.
"""
import pytest

from data import fetcher
from data.fetcher import _normalize_currency


@pytest.fixture(autouse=True)
def _clear_fx_cache():
    fetcher._FX_CACHE.clear()
    yield
    fetcher._FX_CACHE.clear()


def _fake_fx(rates: dict[str, float]):
    """Return a get_info stub serving FX pairs like 'USDZAR=X'."""
    def _get_info(symbol: str) -> dict:
        pair = symbol.removesuffix("=X")
        if pair in rates:
            return {"regularMarketPrice": rates[pair]}
        return {}
    return _get_info


def test_gbp_pence_quote_normalized(monkeypatch):
    # GLEN.L-style: pence quote, USD financials
    monkeypatch.setattr(fetcher, "get_info", _fake_fx({"USDGBP": 0.79}))
    info = {
        "currency": "GBp",
        "financialCurrency": "USD",
        "regularMarketPrice": 514.5,
        "fiftyTwoWeekHigh": 707.2,
        "fiftyTwoWeekLow": 400.0,
        "targetMeanPrice": 627.4,
        "priceToBook": 206.82591,
        "marketCap": 60_289_241_088,
        "enterpriseValue": 94_159_740_928,
        "ebitda": 9_489_999_872,
    }
    out = _normalize_currency(info)

    assert out["currency"] == "GBP"
    assert out["regularMarketPrice"] == pytest.approx(5.145)
    assert out["fiftyTwoWeekHigh"] == pytest.approx(7.072)
    assert out["targetMeanPrice"] == pytest.approx(6.274)
    # 514.5p / £2.4876 book value → true P/B ≈ 2.07
    assert out["priceToBook"] == pytest.approx(2.0682591)
    # marketCap / enterpriseValue are already major-unit — untouched
    assert out["marketCap"] == 60_289_241_088
    assert out["enterpriseValue"] == 94_159_740_928
    # USD EBITDA converted to GBP so EV/EBITDA is consistent
    assert out["ebitda"] == pytest.approx(9_489_999_872 * 0.79)
    # input dict not mutated
    assert info["priceToBook"] == 206.82591


def test_zac_quote_with_usd_financials(monkeypatch):
    # S32.JO-style: ZAR-cents quote, USD financials
    monkeypatch.setattr(fetcher, "get_info", _fake_fx({"USDZAR": 17.9}))
    info = {
        "currency": "ZAc",
        "financialCurrency": "USD",
        "regularMarketPrice": 4581.0,
        "priceToBook": 136.24803,
        "enterpriseValue": 208_149_938_176,
        "ebitda": 1_192_999_936,
        "totalRevenue": 5_905_999_872,
    }
    out = _normalize_currency(info)

    assert out["currency"] == "ZAR"
    assert out["priceToBook"] == pytest.approx(1.3624803)
    ev_ebitda = out["enterpriseValue"] / out["ebitda"]
    assert ev_ebitda == pytest.approx(208_149_938_176 / (1_192_999_936 * 17.9))
    assert 8 < ev_ebitda < 12   # sane EV/EBITDA, not the broken 174x


def test_matching_currencies_untouched(monkeypatch):
    # IMP.JO-style after subunit fix: ZAR financials need no FX call
    def _boom(symbol):
        raise AssertionError("FX lookup should not happen for matching currencies")
    monkeypatch.setattr(fetcher, "get_info", _boom)
    info = {
        "currency": "ZAc",
        "financialCurrency": "ZAR",
        "regularMarketPrice": 18260.0,
        "priceToBook": 169.00209,
        "ebitda": 21_951_000_576,
    }
    out = _normalize_currency(info)
    assert out["priceToBook"] == pytest.approx(1.6900209)
    assert out["ebitda"] == 21_951_000_576


def test_major_unit_market_passthrough(monkeypatch):
    # AUD-quoted, AUD-reporting: nothing changes
    monkeypatch.setattr(fetcher, "get_info", _fake_fx({}))
    info = {
        "currency": "AUD",
        "financialCurrency": "AUD",
        "regularMarketPrice": 42.5,
        "priceToBook": 3.1,
        "ebitda": 1_000_000,
    }
    assert _normalize_currency(info) == info


def test_fx_failure_leaves_financials_unconverted(monkeypatch):
    monkeypatch.setattr(fetcher, "get_info", _fake_fx({}))   # no rate available
    info = {
        "currency": "ZAc",
        "financialCurrency": "USD",
        "priceToBook": 136.0,
        "ebitda": 1_192_999_936,
    }
    out = _normalize_currency(info)
    assert out["priceToBook"] == pytest.approx(1.36)   # subunit fix still applies
    assert out["ebitda"] == 1_192_999_936              # left as-is, not corrupted


def test_none_and_missing_fields_are_safe(monkeypatch):
    monkeypatch.setattr(fetcher, "get_info", _fake_fx({"USDGBP": 0.79}))
    info = {
        "currency": "GBp",
        "financialCurrency": "USD",
        "regularMarketPrice": None,
        "priceToBook": None,
    }
    out = _normalize_currency(info)
    assert out["regularMarketPrice"] is None
    assert out["priceToBook"] is None


def test_fx_rate_cached_per_pair(monkeypatch):
    calls = []
    def _get_info(symbol):
        calls.append(symbol)
        return {"regularMarketPrice": 17.9}
    monkeypatch.setattr(fetcher, "get_info", _get_info)
    info = {"currency": "ZAc", "financialCurrency": "USD", "ebitda": 100.0}
    _normalize_currency(dict(info))
    _normalize_currency(dict(info))
    assert calls == ["USDZAR=X"]
