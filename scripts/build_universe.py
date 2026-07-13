"""
Build the full mining-stock universe for a market from the Yahoo Finance screener.

Queries every mining industry under the Basic Materials sector for a region and
writes universes/{market}.json with (ticker, name, commodity, stage).

Stage is provisional (market-cap tiers); the daily refresh refines it using
revenue (producer vs explorer) — see scheduler/jobs.refine_stages().

Usage:  python scripts/build_universe.py ca us uk za id hk cn
"""
import json
import sys
import time
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.yahoo_client import _session, _CRUMB  # noqa: E402  (reuse crumb auth)
import data.yahoo_client as yc                   # noqa: E402

# ── Market registry ────────────────────────────────────────────────────────────
# suffixes: keep only listings on the market's own exchanges
# fx_to_usd: rough factor to convert local market cap to USD for stage tiers
MARKETS = {
    "ca": {"region": "ca", "suffixes": (".TO", ".V", ".CN"), "fx_to_usd": 0.73},
    "us": {"region": "us", "suffixes": ("",),                "fx_to_usd": 1.00},
    "uk": {"region": "gb", "suffixes": (".L",),              "fx_to_usd": 1.27},
    "za": {"region": "za", "suffixes": (".JO",),             "fx_to_usd": 0.055},
    "id": {"region": "id", "suffixes": (".JK",),             "fx_to_usd": 0.000062},
    "hk": {"region": "hk", "suffixes": (".HK",),             "fx_to_usd": 0.128},
    "cn": {"region": "cn", "suffixes": (".SS", ".SZ"),       "fx_to_usd": 0.14},
}

# Yahoo Basic Materials industries that are mining → our commodity label
INDUSTRY_COMMODITY = {
    "Gold":                              "Gold",
    "Silver":                            "Silver",
    "Copper":                            "Copper",
    "Uranium":                           "Uranium",
    "Coking Coal":                       "Coal",
    "Thermal Coal":                      "Coal",
    "Aluminum":                          "Aluminum",
    "Other Precious Metals & Mining":    "PGM",
    "Other Industrial Metals & Mining":  "Diversified",
}

# Excluded exchange codes (OTC / pink sheets when region=us)
EXCLUDED_EXCHANGES = {"PNK", "OQB", "OQX", "OEM", "OGM", "YHD"}

SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener"


def _screen_industry(region: str, industry: str) -> list[dict]:
    """Return all screener quotes for one region+industry (paginated)."""
    s = _session()
    quotes: list[dict] = []
    offset, total = 0, None
    while total is None or offset < total:
        body = {
            "size": 250,
            "offset": offset,
            "sortField": "intradaymarketcap",
            "sortType": "DESC",
            "quoteType": "EQUITY",
            "query": {
                "operator": "AND",
                "operands": [
                    {"operator": "eq", "operands": ["region", region]},
                    {"operator": "eq", "operands": ["industry", industry]},
                ],
            },
            "userId": "",
            "userIdType": "guid",
        }
        r = s.post(
            f"{SCREENER_URL}?crumb={yc._CRUMB}&lang=en-US&region=US",
            json=body, timeout=30,
        )
        if r.status_code == 429:
            time.sleep(10)
            continue
        r.raise_for_status()
        result = r.json().get("finance", {}).get("result") or [{}]
        batch = result[0].get("quotes") or []
        total = result[0].get("total", 0)
        quotes.extend(batch)
        offset += 250
        if not batch:
            break
        time.sleep(1.0)
    return quotes


def _stage_from_mcap(mcap_local: float | None, fx: float) -> str:
    """Provisional stage from market cap (USD). Refined later with revenue."""
    if not mcap_local:
        return "Explorer"
    usd = mcap_local * fx
    if usd >= 5e9:   return "Major Producer"
    if usd >= 5e8:   return "Producer"
    if usd >= 1e8:   return "Developer"
    return "Explorer"


def build_market(market: str) -> None:
    cfg = MARKETS[market]
    seen: dict[str, dict] = {}
    for industry, commodity in INDUSTRY_COMMODITY.items():
        quotes = _screen_industry(cfg["region"], industry)
        kept = 0
        for q in quotes:
            sym = q.get("symbol", "")
            if not sym or sym in seen:
                continue
            if q.get("exchange") in EXCLUDED_EXCHANGES:
                continue
            if cfg["suffixes"] != ("",):
                if not any(sym.endswith(sfx) for sfx in cfg["suffixes"]):
                    continue
            elif "." in sym:      # us: no suffix means primary US listing
                continue
            name = (q.get("shortName") or q.get("longName") or sym).strip()
            seen[sym] = {
                "ticker":    sym,
                "name":      name[:48],
                "commodity": commodity,
                "stage":     _stage_from_mcap(q.get("marketCap"), cfg["fx_to_usd"]),
            }
            kept += 1
        print(f"  [{market}] {industry:<36} {kept:>4} kept / {len(quotes)} returned")

    out = {
        "market":    market,
        "built":     datetime.date.today().isoformat(),
        "n":         len(seen),
        "companies": sorted(seen.values(), key=lambda c: c["ticker"]),
    }
    out_path = Path(__file__).resolve().parent.parent / "universes" / f"{market}.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[{market}] wrote {len(seen)} companies -> {out_path.name}")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(MARKETS)
    for m in targets:
        if m not in MARKETS:
            print(f"unknown market: {m}")
            continue
        print(f"Building {m} ...")
        build_market(m)
