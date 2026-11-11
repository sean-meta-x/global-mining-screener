"""
Mining stock universe — loaded from universes/{MARKET}.json.

The JSON is produced by scripts/build_universe.py (full Yahoo-screener sweep of
every Basic Materials mining industry) merged with the hand-curated lists from
the original per-market screeners via scripts/merge_curated.py.

Rows carry src="curated" (trusted stage/commodity) or src="screen"
(market-cap-heuristic stage — refined by scheduler.jobs.refine_stages()).
"""
import json
import re

from config import UNIVERSE_JSON

with open(UNIVERSE_JSON, encoding="utf-8") as _f:
    _PAYLOAD = json.load(_f)

_SAYS_SILVER = re.compile(r"silver|argent|plata", re.I)
_SAYS_GOLD = re.compile(r"\bgold\b", re.I)
_SAYS_PGM = re.compile(r"platin|pallad|\bpgm", re.I)


def screen_commodity(name: str, commodity: str, src: str) -> str:
    """The build script maps Yahoo's industry "Other Precious Metals & Mining" to PGM, so
    Avino Silver & Gold Mines (ASM.TO) sat in the PGM peer group (2026-Q3 review,
    proposal C). Where a screen-sourced name says which metal it mines, believe the name.
    Curated rows, and names that say nothing, are left as they are."""
    if src != "screen" or commodity != "PGM" or _SAYS_PGM.search(name):
        return commodity
    if _SAYS_SILVER.search(name):
        return "Silver"
    if _SAYS_GOLD.search(name):
        return "Gold"
    return commodity


# Same tuple shape as the original module: (ticker, name, commodity, stage)
MINING_UNIVERSE = [
    (c["ticker"], c["name"], screen_commodity(c["name"], c["commodity"], c.get("src", "screen")), c["stage"])
    for c in _PAYLOAD["companies"]
]

# ticker -> "curated" | "screen"
UNIVERSE_SRC = {c["ticker"]: c.get("src", "screen") for c in _PAYLOAD["companies"]}

ALL_COMMODITIES = sorted({row[2].split("/")[0] for row in MINING_UNIVERSE})
ALL_STAGES = [
    "Major Producer", "Mid-tier Producer", "Producer",
    "Developer", "Explorer", "Royalty/Streaming",
]


def get_tickers() -> list[str]:
    return [row[0] for row in MINING_UNIVERSE]


def get_ticker_meta() -> dict[str, dict]:
    return {
        row[0]: {"name": row[1], "commodity": row[2], "stage": row[3]}
        for row in MINING_UNIVERSE
    }
