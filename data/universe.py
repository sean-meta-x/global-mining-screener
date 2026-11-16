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


# Decided by Sean on 2026-11-16 (2026-Q3 review, proposal C). Applied here, at load time, so a
# rebuild of universes/*.json by scripts/build_universe.py + merge_curated.py cannot bring them back.
NOT_MINERS = {
    "LODE": "Comstock Inc. — left mining for fuels and metal recycling",
    "POW.TO": "Power Corporation of Canada — a financial holding company",
    "TOU.TO": "Tourmaline Oil — a natural-gas producer",
}
# ticker -> (commodity, stage) by what the company actually does
RELABEL = {
    "ELBM": ("Cobalt", "Developer"),      # Electra Battery Materials: cobalt refinery (Ontario) + Iron Creek Co-Cu project
    "ELBM.V": ("Cobalt", "Developer"),
}

# Screen-sourced rows labelled "PGM" only because Yahoo files them under "Other Precious Metals
# & Mining" (Q3 review, proposal C; Sean 2026-11-16: fix what the name settles, list the rest —
# docs/pgm_label_review_2026-11-16.md). The stage stays the screen's unless the name says royalty.
_ROYALTY = "Royalty/Streaming"
PGM_BY_NAME = {
    **dict.fromkeys(["ACDX.CN", "CGNT.V", "PERU.V"], ("Copper", None)),
    **dict.fromkeys(["ATHA.CN", "ESAU.CN", "GLDS.CN", "MIRA.CN", "PAU.CN", "RGN.CN", "RLGC.CN", "TGLD.CN"], ("Gold", None)),
    "ICG.CN": ("Silver/Gold", None),
    **dict.fromkeys(["DIAM.TO", "LUC.TO", "MPVD.TO", "KDR.L"], ("Diamonds", None)),
    **dict.fromkeys(["GEM.L", "GML.JO"], ("Gemstones", None)),
    "NEXU.CN": ("Uranium", None),
    **dict.fromkeys(["ELE.TO", "ELE", "EMPR.V", "STRR.V", "SUM.V", "VMET.TO", "VMET", "VOXR.TO", "VOXR", "MTA"],
                    ("Diversified", _ROYALTY)),
}


def _row(c: dict) -> tuple:
    commodity, stage = RELABEL.get(c["ticker"], (screen_commodity(c["name"], c["commodity"], c.get("src", "screen")), c["stage"]))
    if commodity == "PGM" and c.get("src", "screen") == "screen" and c["ticker"] in PGM_BY_NAME:
        commodity, new_stage = PGM_BY_NAME[c["ticker"]]
        stage = new_stage or stage
    return c["ticker"], c["name"], commodity, stage


# Same tuple shape as the original module: (ticker, name, commodity, stage)
MINING_UNIVERSE = [_row(c) for c in _PAYLOAD["companies"] if c["ticker"] not in NOT_MINERS]

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
