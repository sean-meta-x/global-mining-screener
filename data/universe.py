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
    # HK, 2026-11-17: Yahoo files these under mining industries, but by their own business
    # description (Yahoo profile) they mine nothing — fabricators, traders, other businesses.
    "0061.HK": "Green Leader — cassava cultivation and IT",
    "0098.HK": "Xingfa Aluminium — aluminium profiles (fabrication)",
    "0505.HK": "Xingye Alloy — copper plates and strips (fabrication)",
    "0521.HK": "CWT International — logistics and commodity marketing",
    "0637.HK": "Lee Kee — non-ferrous metals trading",
    "0645.HK": "Ares Asia — coal trading, no mines",
    "1051.HK": "G-Resources — principal investment, financial services, property",
    "1182.HK": "Success Dragon — gold processing and trading, no mines",
    "1733.HK": "E-Commodities — coal processing and trading, no mines",
    "2078.HK": "PanAsialum — aluminium products (fabrication)",
    "2326.HK": "New Provenance Everlasting — sources and sells minerals (trading)",
    "6128.HK": "Graphex — graphite/graphene processing, no mine",
    "6885.HK": "Henan Jinma Energy — coke and coking chemicals",
    "8095.HK": "Beida Jade Bird — metals trading and tourism",
    "8250.HK": "Du Du Holdings — fresh produce trading",
    # scope = metal ores, coal, uranium (Sean 2026-11-17)
    "0166.HK": "New Times — metals trading",
    "1205.HK": "CITIC Resources — oil exploration and production first",
    "1380.HK": "China Kingstone — marble, not a metal",
    "2693.HK": "Jinyan Kaolin — kaolin, not a metal",
}
# ticker -> (commodity, stage) by what the company actually does
RELABEL = {
    "2623.HK": ("Iron Ore", "Producer"),  # Hong Kong Gold Industry Group: despite the name, iron and ilmenite ores
    "3858.HK": ("Tungsten", "Developer"),  # Jiaxin International: Boguty tungsten, Kazakhstan
    "2245.HK": ("Nickel", "Producer"),     # Lygend: nickel (Indonesia)
    "1091.HK": ("Manganese", "Producer"),  # South Manganese
    "1258.HK": ("Copper", "Major Producer"),  # China Nonferrous Mining: copper (Zambia, DRC)
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


# HK miners the Yahoo sweep does not return (2026-11-17): each is on etnet's Hang Seng industry list
# "Gold & Precious Metals", "Diversified Metals & Minerals" or "Coal", and its own business
# description says it explores, mines or produces ore, concentrate or primary metal.
EXTRA_COMPANIES = [
    ("0159.HK", "Brockman Mining", "Iron Ore", "Developer"),
    ("0433.HK", "North Mining Shares", "Molybdenum", "Producer"),
    ("0893.HK", "China Vanadium Titano-Magnetite Mining", "Iron Ore", "Producer"),
    ("1370.HK", "Aowei Holding", "Iron Ore", "Producer"),
    ("2546.HK", "Xizang Zhihui Mining", "Zinc/Lead", "Producer"),
    ("9936.HK", "Ximei Resources Holding", "Tantalum", "Producer"),
]
# Names Yahoo garbles (counter suffixes such as "-500" are not part of the name)
NAME_FIX = {"2623.HK": "Hong Kong Gold Industry Group"}
# HK names in Chinese as etnet shows them, simplified / traditional: what people search for (灵宝黄金, not "Lingbao")
NAME_ZH = {
    "0065.HK": "弘海高新",
    "0159.HK": "布莱克万矿业 / 布萊克萬礦業",
    "0166.HK": "新时代集团控股 / 新時代集團控股",
    "0274.HK": "复兴亚洲 / 復興亞洲",
    "0276.HK": "蒙古能源",
    "0340.HK": "潼关黄金 / 潼關黃金",
    "0358.HK": "江西铜业股份 / 江西銅業股份",
    "0433.HK": "北方矿业 / 北方礦業",
    "0486.HK": "俄铝 / 俄鋁",
    "0601.HK": "稀镁科技 / 稀鎂科技",
    "0621.HK": "坛金矿业 / 壇金礦業",
    "0639.HK": "首钢资源 / 首鋼資源",
    "0661.HK": "中国大冶有色金属 / 中國大冶有色金屬",
    "0769.HK": "中国稀土 / 中國稀土",
    "0815.HK": "中国白银集团 / 中國白銀集團",
    "0866.HK": "中国秦发 / 中國秦發",
    "0893.HK": "中国铁钛 / 中國鐵鈦",
    "0975.HK": "MONGOL MINING",
    "1029.HK": "铁货 / 鐵貨",
    "1088.HK": "中国神华 / 中國神華",
    "1091.HK": "南方锰业 / 南方錳業",
    "1164.HK": "中广核矿业 / 中廣核礦業",
    "1171.HK": "兖矿能源 / 兗礦能源",
    "1205.HK": "中信资源 / 中信資源",
    "1208.HK": "五矿资源 / 五礦資源",
    "1229.HK": "南南资源 / 南南資源",
    "1258.HK": "中国有色矿业 / 中國有色礦業",
    "1277.HK": "力量发展 / 力量發展",
    "1303.HK": "汇力资源 / 滙力資源",
    "1370.HK": "奥威控股 / 奧威控股",
    "1378.HK": "中国宏桥 / 中國宏橋",
    "1380.HK": "中国金石 / 中國金石",
    "1393.HK": "恒鼎实业 / 恒鼎實業",
    "1712.HK": "龙资源 / 龍資源",
    "1738.HK": "飞尚无烟煤 / 飛尚無煙煤",
    "1772.HK": "赣锋锂业 / 贛鋒鋰業",
    "1787.HK": "山东黄金 / 山東黃金",
    "1818.HK": "招金矿业 / 招金礦業",
    "1878.HK": "南戈壁",
    "1898.HK": "中煤能源",
    "2099.HK": "中国黄金国际 / 中國黃金國際",
    "2112.HK": "恩典生命科技",
    "2212.HK": "高鹏矿业 / 高鵬礦業",
    "2237.HK": "中国石墨 / 中國石墨",
    "2245.HK": "力勤资源 / 力勤資源",
    "2259.HK": "紫金黄金国际 / 紫金黃金國際",
    "2302.HK": "中核国际 / 中核國際",
    "2362.HK": "金川国际 / 金川國際",
    "2489.HK": "集海黄金 / 集海黃金",
    "2546.HK": "智汇矿业 / 智匯礦業",
    "2600.HK": "中国铝业 / 中國鋁業",
    "2610.HK": "南山铝业国际 / 南山鋁業國際",
    "2623.HK": "香港黄金产业集团 / 香港黃金產業集團",
    "2693.HK": "金岩高岭新材 / 金岩高嶺新材",
    "2788.HK": "创新实业 / 創新實業",
    "2798.HK": "久泰邦达能源 / 久泰邦達能源",
    "2899.HK": "紫金矿业 / 紫金礦業",
    "3330.HK": "灵宝黄金 / 靈寶黃金",
    "3636.HK": "金浔资源 / 金潯資源",
    "3668.HK": "兖煤澳大利亚 / 兗煤澳大利亞",
    "3788.HK": "罕王黄金 / 罕王黃金",
    "3833.HK": "新疆新鑫矿业 / 新疆新鑫礦業",
    "3858.HK": "佳鑫国际资源 / 佳鑫國際資源",
    "3939.HK": "万国黄金集团 / 萬國黃金集團",
    "3993.HK": "洛阳钼业 / 洛陽鉬業",
    "6228.HK": "自由黄金 / 自由黃金",
    "6693.HK": "赤峰黄金 / 赤峰黃金",
    "8028.HK": "天时资源 / 天時資源",
    "8203.HK": "凯顺控股 / 凱順控股",
    "8299.HK": "大唐黄金 / 大唐黃金",
    "8331.HK": "倍搏集团 / 倍搏集團",
    "9696.HK": "天齐锂业 / 天齊鋰業",
    "9936.HK": "稀美资源 / 稀美資源",
}


def _named(row: tuple) -> tuple:
    ticker, name, commodity, stage = row
    name = NAME_FIX.get(ticker, name)
    if ticker in NAME_ZH and NAME_ZH[ticker] not in name:
        name = f"{name} {NAME_ZH[ticker]}"
    return ticker, name, commodity, stage


# Same tuple shape as the original module: (ticker, name, commodity, stage)
_LISTED = {c["ticker"] for c in _PAYLOAD["companies"]}
MINING_UNIVERSE = [_named(_row(c)) for c in _PAYLOAD["companies"] if c["ticker"] not in NOT_MINERS] + [
    _named(r) for r in EXTRA_COMPANIES if _PAYLOAD.get("market") == "hk" and r[0] not in _LISTED
]

# ticker -> "curated" | "screen"
UNIVERSE_SRC = {c["ticker"]: c.get("src", "screen") for c in _PAYLOAD["companies"]}
UNIVERSE_SRC.update({r[0]: "curated" for r in EXTRA_COMPANIES})      # stage set by hand, not by the screen

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


# ── Universe review 2026-11-17 (CA/US/UK/ZA/ID/CN): see data/universe_review.py ──
from data import universe_review as _review  # noqa: E402

MINING_UNIVERSE = _review.apply(MINING_UNIVERSE, _PAYLOAD.get("market"))
UNIVERSE_SRC.update({r[0]: "curated" for r in _review.ADD.get(_PAYLOAD.get("market"), [])})
ALL_COMMODITIES = sorted({row[2].split("/")[0] for row in MINING_UNIVERSE})
