"""Universe review, 2026-11-17 (CA, US, UK, ZA, ID, CN; HK is in universe.py).

Scope decided by Sean: metal ores, coal and uranium. Out: fertiliser, salt and potash, helium,
oil and gas, processors and fabricators, traders, mining contractors, other businesses, and codes
that no longer trade (acquired, delisted, ticker changed). Physical holders (Yellow Cake, Sprott
Physical Uranium) stay, tagged. Secondary listings stay, tagged.

How each line was decided: an investor review, then a Bloomberg check of every name on a
terminal (industry classification and listing status, read-only), then the company's own public
business description. Only the decision and the reason are recorded here — no licensed data.

Applied at load time by data/universe.py, so a rebuild of universes/*.json cannot undo it.
"""

_FERT = "fertiliser / potash / salt — not a metal, coal or uranium miner"
_SERV = "mining contractor or services, no mine of its own"
_PROC = "processor or fabricator, no mine"
_TRADE = "metals or coal trading, no mine"
_GONE_A = "acquired — no longer trades"
_GONE_D = "delisted"
_GONE_T = "ticker changed — this code no longer trades"
_OIL = "oil and gas"
_OTHER = "not a mining business"
_DUP = "same company listed twice on this station"
_WARRANT = "warrant, not a share"

REMOVE: dict[str, dict[str, str]] = {
    "ca": {
        "ACX.TO": "ACT Energy Technologies — oil and gas drilling", "CAF.V": "Canaf Investments — investment company",
        "CPS.V": "Canadian Premium Sand — frac sand", "FAR.TO": "Foraco — drilling contractor",
        "GEO.TO": "Geodrill — drilling contractor", "MDI.TO": "Major Drilling — drilling contractor",
        "FNR.V": "49 North Resources — asset manager", "HPQ.V": "HPQ Silicon — silicon processing",
        "IRO.V": "Inter-Rock Minerals — feed and building minerals", "LFST.V": "Lifeist Wellness — consumer health",
        "MET.V": "Metalore Resources — oil and gas", "NBM.V": "NEO Battery Materials — battery materials",
        "NTR.TO": "Nutrien — " + _FERT, "NVRO.V": "NVRO Metals — waste management",
        "PALI.V": "Palisades Goldcorp — investment company", "PLSR.V": "Pulsar Helium — helium",
        "PNE.TO": "Pine Cliff Energy — " + _OIL, "SKK.V": "Strikewell Energy — " + _OIL,
        "TIE.V": "Coloured Ties Capital — shell / asset manager", "TTG.V": "Trans Canada Gold — " + _OIL,
        "UNI.TO": "Unisync — apparel", "VGN.V": "Greencastle Resources — investment company",
        "WCE.V": "Wescan Energy — " + _OIL, "ISGI.V": "InsuraGuest Technologies — software",
        "MVY.H.V": "Moovly Media — software, halted", "JJJ.CN": "37 Capital — shell",
        "VYC.H.V": "Vanity Capital — shell / investment company", "HCC-H.V": "Hanna Capital — shell",
        "KEON.H.V": "Keon Capital — shell, " + _GONE_T, "ATX.V": "ATEX Resources — " + _DUP + " (ATX.TO kept)",
        "TECK-A.TO": "Teck class A — " + _DUP + " (class B kept)",
        **dict.fromkeys(["ADG.V", "AGA.V", "AU.V", "CYG.V", "ELY.V", "FOM.TO", "GTWO.TO", "LMR.V", "NGD.TO", "OLA.TO",
                         "PEGA.V", "QCX.V", "RBX.V", "RSH.CN", "RUP.TO", "RYE.V", "STRR.V", "VROY.V"], _GONE_A),
        **dict.fromkeys(["AII.TO", "GXX.V", "ICON.V", "MPVD.TO", "NZP.V", "RARE.V", "TKU.V", "URC.TO"], _GONE_D),
        **dict.fromkeys(["AFX.CN", "APMI.V", "BMET.V", "BOLT.CN", "CLM.V", "CMET.CN", "ETL-WT.V", "FCLI.V", "FMT.V",
                         "FNC.V", "FSY.TO", "GGA.V", "GGD-WT.TO", "GGLD.CN", "GLDS.CN", "GRAT.V", "GRHK.CN", "GUG.V",
                         "KSM.V", "LEO.CN", "MEX.V", "NEWD.CN", "NFG.V", "NILI.V", "ODV.V", "PGA.CN", "SGN.V", "SMRV.V",
                         "TXG.TO", "VAX.V", "WM.TO", "WRI.V"], _GONE_T),
    },
    "us": {
        "XCPT": "Xcana Petroleum — " + _OIL, "BSAI": "BluSky AI — not a miner (OTC)",
        "CSTM": "Constellium — aluminium rolling, " + _PROC, "KALU": "Kaiser Aluminum — " + _PROC,
        "MTRN": "Materion — advanced materials, " + _PROC, "SXC": "SunCoke — coke making",
        "LEU": "Centrus — uranium enrichment, not mining", "GSM": "Ferroglobe — silicon metal smelting, " + _PROC,
        "CLF": "Cleveland-Cliffs — steelmaker", "EMAT": "Evolution Metals & Technologies — " + _TRADE,
        "WWR": "Westwater Resources — graphite processing", "OICT": "Oicintra — insurance",
        "BLLYF": "80 Mile plc OTC line — primary listing is 80M.L",
        "NTR": "Nutrien — " + _FERT, "MOS": "Mosaic — " + _FERT, "ICL": "ICL Group — " + _FERT,
        "IPI": "Intrepid Potash — " + _FERT, "CMP": "Compass Minerals — " + _FERT,
        "GOLD": "now Gold.com Inc, a precious-metals distributor (Barrick trades as B)",
        "AMKBY": "now the A.P. Moller-Maersk ADR — not a miner",
        "NOVN": "Novan — pharma, " + _GONE_T, "PTM": "platinum ETN — delisted",
        "SBGL": "Sibanye Gold ADR — delisted (SBSW kept)", "LAAC": "Lithium Americas (Argentina) — " + _GONE_T,
        "CEIX": "CONSOL — merged into Core Natural Resources (CNR)", "ARCH": "Arch — merged into Core Natural Resources (CNR)",
        "SAND": "Sandstorm Gold — " + _GONE_A, "NGD": "New Gold — " + _GONE_A, "ORLA": "Orla Mining — " + _GONE_A,
        "AGRTF": _GONE_T, "ODV": "Osisko Development — " + _GONE_T, "REE": "REE Automotive — not a miner",
        "UUUFF": "Uranium One OTC line — no trading",
        **dict.fromkeys(["BGLWW", "CRMLW", "NAMMW", "NIOBW", "ODVWZ", "TMCWW"], _WARRANT),
    },
    "uk": {
        "CEY.L": "Centamin — " + _GONE_A, "HZM.L": "Horizonte Minerals — " + _GONE_D,
        "ZNWD.L": "Zinnwald Lithium — " + _GONE_A, "TM1.L": _GONE_T,
        "PLSR.L": "Pulsar Helium — helium", "HEX.L": "Helix Exploration — helium", "GEX.L": "Georgina Energy — helium / hydrogen",
        "ORCP.L": "Oracle Power — renewable energy", "CAPD.L": "Capital Ltd — drilling contractor",
        "0ICU.L": "DRDGOLD ADR line — " + _DUP, "AMGO.L": "Amigo Holdings — consumer finance",
        "CDL.L": "Cloudbreak Discovery — investment company", "CGO.L": "Contango Holdings — investment company",
        "GDP.L": "Goldplat — gold recovery and processing", "GEMR.L": "Gem Resources — financial services",
        "IRON.L": "Ironveld — processing", "SRES.L": "Sunrise Resources — industrial minerals, not metal",
        "SRVL.L": "Serval Resources — investment company",
    },
    "za": {
        "RNG.JO": "Randgold & Exploration — no operating mine (legal claims vehicle)",
        "AFT.JO": "Afrimat — construction materials",
    },
    "id": {
        "ALKA.JK": "Alakasa — aluminium, " + _PROC, "INAI.JK": "Indal Aluminium — " + _PROC,
        "UNTR.JK": "United Tractors — heavy equipment and contracting", "PTRO.JK": "Petrosea — " + _SERV,
        "DOID.JK": "BUMA — " + _SERV, "RMKO.JK": "Royaltama — " + _SERV, "SGER.JK": "Sumber Global Energy — " + _TRADE,
        "RMKE.JK": "RMK Energy — rail and port logistics", "ABMM.JK": "ABM Investama — " + _SERV,
        "MYOH.JK": "Samindo — " + _SERV, "UNIQ.JK": "Ulima Nitra — " + _SERV, "MINE.JK": "Sinar Terang Mandiri — " + _SERV,
        "DAAZ.JK": "Daaz Bara Lestari — " + _TRADE, "MITI.JK": "Mitra Investindo — shipping",
        "ITMA.JK": "Sumber Energi Andalan — holding, not a miner",
    },
    "cn": {
        **{t: _PROC for t in ["601137.SS", "601609.SS", "601677.SS", "601702.SS", "603876.SS", "603937.SS", "603978.SS",
                              "603271.SS", "688102.SS", "688231.SS", "688388.SS", "300930.SZ", "600255.SS", "603124.SS",
                              "001400.SZ", "000629.SZ", "000795.SZ", "002101.SZ", "002497.SZ", "300697.SZ", "301613.SZ",
                              "600172.SS"]},
        "601388.SS": "metal recycling", "000711.SZ": "engineering services",
        "600740.SS": "coking", "601015.SS": "coking", "600980.SS": "mining technology and chemicals, no mine",
        "900948.SS": "Yitai Coal B share — " + _DUP, "000792.SZ": "Qinghai Salt Lake — " + _FERT,
        "000408.SZ": "Zangge Mining — " + _FERT, "600331.SS": "Sichuan Hongda — " + _FERT,
        "600098.SS": "Guangzhou Development — energy and trading", "600157.SS": "Wintime Energy — power generation",
        "600753.SS": "Haiqin Energy — fuel trading", "601061.SS": "CITIC Metal — " + _TRADE,
    },
}

# (ticker, name, commodity, stage)
ADD: dict[str, list[tuple]] = {
    "ca": [("TXR.TO", "Torex Resources", "Gold", "Producer")],
    "uk": [("KAP.L", "Kazatomprom", "Uranium", "Major Producer"),
           ("ZIOC.L", "Zanaga Iron Ore", "Iron Ore", "Developer")],
    "id": [("ZINC.JK", "Kapuas Prima Coal", "Zinc/Lead", "Producer"),
           ("COAL.JK", "Black Diamond Resources", "Coal", "Producer")],
    "cn": [("002240.SZ", "Chengxin Lithium", "Lithium", "Producer"),
           ("001203.SZ", "Dazhong Mining", "Iron Ore", "Producer"),
           ("002192.SZ", "Youngy", "Lithium", "Producer"),
           ("000923.SZ", "HBIS Resources", "Iron Ore", "Producer"),
           ("600758.SS", "Liaoning Energy Industry", "Coal", "Producer"),
           ("000571.SZ", "Sundiro Holding", "Coal", "Producer"),
           ("001246.SZ", "Lygend Resources (A)", "Nickel", "Producer")],
}

# shown after the name: secondary listings, physical holders, dual listings
TAG = {
    "uk": {"YCA.L": "实物铀持有，不采矿", "KAP.L": "二次上市 · GDR"},
    "ca": {"U-UN.TO": "实物铀持有，不采矿"},
    "za": {t: "二次上市" for t in ("AGL.JO", "BHG.JO", "GLN.JO", "S32.JO")},
    "cn": {"001246.SZ": "A+H 两地上市"},
}


def apply(rows: list[tuple], market: str | None) -> list[tuple]:
    """rows: (ticker, name, commodity, stage) -> the reviewed list for this market."""
    gone, tags = REMOVE.get(market or "", {}), TAG.get(market or "", {})
    have = {r[0] for r in rows}
    out = [r for r in rows if r[0] not in gone] + [r for r in ADD.get(market or "", []) if r[0] not in have]
    return [(t, f"{n} · {tags[t]}" if t in tags else n, c, s) for t, n, c, s in out]
