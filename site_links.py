"""Where the fund's public sites live — one list, used by every station's sidebar."""

DESK = "https://global-mining-fund.streamlit.app/"
BRIEF = "https://global-mining-fund.streamlit.app/brief"
RADAR = "https://global-mining-fund.streamlit.app/radar"
RESEARCH = "https://meta-research-lab.streamlit.app/"

# Every screener station: the ASX one (australia-mining-screener) and the seven overseas ones
# (global-mining-screener), in the order they refresh.
STATIONS = [
    ("🇦🇺 澳洲 AU", "https://meta-mining-screener.streamlit.app/"),
    ("🇨🇦 加拿大 CA", "https://ca-mining-screener.streamlit.app/"),
    ("🇺🇸 美国 US", "https://us-mining-screener.streamlit.app/"),
    ("🇬🇧 英国 UK", "https://uk-mining-screener.streamlit.app/"),
    ("🇿🇦 南非 ZA", "https://za-mining-screener.streamlit.app/"),
    ("🇮🇩 印尼 ID", "https://id-mining-screener.streamlit.app/"),
    ("🇭🇰 香港 HK", "https://hk-mining-screener.streamlit.app/"),
    ("🇨🇳 A 股 CN", "https://cn-mining-screener.streamlit.app/"),
]


def others(current_url: str) -> list[tuple[str, str]]:
    """The stations other than the one on screen."""
    return [(label, url) for label, url in STATIONS if url != current_url]
