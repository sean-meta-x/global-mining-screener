# -*- coding: utf-8 -*-
"""The fund's shared site style — one file, copied unchanged into each Streamlit repo.

Look: a brokerage-terminal feel. Deep navy top bar and sidebar, white content with light
grey panels, dense tables, right-aligned tabular numbers, green up / red down, and red
used for nothing else but emphasis. Colours and fonts that Streamlit can set itself live
in .streamlit/config.toml; this file adds what config cannot (density, the header bar,
number alignment) and two small components: the account summary strip and the panel title.

Copies: fund-hq (source), fund-research, australia-mining-screener, global-mining-screener.
Change it here first and copy the file over; THEME_VERSION says which copy is which.

Usage (top of each app.py, right after st.set_page_config):
    import site_theme
    site_theme.apply()
"""
from html import escape

import streamlit as st

THEME_VERSION = "2026-11-16.3"

NAVY = "#0B1F3A"        # top bar, sidebar, headings
NAVY_2 = "#16325C"      # hover / active in the sidebar
INK = "#1B2430"         # body text
MUTED = "#5F6B7A"       # labels, captions
LINE = "#DDE2E8"        # borders, table rules
PANEL = "#F4F6F9"       # panels and table headers
RED = "#C8102E"         # emphasis and losses — nothing decorative
GREEN = "#1A7F37"       # gains

_CSS = f"""
<style>
:root {{ --navy:{NAVY}; --navy2:{NAVY_2}; --ink:{INK}; --muted:{MUTED}; --line:{LINE};
         --panel:{PANEL}; --red:{RED}; --green:{GREEN}; }}

/* header: a thin navy bar instead of Streamlit's white strip */
[data-testid="stHeader"] {{ background: var(--navy); height: 2.6rem; }}
[data-testid="stHeader"] * {{ color: #fff !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}

/* density: less air around the page and between blocks */
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 3.4rem !important;
    padding-bottom: 2rem !important; max-width: 1480px; }}
[data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}
h1 {{ font-size: 1.45rem !important; font-weight: 700 !important; color: var(--navy) !important;
      margin: 0 0 .2rem !important; padding: 0 !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.15rem !important; color: var(--navy) !important; padding-top: .4rem !important; }}
h3, h4 {{ font-size: 1rem !important; color: var(--navy) !important; padding-top: .3rem !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--muted) !important; font-size: .78rem !important; }}

/* numbers: tabular figures everywhere, right-aligned in tables */
[data-testid="stMetricValue"], [data-testid="stDataFrame"], [data-testid="stTable"], .num {{
    font-variant-numeric: tabular-nums; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ text-align: right; padding: .25rem .6rem !important; }}
[data-testid="stTable"] td:first-child, [data-testid="stTable"] th:first-child {{ text-align: left; }}

/* metrics: flat panels, small labels, no shadows */
[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--line); border-radius: 4px;
    padding: .55rem .8rem; box-shadow: none; }}
[data-testid="stMetricLabel"] p {{ font-size: .74rem !important; color: var(--muted) !important;
    text-transform: uppercase; letter-spacing: .04em; }}
[data-testid="stMetricValue"] {{ font-size: 1.35rem !important; font-weight: 600 !important; color: var(--ink); }}

/* containers, expanders, tabs: square-ish, thin rules */
[data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 4px !important;
    border-color: var(--line) !important; }}
[data-testid="stExpander"] summary {{ font-weight: 600; color: var(--navy); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ padding: .35rem .9rem; font-size: .85rem; font-weight: 600; color: var(--muted); }}
.stTabs [aria-selected="true"] {{ color: var(--navy) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--red) !important; height: 2px; }}

/* sidebar: navy, grouped navigation */
[data-testid="stSidebar"] {{ background: var(--navy); }}
[data-testid="stSidebar"] * {{ color: #E8EDF4; }}
[data-testid="stSidebarNav"] a {{ border-radius: 3px; padding: .2rem .5rem; }}
[data-testid="stSidebarNav"] a:hover {{ background: var(--navy2); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: var(--navy2); box-shadow: inset 3px 0 0 var(--red); }}
[data-testid="stSidebarNavSeparator"], [data-testid="stNavSectionHeader"] {{ color: #8FA3BF !important;
    font-size: .7rem !important; text-transform: uppercase; letter-spacing: .08em; }}
[data-testid="stSidebar"] .stButton button, [data-testid="stSidebar"] [data-testid="stLinkButton"] a {{
    background: transparent; border: 1px solid #2E4A72; color: #E8EDF4 !important; border-radius: 3px; }}
[data-testid="stSidebar"] .stButton button {{ padding: .2rem .4rem; min-height: 2rem; }}
[data-testid="stSidebar"] .stButton button p {{ font-size: .8rem; white-space: nowrap; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] [data-testid="stMetric"] {{ background: var(--navy2); border-color: #2E4A72; }}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ color: #FFFFFF; font-size: 1.05rem !important; }}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {{ color: #9FB3CE !important; }}
[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: #2E4A72 !important; border-radius: 3px; }}
[data-testid="stSidebar"] [data-baseweb="tag"] span {{ color: #FFFFFF !important; }}

/* account strip and panel titles (components below) */
.acct-strip {{ display: flex; flex-wrap: wrap; background: var(--navy); color: #fff; border-radius: 4px;
    margin: 0 0 .5rem; }}
.acct-strip .cell {{ padding: .45rem 1rem; border-right: 1px solid #23406A; min-width: 8.5rem; }}
.acct-strip .cell:last-child {{ border-right: 0; }}
.acct-strip .lbl {{ font-size: .68rem; color: #9FB3CE; text-transform: uppercase; letter-spacing: .06em; }}
.acct-strip .val {{ font-size: 1.05rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.acct-strip .sub {{ font-size: .72rem; color: #9FB3CE; font-variant-numeric: tabular-nums; }}
.acct-strip .up {{ color: #4ADE80; }} .acct-strip .down {{ color: #FF6B7D; }}
.panel-title {{ font-size: .8rem; font-weight: 700; color: var(--navy); text-transform: uppercase;
    letter-spacing: .06em; border-bottom: 2px solid var(--navy); padding-bottom: .15rem; margin: .4rem 0 .3rem; }}
.up {{ color: var(--green); }} .down {{ color: var(--red); }}

@media (max-width: 640px) {{
  .acct-strip .cell {{ min-width: 48%; border-right: 0; border-bottom: 1px solid #23406A; }}
  .block-container, [data-testid="stMainBlockContainer"] {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
}}
</style>
"""


def apply() -> None:
    """Inject the shared CSS. Call once per page run, after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sign_class(value) -> str:
    """'up' / 'down' / '' for a signed number (None and 0 are neutral)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return "up" if v > 0 else "down" if v < 0 else ""


def signed_color(value) -> str:
    """For DataFrame.style.map: green for gains, red for losses."""
    cls = sign_class(value)
    return f"color: {GREEN}" if cls == "up" else f"color: {RED}" if cls == "down" else ""


def account_strip(cells: list[tuple]) -> None:
    """The navy summary bar at the top of a page.

    cells: (label, value) or (label, value, sub) or (label, value, sub, signed_number) —
    the last one, when given, colours the value green / red. Every string is escaped."""
    parts = []
    for cell in cells:
        label, value, sub, signed = (list(cell) + [None, None])[:4]
        cls = sign_class(signed) if signed is not None else ""
        parts.append(f'<div class="cell"><div class="lbl">{escape(str(label))}</div>'
                     f'<div class="val {cls}">{escape(str(value))}</div>'
                     + (f'<div class="sub">{escape(str(sub))}</div>' if sub else "") + "</div>")
    st.markdown(f'<div class="acct-strip">{"".join(parts)}</div>', unsafe_allow_html=True)


def panel_title(text: str) -> None:
    """A small uppercase section rule — lighter than st.subheader."""
    st.markdown(f'<div class="panel-title">{escape(text)}</div>', unsafe_allow_html=True)
