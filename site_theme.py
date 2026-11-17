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

THEME_VERSION = "2026-11-17.3"

NAVY = "#0B1F3A"        # top bar, sidebar, headings
NAVY_2 = "#16325C"      # hover / active in the sidebar
INK = "#1B2430"         # body text
MUTED = "#4A5565"       # labels, captions (darker since 11-17: grey-on-white was too faint)
LINE = "#DDE2E8"        # borders, table rules
PANEL = "#F4F6F9"       # panels and table headers
RED = "#C8102E"         # emphasis and losses — nothing decorative
GREEN = "#1A7F37"       # gains
AMBER = "#B26A00"       # warnings

_CSS = f"""
<style>
:root {{ --navy:{NAVY}; --navy2:{NAVY_2}; --ink:{INK}; --muted:{MUTED}; --line:{LINE};
         --panel:{PANEL}; --red:{RED}; --green:{GREEN}; --amber:{AMBER}; }}

/* header: a thin navy bar instead of Streamlit's white strip */
[data-testid="stHeader"] {{ background: var(--navy); height: 2.6rem; border-bottom: 2px solid var(--red); }}
[data-testid="stHeader"] * {{ color: #fff !important; }}
[data-testid="stHeader"]::before {{ content: var(--brand); position: absolute; left: 3.4rem; top: .62rem;
    color: #fff; font-weight: 700; font-size: .95rem; letter-spacing: .02em; white-space: nowrap; }}
/* the toolbar holds the button that opens the sidebar on a phone: hide its actions, never the toolbar */
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
[data-testid="stExpandSidebarButton"] {{ display: inline-flex !important; visibility: visible !important; }}
[data-testid="stExpandSidebarButton"] svg {{ fill: #fff; color: #fff; width: 1.4rem; height: 1.4rem; }}

/* density: less air around the page and between blocks */
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 3.4rem !important;
    padding-bottom: 2rem !important; max-width: 1480px; }}
[data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}
html, body, [data-testid="stAppViewContainer"] {{ font-size: 15px; }}
p, li, [data-testid="stMarkdownContainer"] {{ line-height: 1.55; }}
h1 {{ font-size: 1.5rem !important; font-weight: 700 !important; color: var(--navy) !important;
      margin: 0 0 .2rem !important; padding: 0 !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.15rem !important; color: var(--navy) !important; padding-top: .4rem !important; }}
h3, h4 {{ font-size: 1rem !important; color: var(--navy) !important; padding-top: .3rem !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--muted) !important; font-size: .84rem !important; line-height: 1.5; }}

/* numbers: tabular figures everywhere, right-aligned in tables */
[data-testid="stMetricValue"], [data-testid="stDataFrame"], [data-testid="stTable"], .num {{
    font-variant-numeric: tabular-nums; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ text-align: right; padding: .25rem .6rem !important; }}
[data-testid="stTable"] td:first-child, [data-testid="stTable"] th:first-child {{ text-align: left; }}
[data-testid="stTable"] th, [data-testid="stMarkdownContainer"] table th {{ background: var(--panel); color: var(--navy);
    font-weight: 700; font-size: .82rem; border-bottom: 2px solid var(--navy) !important; }}
[data-testid="stMarkdownContainer"] table td {{ font-variant-numeric: tabular-nums; }}

/* metrics: flat panels, small labels, no shadows */
[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--line); border-radius: 4px;
    padding: .55rem .8rem; box-shadow: none; }}
[data-testid="stMetricLabel"] p {{ font-size: .78rem !important; color: var(--muted) !important;
    text-transform: uppercase; letter-spacing: .04em; }}
[data-testid="stMetricValue"] {{ font-size: 1.35rem !important; font-weight: 600 !important; color: var(--ink); }}

/* containers, expanders, tabs: square-ish, thin rules */
[data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 4px !important;
    border-color: var(--line) !important; }}
[data-testid="stExpander"] summary {{ font-weight: 600; color: var(--navy); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ padding: .4rem 1rem; font-size: .92rem; font-weight: 600; color: var(--muted); }}
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
/* other-site buttons look like the navigation above them, not like form buttons */
[data-testid="stSidebar"] .stButton button, [data-testid="stSidebar"] [data-testid="stLinkButton"] a,
[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
    background: transparent; border: 0; color: #E8EDF4 !important; border-radius: 3px; justify-content: flex-start;
    text-align: left; }}
[data-testid="stSidebar"] .stButton button:hover, [data-testid="stSidebar"] [data-testid="stLinkButton"] a:hover,
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{ background: var(--navy2); }}
[data-testid="stSidebar"] .stButton button {{ padding: .2rem .5rem; min-height: 2rem; }}
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
.acct-strip .cell {{ padding: .5rem .9rem; border-right: 1px solid #23406A; flex: 1 1 0; min-width: 7rem; }}
.acct-strip .cell:last-child {{ border-right: 0; }}
.acct-strip .lbl {{ font-size: .72rem; color: #9FB3CE; text-transform: uppercase; letter-spacing: .06em; }}
.acct-strip .val {{ font-size: 1.25rem; font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }}
.acct-strip .sub {{ font-size: .78rem; color: #B9C8DC; font-variant-numeric: tabular-nums; text-align: right; }}
/* on navy the brand red is too dark to read: the same hue, lighter */
.acct-strip .up {{ color: #4ADE80; }} .acct-strip .down {{ color: #F2546A; }}
.panel-title {{ font-size: .86rem; font-weight: 700; color: var(--navy); text-transform: uppercase;
    letter-spacing: .06em; border-bottom: 2px solid var(--navy); padding-bottom: .15rem; margin: .4rem 0 .3rem; }}
.up {{ color: var(--green); }} .down {{ color: var(--red); }}
.badge {{ display: inline-block; padding: .05rem .45rem; border-radius: 3px; font-size: .74rem; font-weight: 700;
    letter-spacing: .03em; vertical-align: .08em; margin-right: .35rem; white-space: nowrap; }}
.badge.red {{ background: #FCE8EB; color: var(--red); }} .badge.amber {{ background: #FFF2DB; color: var(--amber); }}
.badge.green {{ background: #E3F4E8; color: var(--green); }} .badge.grey {{ background: #ECEFF3; color: var(--muted); }}
.badge.navy {{ background: var(--navy); color: #fff; }}
.dcards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); gap: .6rem; margin: .2rem 0 .4rem; }}
.dcard {{ border: 1px solid var(--line); border-left: 4px solid var(--navy); border-radius: 4px; padding: .55rem .75rem;
    background: #fff; }}
.dcard .t {{ font-weight: 700; color: var(--navy); font-size: 1rem; }}
.dcard .kv {{ display: grid; grid-template-columns: auto 1fr; gap: .1rem .7rem; margin-top: .35rem; font-size: .86rem; }}
.dcard .kv span:nth-child(odd) {{ color: var(--muted); }}
.dcard .kv span:nth-child(even) {{ font-variant-numeric: tabular-nums; text-align: right; }}
.dcard .s {{ margin-top: .35rem; font-size: .8rem; color: var(--muted); }}
.ddbar {{ height: .55rem; background: var(--panel); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }}
.ddbar > div {{ height: 100%; }}
.ddbar .green {{ background: var(--green); }} .ddbar .amber {{ background: var(--amber); }} .ddbar .red {{ background: var(--red); }}

@media (max-width: 640px) {{
  .acct-strip .cell {{ min-width: 48%; border-right: 0; border-bottom: 1px solid #23406A; }}
  .block-container, [data-testid="stMainBlockContainer"] {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
}}
</style>
"""


def apply(brand: str = "") -> None:
    """Inject the shared CSS. Call once per page run, after st.set_page_config.
    brand: the site name shown in the navy top bar (left, next to the sidebar toggle)."""
    label = brand.replace("\\", "").replace('"', "").replace("\n", " ")[:60]
    st.markdown(_CSS + f'<style>:root {{ --brand: "{label}"; }}</style>', unsafe_allow_html=True)


# status words -> badge colour, for badge() and for turning the old emoji dots into badges
TONE = {"red": "red", "amber": "amber", "green": "green", "grey": "grey", "navy": "navy"}
EMOJI_TONE = {"🔴": "red", "🟠": "amber", "🟡": "amber", "🟢": "green", "⚪": "grey", "🔵": "navy"}


def badge(text: str, tone: str = "grey") -> str:
    """HTML for a small status badge (escaped). Use inside st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="badge {TONE.get(tone, "grey")}">{escape(str(text))}</span>'


def emoji_to_badge(text: str, label: str | None = None) -> str:
    """'🔴 生效' -> a red badge reading '生效'. Text without a leading status emoji is returned escaped."""
    t = str(text).strip()
    for e, tone in EMOJI_TONE.items():
        if t.startswith(e):
            return badge(label or t[len(e):].strip() or "●", tone)
    return escape(t)


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
