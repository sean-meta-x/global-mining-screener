"""site_theme.py — the shared site style (copied unchanged into every Streamlit repo)."""
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import site_theme as th

ROOT = Path(__file__).parent.parent


def test_sign_class_and_colour():
    assert [th.sign_class(v) for v in (1.5, -0.1, 0, None, "x")] == ["up", "down", "", "", ""]
    assert th.signed_color(-2) == f"color: {th.RED}" and th.signed_color(3) == f"color: {th.GREEN}"


def test_account_strip_escapes_and_colours(monkeypatch):
    out = []
    monkeypatch.setattr(th.st, "markdown", lambda html, **k: out.append(html))
    th.account_strip([("NAV · US$", "116,358"), ("当日盈亏", "-1,470", "-1.3%", -1470), ("<b>x</b>", "1")])
    html = out[0]
    assert 'class="val down"' in html and "-1.3%" in html and "<b>x</b>" not in html and "&lt;b&gt;" in html


def test_config_theme_matches_the_module():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    assert cfg["primaryColor"] == th.NAVY and cfg["redColor"] == th.RED and cfg["greenColor"] == th.GREEN
    assert cfg["sidebar"]["backgroundColor"] == th.NAVY
