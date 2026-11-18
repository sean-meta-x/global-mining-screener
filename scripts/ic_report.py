"""Weekly IC report + decision-card gate for the 7 stations (data/ic_gate.py has the rules).

Reads screener_{m}.db in the repo root (the workflow downloads them from the db-latest release) and
writes reports/ic_weekly.md and ic_status.json. Read-only on the DBs.

    python scripts/ic_report.py
"""
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MARKETS                       # noqa: E402
from data import ic_gate as g                    # noqa: E402

FACTOR_ZH = {"score_composite": "综合分", "score_valuation": "估值", "score_health": "财务健康",
             "score_momentum": "超跌反转（权重 0）", "score_mining": "矿业（权重 0）",
             "score_commodity": "商品（权重 0）", "score_stage": "阶段（权重 0）"}


def load(db: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        have = {r[1] for r in c.execute("PRAGMA table_info(stock_snapshots)")}
        cols = [k for k in g.COLUMNS if k in have]
        df = pd.read_sql(f"SELECT {','.join(cols)} FROM stock_snapshots", c)
    for k in g.COLUMNS:
        if k not in df:
            df[k] = np.nan
    return df


def _f(v, fmt):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def table(recs: list[dict]) -> list[str]:
    lines = ["| 因子 | 平均 rank IC | t | IC>0 占比 | 窗口 |", "|---|---|---|---|---|"]
    for f in g.FACTORS:
        s = g.stats(recs, f)
        lines.append(f"| {FACTOR_ZH[f]} | {_f(s['mean'], '+.3f')} | {_f(s['t'], '+.2f')} | {_f(s['pos'], '.0%')} | {s['n']} |")
    s = g.stats(recs, "buy_excess")
    lines.append(f"| Buy+Strong Buy 超额（%/20日） | {_f(s['mean'], '+.2f')} | {_f(s['t'], '+.2f')} | {_f(s['pos'], '.0%')} | {s['n']} |")
    return lines


def report(recs: list[dict], v: dict, today: str, title: str) -> str:
    since = [r for r in recs if r["d0"] >= v["since"]]
    hist = g.non_overlapping(recs)
    by_m = pd.DataFrame(recs).groupby("market") if recs else None
    out = [f"# {title} · {today}", "",
           f"**评级门：{'✅ 已通过，评级可用于【决策】卡' if v['validated'] else '⛔ 未通过，评级仅供研究，不生成【决策】卡'}**",
           "", f"{v['reason']}。", "",
           f"规则：只看新评分（估值 + 财务健康）上线 {v['since']} 之后、同一市场内不重叠的 20 个交易日窗口；"
           f"需要 ≥{g.MIN_WINDOWS} 个窗口、≥{g.MIN_STARTS} 个起点，平均 rank IC ≥{g.MIN_IC}、t ≥{g.MIN_T}，"
           "且 Buy+Strong Buy 平均跑赢同市场可交易名单。名单 = 市值 ≥US$20M、日成交 ≥US$25k、"
           "不含停牌冻结价 / 实物持有 / 副上市。", "",
           f"## 新评分上线以来（每周起点，窗口有重叠，t 偏高；{len(since)} 个窗口）", ""]
    out += table(since) if since else ["还没有走完 20 个交易日的窗口。"]
    out += ["", f"## 全部历史，不重叠窗口（{len(hist)} 个；{v['since']} 之前是旧评分）", ""] + table(hist)
    if by_m is not None:
        out += ["", "## 分市场（全部历史，每周起点）", "", "| 市场 | 综合分 IC | 估值 IC | 健康 IC | Buy 超额% | 窗口 |",
                "|---|---|---|---|---|---|"]
        for m, d in by_m:
            out.append(f"| {m} | {_f(d['score_composite'].mean(), '+.3f')} | {_f(d['score_valuation'].mean(), '+.3f')} | "
                       f"{_f(d['score_health'].mean(), '+.3f')} | {_f(d['buy_excess'].mean(), '+.2f')} | {len(d)} |")
    out += ["", "限制：每日快照只从 2026-07 起；名单取当时在库的公司；不含交易成本。"]
    return "\n".join(out) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = []
    for m, spec in MARKETS.items():
        db = ROOT / f"screener_{m}.db"
        if db.exists():
            recs += g.windows(load(db), spec["fx_to_usd"], m)
        else:
            print(f"missing {db.name}")
    v = g.verdict(recs)
    today = dt.date.today().isoformat()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "ic_weekly.md").write_text(report(recs, v, today, "交易评分 IC 周报（7 站）"), encoding="utf-8")
    status = dict(as_of=today, decision_cards="allowed" if v["validated"] else "blocked",
                  **{k: (None if isinstance(x, float) and np.isnan(x) else x) for k, x in v.items()})
    (ROOT / "ic_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(v["reason"])


if __name__ == "__main__":
    main()
