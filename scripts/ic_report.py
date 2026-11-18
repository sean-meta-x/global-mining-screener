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

from config import MARKETS, SCORE_WEIGHTS        # noqa: E402
from data import ic_gate as g                    # noqa: E402

FACTOR_ZH = {"score_composite": "综合分", "score_valuation": "估值", "score_health": "财务健康",
             "score_momentum": "超跌反转（权重 0）", "score_mining": "矿业（权重 0）",
             "score_commodity": "商品（权重 0）", "score_stage": "阶段（权重 0）"}


def metals(db: Path) -> dict[str, pd.Series]:
    """Metal prices the station holds (its own feed), indexed by ISO date."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        m = pd.read_sql("SELECT price_date, commodity, price FROM commodity_prices", c)
    return {k: g.set_index("price_date")["price"].sort_index() for k, g in m.groupby("commodity")}


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
    lines = ["| 因子（市场×矿种组内、扣金属贝塔） | 平均 rank IC | t | IC>0 占比 | 窗口 |", "|---|---|---|---|---|"]
    for f in g.FACTORS:
        s = g.stats(recs, f)
        lines.append(f"| {FACTOR_ZH[f]} | {_f(s['mean'], '+.3f')} | {_f(s['t'], '+.2f')} | {_f(s['pos'], '.0%')} | {s['n']} |")
    s = g.stats(recs, "q5q1")
    lines.append(f"| Q5−Q1（毛，%/窗口） | {_f(s['mean'] * 100, '+.2f')} | {_f(s['t'], '+.2f')} | {_f(s['pos'], '.0%')} | {s['n']} |")
    s = g.stats(recs, "q5q1_net")
    lines.append(f"| Q5−Q1（扣单边 50bp×4，%/窗口） | {_f(s['mean'] * 100, '+.2f')} | {_f(s['t'], '+.2f')} | {_f(s['pos'], '.0%')} | {s['n']} |")
    s = g.stats(recs, "buy_excess")
    lines.append(f"| Buy+Strong Buy 超额（%/20日，扣金属） | {_f(s['mean'], '+.2f')} | {_f(s['t'], '+.2f')} | {_f(s['pos'], '.0%')} | {s['n']} |")
    return lines


def report(recs: list[dict], v: dict, today: str, title: str) -> str:
    since = [r for r in recs if r["live"] and r["d0"] >= v["since"]]
    hist = g.non_overlapping(recs)
    by_m = pd.DataFrame(recs).groupby("market") if recs else None
    out = [f"# {title} · {today}", "",
           f"**评级门：{'✅ 已通过，评级可用于【决策】卡' if v['validated'] else '⛔ 未通过，评级仅供研究，不生成【决策】卡'}**",
           "", f"{v['reason']}。", "",
           f"规则：只看新评分（估值 + 财务健康）{v['since']} 起真正生效之后、跨市场互不重叠的"
           f" {g.HORIZON_DAYS} 天（≈20 个交易日）独立期（每期取各市场在期初 {g.STEP_DAYS} 天内开始的第一个窗口并取平均），"
           f"需要 ≥{g.MIN_STARTS} 个独立期，市场×矿种组内、扣金属贝塔后的平均 rank IC ≥{g.MIN_IC}、t ≥{g.MIN_T}，"
           "且扣单边 50bp×4 成本后 Q5（组内评分前五分之一，组 ≥8 家才分档）平均不跑输 Q1。名单 = 市值 ≥US$20M、日成交 ≥US$25k、"
           "不含停牌冻结价 / 实物持有 / 副上市；窗口内退市或被收购的按最后成交价计入，不剔除。"
           f"每周起点（重叠）的 Newey-West t：{_f(v.get('nw_t'), '+.2f')}。规则版本 {v.get('rules_version')}，"
           "每次运行记录在 reports/ic_trials.jsonl。", "",
           f"## 新评分上线以来（每周起点、各市场分开，窗口有重叠，t 偏高；{len(since)} 个窗口）", ""]
    out += table(since) if since else ["还没有走完 20 个交易日的窗口。"]
    out += ["", f"## 全部历史，不重叠窗口（{len(hist)} 个；{v['since']} 之前是旧评分）", ""] + table(hist)
    if by_m is not None:
        out += ["", "## 分市场（全部历史，每周起点）", "", "| 市场 | 综合分 IC | 估值 IC | 健康 IC | Q5−Q1 % | 窗口 |",
                "|---|---|---|---|---|---|"]
        for m, d in by_m:
            out.append(f"| {m} | {_f(d['score_composite'].mean(), '+.3f')} | {_f(d['score_valuation'].mean(), '+.3f')} | "
                       f"{_f(d['score_health'].mean(), '+.3f')} | {_f(d['q5q1'].mean() * 100, '+.2f')} | {len(d)} |")
    left = sum(r["left_in_window"] for r in recs)
    out += ["", f"限制：每日快照只从 2026-07 起；名单取当时在库的公司（之前已退市的不在样本）；窗口内离开快照的"
            f"共 {left} 个名字次按最后成交价计入（收购价拿不到时以最后成交价近似）；Q5−Q1 净值按每个窗口两腿全换手估算，偏保守。"]
    return "\n".join(out) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = []
    for m, spec in MARKETS.items():
        db = ROOT / f"screener_{m}.db"
        if not db.exists():
            sys.exit(f"missing {db.name}: no report rather than a report on fewer markets")
        recs += g.windows(load(db), spec["fx_to_usd"], m, SCORE_WEIGHTS, metals(db))
    v = g.verdict(recs)
    (ROOT / "reports").mkdir(exist_ok=True)
    trial = dict(run=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), rules_version=v["rules_version"],
                 windows=len(recs), **{k: v[k] for k in ("starts", "ic", "t", "nw_t", "q5q1", "validated")})
    with open(ROOT / "reports" / "ic_trials.jsonl", "a", encoding="utf-8") as f:      # every attempt, for deflated Sharpe
        f.write(json.dumps({k: (None if isinstance(x, float) and np.isnan(x) else x) for k, x in trial.items()},
                           ensure_ascii=False) + "\n")
    today = dt.date.today().isoformat()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "ic_weekly.md").write_text(report(recs, v, today, "交易评分 IC 周报（7 站）"), encoding="utf-8")
    status = dict(as_of=today, decision_cards="allowed" if v["validated"] else "blocked",
                  **{k: (None if isinstance(x, float) and np.isnan(x) else x) for k, x in v.items()})
    (ROOT / "ic_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(v["reason"])


if __name__ == "__main__":
    main()
