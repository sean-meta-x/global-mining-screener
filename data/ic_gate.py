"""Weekly IC check of the trading score, and the gate on decision cards (Sean 2026-11-18).

The screener has two uses. For trading, the composite score only counts once it has shown it ranks
forward returns: until then, grades are for research and no 【决策】 card is made from them.

How it is measured, from the station DB's own daily snapshots (what the site actually said):
- a window starts on a snapshot date and ends HORIZON snapshots later (≈ 20 trading days);
- names: tradable (US$20M cap, US$25k/day traded), graded (no Stale / physical / secondary listing);
- rank IC = Spearman correlation of the score with the window's return;
- Buy excess = average return of Strong Buy + Buy minus the average of all those names.

The gate only uses windows that start on or after SCORER_SINCE (the day the trading score went live)
and do not overlap within a market. Validated when all of these hold:
  at least MIN_WINDOWS windows from at least MIN_STARTS different start dates,
  mean rank IC >= MIN_IC with t >= MIN_T, and Buy excess > 0 on average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCORER_SINCE = "2026-11-19"   # trading score (valuation + health) live from the 11-19 refresh
HORIZON = 20                  # snapshots ≈ trading days
STEP = 5                      # a new window every week (for the report; the gate de-overlaps)
MIN_NAMES = 15
MIN_BUYS = 3
MIN_WINDOWS, MIN_STARTS, MIN_IC, MIN_T = 6, 3, 0.03, 2.0
NOT_GRADED = "Stale|实物持有|副上市"
FACTORS = ["score_composite", "score_valuation", "score_health", "score_momentum",
           "score_mining", "score_commodity", "score_stage"]
LIQ_MCAP_USD, LIQ_ADV_USD = 20e6, 25e3
COLUMNS = ["ticker", "snap_date", "price", "market_cap", "avg_turnover", "grade"] + FACTORS


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < MIN_NAMES or x[ok].nunique() < 3:
        return np.nan
    return float(x[ok].rank().corr(y[ok].rank()))


def windows(snaps: pd.DataFrame, fx: float, market: str,
            horizon: int = HORIZON, step: int = STEP) -> list[dict]:
    """One record per window: start/end date, names, rank IC per factor, Buy excess (%)."""
    snaps = snaps.drop_duplicates(["ticker", "snap_date"], keep="last")
    dates = sorted(snaps["snap_date"].unique())
    px = snaps.pivot(index="snap_date", columns="ticker", values="price")
    out = []
    for i in range(0, len(dates) - horizon, step):
        d0, d1 = dates[i], dates[i + horizon]
        x = snaps[snaps["snap_date"] == d0].set_index("ticker")
        x = x[~x["grade"].fillna("").str.contains(NOT_GRADED)
              & (pd.to_numeric(x["market_cap"], errors="coerce") * fx >= LIQ_MCAP_USD)
              & (pd.to_numeric(x["avg_turnover"], errors="coerce") * fx >= LIQ_ADV_USD)]
        fwd = (px.loc[d1] / px.loc[d0] - 1).reindex(x.index)
        fwd = fwd[fwd.abs() < 3]                       # drop broken prices (> +300 % / -100 %)
        x = x.loc[fwd.index]
        if len(x) < MIN_NAMES:
            continue
        rec = dict(market=market, d0=d0, d1=d1, n=len(x))
        for f in FACTORS:
            rec[f] = _rank_ic(pd.to_numeric(x[f], errors="coerce"), fwd) if f in x else np.nan
        buy = x["grade"].fillna("").str.contains("Buy")
        rec["n_buy"] = int(buy.sum())
        rec["buy_excess"] = float((fwd[buy].mean() - fwd.mean()) * 100) if buy.sum() >= MIN_BUYS else np.nan
        out.append(rec)
    return out


def non_overlapping(recs: list[dict], since: str | None = None) -> list[dict]:
    """Per market, the first window on/after `since`, then each next one starting at or after the
    previous one's end."""
    out = []
    for m in sorted({r["market"] for r in recs}):
        last_end = since or ""
        for r in sorted((r for r in recs if r["market"] == m), key=lambda r: r["d0"]):
            if r["d0"] >= last_end:
                out.append(r)
                last_end = r["d1"]
    return out


def stats(recs: list[dict], key: str) -> dict:
    s = pd.Series([r.get(key) for r in recs], dtype=float).dropna()
    n = len(s)
    t = float(s.mean() / s.std() * np.sqrt(n)) if n > 1 and s.std() > 0 else np.nan
    return dict(mean=float(s.mean()) if n else np.nan, t=t, pos=float((s > 0).mean()) if n else np.nan, n=n)


def verdict(recs: list[dict], since: str = SCORER_SINCE) -> dict:
    """Gate on decision cards: validated (bool), the numbers behind it, and the reason in Chinese."""
    win = [r for r in non_overlapping(recs, since) if not np.isnan(r["score_composite"])]
    ic, ex = stats(win, "score_composite"), stats(win, "buy_excess")
    starts = len({r["d0"] for r in win})
    out = dict(validated=False, since=since, windows=len(win), starts=starts,
               ic=ic["mean"], t=ic["t"], ic_pos=ic["pos"], buy_excess=ex["mean"], buy_windows=ex["n"])
    if len(win) < MIN_WINDOWS or starts < MIN_STARTS:
        out["reason"] = (f"样本不够：新评分 {since} 起只有 {len(win)} 个不重叠的 20 日窗口、{starts} 个起点"
                         f"（需要 ≥{MIN_WINDOWS} 个窗口、≥{MIN_STARTS} 个起点）")
    elif not (ic["mean"] >= MIN_IC and ic["t"] >= MIN_T):
        out["reason"] = f"IC 不够：平均 rank IC {ic['mean']:+.3f}、t {ic['t']:+.2f}（需要 ≥{MIN_IC} 且 t ≥{MIN_T}）"
    elif not (ex["n"] and ex["mean"] > 0):
        out["reason"] = "Buy/Strong Buy 没跑赢同市场可交易名单" if ex["n"] else "Buy/Strong Buy 太少，算不出超额"
    else:
        out["validated"] = True
        out["reason"] = f"通过：{len(win)} 个窗口，rank IC {ic['mean']:+.3f}、t {ic['t']:+.2f}，Buy 超额 {ex['mean']:+.2f}%"
    return out
