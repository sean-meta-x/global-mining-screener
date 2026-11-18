"""Weekly IC check of the trading score, and the gate on decision cards (Sean 2026-11-18).

The screener has two uses. For trading, the composite score only counts once it has shown it ranks
forward returns: until then, grades are for research and no 【决策】 card is made from them.

How it is measured, from the station DB's own daily snapshots (what the site actually said):
- a window starts on a snapshot date and ends on the first snapshot HORIZON_DAYS calendar days later
  (28 days ≈ 20 trading days; the refresh also runs at weekends, so snapshots are not trading days);
- names: tradable (US$20M cap, US$25k/day traded), graded (no Stale / physical / secondary listing);
- rank IC = Spearman correlation of the score with the window's return;
- Buy excess = average return of Strong Buy + Buy minus the average of all those names.

The gate only counts snapshots made by the trading score: on or after SCORER_SINCE, and only from the
first date whose composites match config.SCORE_WEIGHTS (so snapshots made before the merge never
count). Windows do not overlap within a market. The markets of one start date move together, so they
are averaged into one number per date, and the test runs on those dates. Validated when:
  at least MIN_STARTS start dates, mean rank IC >= MIN_IC with t >= MIN_T,
  and Buy excess > 0 on average with t > 0 over at least half of those dates.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

SCORER_SINCE = "2026-11-19"   # trading score (valuation + health) live from the 11-19 refresh
HORIZON_DAYS = 28             # calendar days ≈ 20 trading days
STEP_DAYS = 7                 # a new window every week (for the report; the gate de-overlaps)
MIN_NAMES = 15
MIN_BUYS = 3
MIN_STARTS, MIN_IC, MIN_T = 4, 0.03, 2.0
MATCH_SHARE = 0.9             # a date is "trading score" when this share of composites match the weights
NOT_GRADED = "Stale|实物持有|副上市"
FACTORS = ["score_composite", "score_valuation", "score_health", "score_momentum",
           "score_mining", "score_commodity", "score_stage"]
SUB = {"valuation": "score_valuation", "health": "score_health", "momentum": "score_momentum",
       "mining": "score_mining", "commodity": "score_commodity", "stage": "score_stage"}
LIQ_MCAP_USD, LIQ_ADV_USD = 20e6, 25e3
COLUMNS = ["ticker", "snap_date", "price", "market_cap", "avg_turnover", "grade"] + FACTORS


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < MIN_NAMES or x[ok].nunique() < 3:
        return np.nan
    return float(x[ok].rank().corr(y[ok].rank()))


def _plus(day: str, days: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=days)).isoformat()


def scorer_dates(snaps: pd.DataFrame, weights: dict[str, float]) -> set[str]:
    """Snapshot dates whose composites are the weighted sum of the sub-scores under `weights`
    (micro caps pinned at 50 and Stale names capped are ignored by the share)."""
    w = sum(weights.values()) or 1.0
    expect = sum(pd.to_numeric(snaps[c], errors="coerce").fillna(0) * weights.get(k, 0) / w
                 for k, c in SUB.items()).clip(0, 100)
    ok = (pd.to_numeric(snaps["score_composite"], errors="coerce") - expect).abs() <= 0.15
    return {d for d, s in ok.groupby(snaps["snap_date"]) if s.mean() >= MATCH_SHARE}


def windows(snaps: pd.DataFrame, fx: float, market: str, weights: dict[str, float] | None = None,
            horizon: int = HORIZON_DAYS, step: int = STEP_DAYS) -> list[dict]:
    """One record per window: start/end date, names, rank IC per factor, Buy excess (%), whether the
    start date was scored with `weights`, and how many names had no end price (dropped)."""
    snaps = snaps.drop_duplicates(["ticker", "snap_date"], keep="last")
    dates = sorted(snaps["snap_date"].unique())
    live = scorer_dates(snaps, weights) if weights else set()
    px = snaps.pivot(index="snap_date", columns="ticker", values="price")
    out, next_start = [], ""
    for d0 in dates:
        if d0 < next_start:
            continue
        d1 = next((d for d in dates if d >= _plus(d0, horizon)), None)
        if d1 is None:
            break
        next_start = _plus(d0, step)
        x = snaps[snaps["snap_date"] == d0].set_index("ticker")
        x = x[~x["grade"].fillna("").str.contains(NOT_GRADED)
              & (pd.to_numeric(x["market_cap"], errors="coerce") * fx >= LIQ_MCAP_USD)
              & (pd.to_numeric(x["avg_turnover"], errors="coerce") * fx >= LIQ_ADV_USD)]
        fwd = (px.loc[d1] / px.loc[d0] - 1).reindex(x.index)
        kept = fwd[fwd.notna() & (fwd.abs() < 3)]      # no end price, or a broken one (> +300 %)
        x = x.loc[kept.index]
        if len(x) < MIN_NAMES:
            continue
        rec = dict(market=market, d0=d0, d1=d1, n=len(x), dropped=int(len(fwd) - len(kept)),
                   live=d0 in live)
        for f in FACTORS:
            rec[f] = _rank_ic(pd.to_numeric(x[f], errors="coerce"), kept) if f in x else np.nan
        buy = x["grade"].fillna("").str.contains("Buy")
        rec["n_buy"] = int(buy.sum())
        rec["buy_excess"] = float((kept[buy].mean() - kept.mean()) * 100) if buy.sum() >= MIN_BUYS else np.nan
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


def by_date(recs: list[dict], keys=("score_composite", "buy_excess")) -> list[dict]:
    """Average the markets that share a start date into one record per date."""
    if not recs:
        return []
    df = pd.DataFrame(recs).groupby("d0")[list(keys)].mean()
    return [dict(d0=d, **row) for d, row in df.to_dict("index").items()]


def stats(recs: list[dict], key: str) -> dict:
    s = pd.Series([r.get(key) for r in recs], dtype=float).dropna()
    n = len(s)
    t = float(s.mean() / s.std() * np.sqrt(n)) if n > 1 and s.std() > 0 else np.nan
    return dict(mean=float(s.mean()) if n else np.nan, t=t, pos=float((s > 0).mean()) if n else np.nan, n=n)


def verdict(recs: list[dict], since: str = SCORER_SINCE) -> dict:
    """Gate on decision cards: validated (bool), the numbers behind it, and the reason in Chinese."""
    win = [r for r in non_overlapping([r for r in recs if r.get("live")], since)
           if not np.isnan(r["score_composite"])]
    dates = by_date(win)
    ic, ex = stats(dates, "score_composite"), stats(dates, "buy_excess")
    out = dict(validated=False, since=since, windows=len(win), starts=len(dates),
               ic=ic["mean"], t=ic["t"], ic_pos=ic["pos"], buy_excess=ex["mean"], buy_t=ex["t"],
               buy_dates=ex["n"])
    if len(dates) < MIN_STARTS:
        out["reason"] = (f"样本不够：新评分 {since} 起只有 {len(dates)} 个不重叠的 20 个交易日窗口起点"
                         f"（需要 ≥{MIN_STARTS} 个）")
    elif not (ic["mean"] >= MIN_IC and ic["t"] >= MIN_T):
        out["reason"] = f"IC 不够：平均 rank IC {ic['mean']:+.3f}、t {ic['t']:+.2f}（需要 ≥{MIN_IC} 且 t ≥{MIN_T}）"
    elif not (ex["n"] * 2 >= len(dates) and ex["mean"] > 0 and ex["t"] > 0):
        out["reason"] = ("Buy/Strong Buy 没跑赢同市场可交易名单" if ex["n"] * 2 >= len(dates)
                         else "Buy/Strong Buy 太少，一半以上的窗口算不出超额")
    else:
        out["validated"] = True
        out["reason"] = (f"通过：{len(dates)} 个起点，rank IC {ic['mean']:+.3f}、t {ic['t']:+.2f}，"
                         f"Buy 超额 {ex['mean']:+.2f}%")
    return out
