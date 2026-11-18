"""Weekly IC check of the trading score, and the gate on decision cards (Sean 2026-11-18).

The screener has two uses. For trading, the composite score only counts once it has shown it ranks
forward returns: until then, grades are for research and no 【决策】 card is made from them.

Method (v2, expert methodology review 2026-11-18):
- Window: from a snapshot date to the first snapshot HORIZON_DAYS calendar days later (≈ 20 trading
  days; the refresh also runs at weekends).
- Names: tradable (US$20M cap, US$25k/day traded) and graded (no Stale / physical / secondary listing)
  at the start. A name that leaves the snapshots inside the window (delisted, acquired) stays in, at
  its last price before the window end - never dropped.
- Forward return net of the metal: return minus beta x the metal's return over the window, beta from
  the name's last BETA_DAYS snapshots against its commodity's price (gold, copper, silver, nickel, zinc,
  uranium); names without a metal keep their raw return.
- IC inside each market x commodity group (>= MIN_GROUP names), averaged weighted by group size: what is
  left is stock selection, not market or metal exposure.
- Q5 - Q1: average metal-net return of the top fifth minus the bottom fifth, by within-group score
  rank; reported gross and net of 50 bp one way on both legs' entry and exit.
- Gate: windows starting on/after SCORER_SINCE and made by the trading score, non-overlapping within a
  market, markets of one start date averaged; needs >= MIN_STARTS start dates, mean IC >= MIN_IC with
  t >= MIN_T, and Q5 - Q1 >= 0 on average. Newey-West t on the weekly (overlapping) series is reported.
- Every run is appended to reports/ic_trials.jsonl with the rules version (for deflated Sharpe).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCORER_SINCE = "2026-11-19"   # trading score (valuation + health) live from the 11-19 refresh
HORIZON_DAYS = 28             # calendar days ≈ 20 trading days
STEP_DAYS = 7                 # a new window every week (for the report; the gate de-overlaps)
BETA_DAYS = 60                # snapshots for each name's metal beta
MIN_NAMES, MIN_GROUP, MIN_BUYS = 15, 8, 3
MIN_STARTS, MIN_IC, MIN_T = 12, 0.03, 2.0
COST = 0.005                  # one way
MATCH_SHARE = 0.9             # a date is "trading score" when this share of composites match the weights
NW_LAGS = 3                   # weekly starts, 4-week windows: 3 overlapping neighbours
NOT_GRADED = "Stale|实物持有|副上市"
FACTORS = ["score_composite", "score_valuation", "score_health", "score_momentum",
           "score_mining", "score_commodity", "score_stage"]
SUB = {"valuation": "score_valuation", "health": "score_health", "momentum": "score_momentum",
       "mining": "score_mining", "commodity": "score_commodity", "stage": "score_stage"}
METALS = ("Gold", "Copper", "Silver", "Nickel", "Zinc", "Uranium")
LIQ_MCAP_USD, LIQ_ADV_USD = 20e6, 25e3
COLUMNS = ["ticker", "snap_date", "price", "market_cap", "avg_turnover", "grade", "commodity"] + FACTORS
RULES_VERSION = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]


def _plus(day: str, days: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=days)).isoformat()


def scorer_dates(snaps: pd.DataFrame, weights: dict[str, float]) -> set[str]:
    """Snapshot dates whose composites are the weighted sum of the sub-scores under `weights`."""
    w = sum(weights.values()) or 1.0
    expect = sum(pd.to_numeric(snaps[c], errors="coerce").fillna(0) * weights.get(k, 0) / w
                 for k, c in SUB.items()).clip(0, 100)
    ok = (pd.to_numeric(snaps["score_composite"], errors="coerce") - expect).abs() <= 0.15
    return {d for d, s in ok.groupby(snaps["snap_date"]) if s.mean() >= MATCH_SHARE}


def metal_of(commodity) -> str | None:
    head = str(commodity or "").split("/")[0].strip()
    return head if head in METALS else None


def _betas(px: pd.DataFrame, metal_px: dict[str, pd.Series], d0: str, names: pd.Series) -> pd.Series:
    """Beta of each name's snapshot-to-snapshot returns on its metal's, over the BETA_DAYS snapshots up to d0."""
    hist = px.loc[:d0].tail(BETA_DAYS + 1).pct_change(fill_method=None).iloc[1:]
    out = {}
    for t, metal in names.items():
        m = metal_px.get(metal)
        if m is None or t not in hist:
            continue
        mr = m.reindex(hist.index).ffill().pct_change(fill_method=None)
        ok = hist[t].notna() & mr.notna()
        if ok.sum() >= 20 and mr[ok].var() > 0:
            out[t] = float(np.cov(hist[t][ok], mr[ok])[0, 1] / mr[ok].var())
    return pd.Series(out, dtype=float)


def _at(m: pd.Series, day: str) -> float:
    """Metal price on `day` or the last one before (index: ISO date strings)."""
    m = m.loc[:day].dropna()
    return float(m.iloc[-1]) if len(m) else np.nan


def _group_ic(score: pd.Series, ret: pd.Series, group: pd.Series) -> float:
    ics, ns = [], []
    for _, idx in group.groupby(group).groups.items():
        s, r = score.reindex(idx), ret.reindex(idx)
        ok = s.notna() & r.notna()
        if ok.sum() >= MIN_GROUP and s[ok].nunique() >= 3:
            ics.append(s[ok].rank().corr(r[ok].rank()))
            ns.append(ok.sum())
    return float(np.average(ics, weights=ns)) if ics else np.nan


def _q5q1(score: pd.Series, ret: pd.Series, group: pd.Series) -> float:
    """Top fifth minus bottom fifth by within-group score percentile."""
    pct = score.groupby(group).rank(pct=True)
    ok = pct.notna() & ret.notna()
    top, bot = ret[ok & (pct > 0.8)], ret[ok & (pct <= 0.2)]
    return float(top.mean() - bot.mean()) if len(top) >= MIN_BUYS and len(bot) >= MIN_BUYS else np.nan


def windows(snaps: pd.DataFrame, fx: float, market: str, weights: dict[str, float] | None = None,
            metal_px: dict[str, pd.Series] | None = None, horizon: int = HORIZON_DAYS,
            step: int = STEP_DAYS) -> list[dict]:
    """One record per window: dates, names, metal-net group IC per factor, Q5-Q1, names priced at
    their last trade (left inside the window), whether the start was made by `weights`."""
    snaps = snaps.drop_duplicates(["ticker", "snap_date"], keep="last")
    metal_px = metal_px or {}
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
        inside = px.loc[d0:d1, x.index]
        end = inside.ffill().iloc[-1]                       # last price up to d1: delisted / acquired stay in
        left = int((inside.loc[d1].isna() & end.notna()).sum()) if d1 in inside.index else 0
        fwd = (end / px.loc[d0, x.index] - 1)
        fwd = fwd[fwd.notna() & (fwd.abs() < 3)]            # broken prices (> +300 %) only
        x = x.loc[fwd.index]
        if len(x) < MIN_NAMES:
            continue
        metal = x["commodity"].map(metal_of)
        beta = _betas(px, metal_px, d0, metal.dropna())
        m_fwd = pd.Series({t: _at(metal_px[m], d1) / _at(metal_px[m], d0) - 1 if m in metal_px else np.nan
                           for t, m in metal.dropna().items()}, dtype=float)
        net = fwd - beta.reindex(fwd.index).fillna(0) * m_fwd.reindex(fwd.index).fillna(0)
        group = x["commodity"].fillna("?").map(lambda c: str(c).split("/")[0])
        rec = dict(market=market, d0=d0, d1=d1, n=len(x), left_in_window=left, live=d0 in live,
                   beta_names=int(beta.notna().sum()))
        for f in FACTORS:
            rec[f] = _group_ic(pd.to_numeric(x[f], errors="coerce"), net, group) if f in x else np.nan
        sc = pd.to_numeric(x["score_composite"], errors="coerce")
        rec["q5q1"] = _q5q1(sc, net, group)
        rec["q5q1_net"] = rec["q5q1"] - 4 * COST if rec["q5q1"] == rec["q5q1"] else np.nan   # in + out, both legs
        buy = x["grade"].fillna("").str.contains("Buy")
        rec["n_buy"] = int(buy.sum())
        rec["buy_excess"] = float((net[buy].mean() - net.mean()) * 100) if buy.sum() >= MIN_BUYS else np.nan
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


def by_date(recs: list[dict], keys=("score_composite", "q5q1", "q5q1_net", "buy_excess")) -> list[dict]:
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


def newey_west_t(values, lags: int = NW_LAGS) -> float:
    """t of the mean with Newey-West (Bartlett) standard errors, for overlapping weekly windows."""
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < lags + 3:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else np.nan


def verdict(recs: list[dict], since: str = SCORER_SINCE) -> dict:
    """Gate on decision cards: validated (bool), the numbers behind it, and the reason in Chinese."""
    win = [r for r in non_overlapping([r for r in recs if r.get("live")], since)
           if not np.isnan(r["score_composite"])]
    dates = by_date(win)
    ic, q = stats(dates, "score_composite"), stats(dates, "q5q1")
    weekly = by_date([r for r in recs if r.get("live") and r["d0"] >= since])
    out = dict(validated=False, since=since, rules_version=RULES_VERSION, windows=len(win), starts=len(dates),
               ic=ic["mean"], t=ic["t"], ic_pos=ic["pos"], q5q1=q["mean"], q5q1_net=stats(dates, "q5q1_net")["mean"],
               nw_t=newey_west_t([r["score_composite"] for r in weekly]),
               buy_excess=stats(dates, "buy_excess")["mean"])
    if len(dates) < MIN_STARTS:
        out["reason"] = (f"样本不够：新评分 {since} 起只有 {len(dates)} 个不重叠的 20 个交易日窗口起点"
                         f"（需要 ≥{MIN_STARTS} 个）")
    elif not (ic["mean"] >= MIN_IC and ic["t"] >= MIN_T):
        out["reason"] = (f"IC 不够：组内扣金属后平均 rank IC {ic['mean']:+.3f}、t {ic['t']:+.2f}"
                         f"（需要 ≥{MIN_IC} 且 t ≥{MIN_T}）")
    elif not (q["n"] and q["mean"] >= 0):
        out["reason"] = (f"Q5 跑输 Q1：平均 {q['mean']:+.2%} / 窗口" if q["n"] else "Q5/Q1 名单太小，算不出")
    else:
        out["validated"] = True
        out["reason"] = (f"通过：{len(dates)} 个起点，组内扣金属 rank IC {ic['mean']:+.3f}、t {ic['t']:+.2f}，"
                         f"Q5−Q1 {q['mean']:+.2%}")
    return out
