"""
AI overlay — the qualitative layer on top of the quant screen.

Two stages, both producing auditable JSON committed to the repo:

  outlook            One global call. Sets COMMODITY_OUTLOOK multipliers from
                     recent spot moves + macro context -> ai_outlook.json
                     (clamped 0.7-1.3; applied pre-scoring by scheduler.jobs).

  review --market m  Reviews the top-N quant candidates of one market against
                     recent headlines -> ai_overlay_{m}.json with per-ticker
                     {ai_view, ai_adjust (-10..+10), ai_note}. The app shows
                     an AI-adjusted score = composite + ai_adjust.

Provider-agnostic: any OpenAI-compatible endpoint (default: Bailian workspace,
model qwen3.7-plus). Switch to Anthropic by setting AI_PROVIDER=anthropic and
AI_API_KEY (uses the Messages API). Every failure degrades to "no overlay" —
the quant pipeline never depends on this script succeeding.
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Provider ──────────────────────────────────────────────────────────────────
PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()      # openai-compatible | anthropic
API_KEY  = os.getenv("AI_API_KEY", "")
BASE_URL = os.getenv(
    "AI_BASE_URL",
    "https://llm-hn7t2xu3nkwu7buk.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)
MODEL    = os.getenv("AI_MODEL", "qwen3.7-plus")

TOP_N          = int(os.getenv("AI_REVIEW_TOP_N", "20"))
ADJUST_LIMIT   = 10
MULT_LO, MULT_HI = 0.7, 1.3
VALID_VIEWS    = {"bullish", "neutral", "caution", "red_flag"}


def _chat(system: str, user: str, max_tokens: int = 3000) -> str:
    """One chat call, 2 retries. Returns assistant text ('' on failure)."""
    for attempt in range(3):
        try:
            if PROVIDER == "anthropic":
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=json.dumps({
                        "model": MODEL, "max_tokens": max_tokens,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                    }).encode(),
                    headers={"x-api-key": API_KEY,
                             "anthropic-version": "2023-06-01",
                             "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=120))
                return r["content"][0]["text"]
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions",
                data=json.dumps({
                    "model": MODEL, "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                }).encode(),
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"]
        except Exception as e:                                # noqa: BLE001
            print(f"[ai] attempt {attempt+1} failed: {e}", file=sys.stderr)
    return ""


def _extract_json(text: str):
    """Parse the first JSON object/array in a model reply."""
    m = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.S)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        raw = m.group(0) if m else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


# ── Stage 1: commodity outlook ────────────────────────────────────────────────

def run_outlook() -> None:
    import config

    # 30-day spot change from any market DB that has history
    changes = {}
    for db in sorted(ROOT.glob("screener_*.db")):
        try:
            c = sqlite3.connect(db)
            rows = c.execute(
                "SELECT commodity, price, price_date FROM commodity_prices "
                "ORDER BY price_date").fetchall()
            c.close()
            hist: dict[str, list] = {}
            for comm, price, d in rows:
                hist.setdefault(comm, []).append((d, price))
            for comm, seq in hist.items():
                if len(seq) >= 2 and seq[-1][1] and seq[0][1]:
                    changes[comm] = round((seq[-1][1] / seq[0][1] - 1) * 100, 1)
            if changes:
                break
        except Exception:                                     # noqa: BLE001
            continue

    facts = "\n".join(
        f"- {c}: spot {config.COMMODITY_SPOT.get(c, 'n/a')} USD"
        + (f", {changes[c]:+.1f}% over stored history" if c in changes else "")
        for c in config.COMMODITY_OUTLOOK
    )

    system = (
        "You are the commodity strategist of a mining-equities fund. "
        "Output ONLY a JSON object, no prose.")
    user = f"""Set today's 12-month outlook multiplier for each commodity below.
1.00 = neutral, above 1 bullish, below 1 bearish. Stay within {MULT_LO}-{MULT_HI}.
Base your view on the supplied spot levels/changes plus your knowledge of supply,
demand, inventories and the macro cycle. Be decisive but not extreme.

Current data:
{facts}

Return JSON exactly like:
{{"Gold": {{"multiplier": 1.15, "note": "one short reason"}}, ...}}
Cover every commodity listed above."""

    reply = _chat(system, user)
    data = _extract_json(reply) or {}
    outlook = {}
    for comm in config.COMMODITY_OUTLOOK:
        item = data.get(comm) or {}
        try:
            mult = float(item.get("multiplier"))
        except (TypeError, ValueError):
            continue
        outlook[comm] = {
            "multiplier": round(min(MULT_HI, max(MULT_LO, mult)), 2),
            "note": str(item.get("note", ""))[:160],
        }

    if not outlook:
        print("[ai] outlook: no valid output — keeping static config")
        return
    payload = {"generated_at": _now(), "model": MODEL, "outlook": outlook}
    (ROOT / "ai_outlook.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[ai] outlook written for {len(outlook)} commodities")


# ── Stage 2: top-N review per market ──────────────────────────────────────────

def _headlines(ticker: str, limit: int = 4) -> list[str]:
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        out = []
        for n in news[:limit]:
            content = n.get("content") or n
            title = content.get("title") or ""
            when = (content.get("pubDate") or "")[:10]
            if title:
                out.append(f"{when} {title}".strip())
        return out
    except Exception:                                         # noqa: BLE001
        return []


def run_review(market: str) -> None:
    db = ROOT / f"screener_{market}.db"
    if not db.exists():
        print(f"[ai] {db.name} missing — skip")
        return
    c = sqlite3.connect(db)
    rows = c.execute(
        "SELECT ticker, name, commodity, stage, score_composite, "
        "market_cap, price, return_1m "
        "FROM stock_snapshots WHERE snap_date=(SELECT MAX(snap_date) FROM stock_snapshots) "
        "ORDER BY score_composite DESC LIMIT ?", (TOP_N,)).fetchall()
    c.close()
    if not rows:
        print(f"[ai] {market}: no snapshot rows — skip")
        return

    lines, tickers = [], set()
    for tk, name, comm, stage, score, mcap, price, ret1m in rows:
        tickers.add(tk)
        hl = _headlines(tk)
        lines.append(
            f"### {tk} — {name} ({comm}, {stage})\n"
            f"quant score {score:.0f}/100"
            + (f", 1M return {ret1m:+.1f}%" if ret1m is not None else "")
            + (f", mkt cap {mcap:,.0f}" if mcap else "")
            + ("\nheadlines:\n" + "\n".join(f"  - {h}" for h in hl) if hl
               else "\nheadlines: none found"))

    system = (
        "You are the senior analyst of a mining-equities fund doing a daily "
        "sanity check on a quantitative screen. Judge what the quant model "
        "cannot see: dilution, failed drilling, permits, governance, promotion. "
        "Output ONLY a JSON array, no prose.")
    user = f"""Review each candidate below. For each, return:
- "ticker"
- "ai_view": one of "bullish" | "neutral" | "caution" | "red_flag"
- "ai_adjust": integer -{ADJUST_LIMIT}..+{ADJUST_LIMIT} to add to the quant score
  (0 if nothing notable; negative for risks the quant score misses; positive
  only for concrete catalysts)
- "ai_note": ONE short sentence in Chinese (≤60 chars) justifying it

Candidates:
{chr(10).join(lines)}

Return a JSON array covering every ticker."""

    data = []
    for _parse_try in range(2):          # retry once if the reply isn't valid JSON
        reply = _chat(system, user, max_tokens=4000)
        data = _extract_json(reply) or []
        if data:
            break
    items = []
    for it in data if isinstance(data, list) else []:
        tk = str(it.get("ticker", ""))
        if tk not in tickers:
            continue
        view = str(it.get("ai_view", "neutral")).lower()
        try:
            adj = int(it.get("ai_adjust", 0))
        except (TypeError, ValueError):
            adj = 0
        items.append({
            "ticker": tk,
            "ai_view": view if view in VALID_VIEWS else "neutral",
            "ai_adjust": max(-ADJUST_LIMIT, min(ADJUST_LIMIT, adj)),
            "ai_note": str(it.get("ai_note", ""))[:160],
        })

    if not items:
        print(f"[ai] {market}: no valid review output — skip")
        return
    payload = {"generated_at": _now(), "model": MODEL,
               "market": market, "items": items}
    (ROOT / f"ai_overlay_{market}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[ai] {market}: reviewed {len(items)}/{len(rows)} candidates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("outlook")
    pr = sub.add_parser("review")
    pr.add_argument("--market", required=True)
    args = ap.parse_args()
    if not API_KEY:
        print("[ai] AI_API_KEY not set — overlay skipped")
        sys.exit(0)
    if args.cmd == "outlook":
        run_outlook()
    else:
        run_review(args.market)
