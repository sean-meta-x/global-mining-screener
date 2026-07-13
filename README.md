# Global Mining Stock Screener

Multi-market mining stock undervaluation screener (Streamlit). One codebase,
one deployment per market — the market is selected with the `MARKET` env var.

| Market | Code | Universe | Exchanges |
|---|---|---|---|
| Canada | `ca` | ~1,456 | TSX · TSXV · CSE |
| United States | `us` | ~274 | NYSE · NASDAQ |
| United Kingdom | `uk` | ~154 | LSE · AIM |
| South Africa | `za` | ~36 | JSE |
| Indonesia | `id` | ~56 | IDX |
| Hong Kong | `hk` | ~78 | HKEX |
| China A-shares | `cn` | ~165 | SSE · SZSE |

Sister project: the (private) Australia screener this codebase derives from.

## How it works

- **Universe**: full Yahoo-screener sweep of every Basic Materials mining
  industry per region (`scripts/build_universe.py`), merged with hand-curated
  lists (`scripts/merge_curated.py`). Curated rows carry trusted
  commodity/stage labels; screen-sourced rows get revenue-based stage
  refinement during the daily refresh.
- **Scoring**: multi-factor undervaluation score (valuation 30%, balance-sheet
  health 20%, momentum 15%, mining metrics 25%, commodity outlook 5%, stage 5%).
- **Refresh**: GitHub Actions daily at 21:30 UTC (matrix over all markets),
  commits each `screener_{market}.db` back → Streamlit Cloud auto-redeploys.
- **Optional CIQ overlay**: drop a `spg_{market}.json` (S&P Capital IQ Pro
  Excel add-in export, same schema as the AU screener) into the repo root to
  light up P/NAV, AISC and reserve metrics; `commodity_spots.json` overrides
  spot prices.

## Run locally

```bash
pip install -r requirements.txt
MARKET=ca python serve.py     # http://localhost:8502
```

## Deploy (Streamlit Community Cloud)

Main file `app.py`, Python 3.12, secrets:

```toml
MARKET = "ca"
DISABLE_SCHEDULER = "true"
```
