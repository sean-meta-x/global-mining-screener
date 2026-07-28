"""Tests for the 2026-07-28 data-quality port from the AU station (#87 / #104).

Covers:
  1. Stale penalty needs corroboration: frozen price + healthy turnover is a
     quantised low-priced stock, not a halt.
  2. The 10-snapshot hard rule catches a halt whose trailing volume field is
     still fat (Yahoo's averageVolume is a 3-month figure and lies for weeks
     after a suspension).
  3. Scale-break quarantine marks a magnitude jump (rand/cent flips, unadjusted
     consolidations) and is idempotent.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest


def _station_db(tmp_path, monkeypatch):
    """A tiny stock_snapshots DB, with data.database._engine pointed at it."""
    import data.database as database
    from sqlalchemy import create_engine

    db = tmp_path / "station.db"
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE stock_snapshots (
            ticker TEXT, snap_date TEXT, price FLOAT, avg_turnover FLOAT,
            score_composite FLOAT, grade TEXT,
            UNIQUE (ticker, snap_date)
        )
    """)
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{db}")
    monkeypatch.setattr(database, "_engine", lambda: engine)
    return engine


def _insert(engine, ticker, days, price, turnover, score=80.0):
    prices = price if isinstance(price, list) else [price] * len(days)
    with engine.begin() as conn:
        from sqlalchemy import text
        for day, px in zip(days, prices):
            conn.execute(text(
                "INSERT INTO stock_snapshots "
                "(ticker, snap_date, price, avg_turnover, score_composite, grade) "
                "VALUES (:t, :d, :p, :v, :s, '🔵 Buy')"),
                {"t": ticker, "d": day, "p": px, "v": turnover, "s": score})


DAYS10 = [f"2026-07-{d:02d}" for d in range(16, 26)]
LATEST = DAYS10[-1]


@pytest.mark.unit
def test_frozen_with_healthy_turnover_is_not_stale(tmp_path, monkeypatch):
    engine = _station_db(tmp_path, monkeypatch)
    _insert(engine, "FLAT.JO", DAYS10[-5:], price=0.265, turnover=79_170.0)

    from scheduler.jobs import apply_stale_price_penalty

    assert apply_stale_price_penalty(LATEST) == []


@pytest.mark.unit
def test_frozen_with_no_turnover_is_stale(tmp_path, monkeypatch):
    engine = _station_db(tmp_path, monkeypatch)
    _insert(engine, "DEAD.L", DAYS10[-5:], price=0.10, turnover=0.0)

    from scheduler.jobs import STALE_GRADE, apply_stale_price_penalty

    assert apply_stale_price_penalty(LATEST) == ["DEAD.L"]
    row = pd.read_sql("SELECT grade, score_composite FROM stock_snapshots "
                      f"WHERE ticker='DEAD.L' AND snap_date='{LATEST}'", engine)
    assert row["grade"].iloc[0] == STALE_GRADE
    assert row["score_composite"].iloc[0] <= 40.0


@pytest.mark.unit
def test_long_freeze_is_stale_even_with_fat_volume_field(tmp_path, monkeypatch):
    engine = _station_db(tmp_path, monkeypatch)
    _insert(engine, "HALT.HK", DAYS10, price=0.66, turnover=2_000_000.0)

    from scheduler.jobs import apply_stale_price_penalty

    assert apply_stale_price_penalty(LATEST) == ["HALT.HK"]


@pytest.mark.unit
def test_scale_break_quarantined_and_idempotent(tmp_path, monkeypatch):
    engine = _station_db(tmp_path, monkeypatch)
    _insert(engine, "FLIP.JO", DAYS10[:4], price=[1.0, 1.0, 100.0, 100.0],
            turnover=50_000.0)
    _insert(engine, "OK.TO", DAYS10[:4], price=[1.0, 1.02, 1.01, 1.05],
            turnover=50_000.0)

    from scheduler.jobs import SCALE_BREAK_GRADE, apply_scale_break_quarantine

    first = apply_scale_break_quarantine()
    second = apply_scale_break_quarantine()

    assert [(t, d) for t, d, _ in first] == [("FLIP.JO", DAYS10[2])]
    assert first == second
    marked = pd.read_sql("SELECT grade, score_composite FROM stock_snapshots "
                         f"WHERE ticker='FLIP.JO' AND snap_date='{DAYS10[2]}'",
                         engine)
    assert marked["grade"].iloc[0] == SCALE_BREAK_GRADE
    assert marked["score_composite"].iloc[0] <= 40.0
    ok = pd.read_sql("SELECT DISTINCT grade FROM stock_snapshots "
                     "WHERE ticker='OK.TO'", engine)
    assert ok["grade"].tolist() == ["🔵 Buy"]
