"""
Merge hand-curated universes (from the old per-market screener folders) into the
Yahoo-screener sweep JSONs. Curated rows win on name/commodity/stage.

Each company gets "src": "curated" | "screen" — refine_stages() only touches
screen-sourced rows (curated stages are trusted).

Usage: python scripts/merge_curated.py
"""
import importlib.util
import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent.parent   # ...\Documents
ROOT = Path(__file__).resolve().parent.parent

CURATED_DIRS = {
    "ca": "canada-mining-screener",
    "us": "usa-mining-screener",
    "uk": "uk-mining-screener",
    "za": "south-africa-mining-screener",
    "id": "indonesia-mining-screener",
    "hk": "hk-mining-screener",
    "cn": "china-mining-screener",
}


def load_curated(folder: str) -> list[tuple]:
    path = DOCS / folder / "data" / "universe.py"
    if not path.exists():
        return []
    spec = importlib.util.spec_from_file_location(f"uni_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    return list(getattr(mod, "MINING_UNIVERSE", []))


def main() -> None:
    for mkt, folder in CURATED_DIRS.items():
        sweep_path = ROOT / "universes" / f"{mkt}.json"
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        by_ticker = {c["ticker"]: {**c, "src": "screen"} for c in sweep["companies"]}

        curated = load_curated(folder)
        n_new, n_upd = 0, 0
        for row in curated:
            tk, name, commodity, stage = row[0], row[1], row[2], row[3]
            if tk in by_ticker:
                by_ticker[tk].update(
                    name=name, commodity=commodity, stage=stage, src="curated")
                n_upd += 1
            else:
                by_ticker[tk] = {"ticker": tk, "name": name, "commodity": commodity,
                                 "stage": stage, "src": "curated"}
                n_new += 1

        sweep["companies"] = sorted(by_ticker.values(), key=lambda c: c["ticker"])
        sweep["n"] = len(sweep["companies"])
        sweep_path.write_text(
            json.dumps(sweep, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{mkt}] total {sweep['n']:>4}  "
              f"(curated matched {n_upd}, curated-only added {n_new}, "
              f"screen-only {sweep['n'] - n_upd - n_new})")


if __name__ == "__main__":
    main()
