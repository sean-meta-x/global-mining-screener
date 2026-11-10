"""Station DB from the repo's `db-latest` GitHub Release instead of git.

The daily refresh uploads screener_<market>.db as a release asset. A cloud app
that holds STATION_PAT (Contents: read on this repo) downloads it at start
into config.RELEASE_DB_PATH and re-checks every CHECK_EVERY_S. Without a token,
or when GitHub cannot be reached, the copy already on disk keeps being served.
"""
import logging
import os
import threading
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

REPO = "sean-meta-x/global-mining-screener"
TAG = "db-latest"
RELEASE_URL = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
CHECK_EVERY_S = 1800

_checked_at: dict[Path, float] = {}     # db path -> last time the release was asked
_lock = threading.Lock()


def token(secrets=None) -> str:
    """STATION_PAT from Streamlit secrets (passed in by the app) or the env."""
    try:
        tok = (secrets or {}).get("STATION_PAT", "")
    except Exception:           # st.secrets raises when no secrets file exists
        tok = ""
    return tok or os.environ.get("STATION_PAT", "")


def _marker(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".asset")


def source(db_path) -> str:
    """Where the DB on disk came from: "release <uploaded at>" or "bundled"."""
    marker = _marker(Path(db_path))
    return f"release {marker.read_text()}" if marker.exists() else "bundled"


def sync(db_path, tok: str, get=requests.get) -> str | None:
    """Swap in the release asset named like db_path when it is newer than the
    copy on disk. Returns the asset's upload time when a new copy landed, else
    None — callers must then drop open connections and cached frames."""
    db_path = Path(db_path)
    with _lock:
        if not tok or time.time() - _checked_at.get(db_path, 0) < CHECK_EVERY_S:
            return None
        _checked_at[db_path] = time.time()
    auth = {"Authorization": f"Bearer {tok}"}
    tmp = db_path.with_name(f"{db_path.name}.{os.getpid()}.tmp")
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        r = get(RELEASE_URL, timeout=30,
                headers={**auth, "Accept": "application/vnd.github+json"})
        if r.status_code != 200:
            log.warning("release %s: HTTP %s", TAG, r.status_code)
            return None
        asset = next((a for a in r.json().get("assets", [])
                      if a["name"] == db_path.name), None)
        if asset is None:
            log.warning("release %s has no %s", TAG, db_path.name)
            return None
        stamp = asset["updated_at"]
        marker = _marker(db_path)
        if db_path.exists() and marker.exists() and marker.read_text() == stamp:
            return None
        with get(asset["url"], timeout=300, stream=True,
                 headers={**auth, "Accept": "application/octet-stream"}) as d:
            if d.status_code != 200:
                log.warning("%s download: HTTP %s", db_path.name, d.status_code)
                return None
            with open(tmp, "wb") as f:
                for chunk in d.iter_content(1 << 20):
                    f.write(chunk)
        if tmp.stat().st_size != asset["size"]:
            log.warning("%s download truncated", db_path.name)
            return None
        os.replace(tmp, db_path)    # readers never see a half-written file
        try:
            marker.write_text(stamp)
        except OSError:             # the new DB is in place — callers must still reload
            log.warning("%s: could not record the asset stamp", db_path.name)
        return stamp
    except (requests.RequestException, OSError, ValueError, KeyError) as e:
        log.warning("%s sync failed: %s", db_path.name, e)
        return None
    finally:
        tmp.unlink(missing_ok=True)
