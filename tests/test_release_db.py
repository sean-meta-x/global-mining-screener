"""Station DB served from the `db-latest` GitHub Release (stage 5: big DBs out of git).

sync() must only ever swap in a complete, newer copy: no token, GitHub down,
a missing asset or a short download all leave the DB on disk untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import requests

import data.release_db as release_db


class _Resp:
    def __init__(self, status=200, payload=None, body=b""):
        self.status_code, self._payload, self._body = status, payload, body

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _github(body=b"new-db", stamp="2026-11-10T22:00:00Z", name="screener_ca.db",
            size=None, release_status=200, asset_status=200):
    """A fake requests.get: the release JSON first, then the asset bytes."""
    calls = []

    def get(url, **kw):
        calls.append((url, kw))
        if url.endswith("/releases/tags/db-latest"):
            return _Resp(release_status, {"assets": [{
                "name": name, "updated_at": stamp, "url": "https://api/asset/1",
                "size": len(body) if size is None else size}]})
        return _Resp(asset_status, body=body)

    get.calls = calls
    return get


@pytest.fixture(autouse=True)
def _fresh_throttle():
    release_db._checked_at.clear()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "screener_ca.db"
    path.write_bytes(b"bundled")
    return path


def test_no_token_makes_no_request_and_keeps_the_db(db):
    get = _github()
    assert release_db.sync(db, "", get=get) is None
    assert get.calls == [] and db.read_bytes() == b"bundled"


def test_new_asset_replaces_the_db_and_records_its_stamp(db):
    get = _github()
    assert release_db.sync(db, "tok", get=get) == "2026-11-10T22:00:00Z"
    assert db.read_bytes() == b"new-db"
    assert release_db.source(db) == "release 2026-11-10T22:00:00Z"
    release_hdr, asset_hdr = (kw["headers"] for _, kw in get.calls)
    assert release_hdr["Authorization"] == "Bearer tok"
    assert asset_hdr["Accept"] == "application/octet-stream"


def test_missing_db_is_downloaded(tmp_path):
    path = tmp_path / "screener_ca.db"
    assert release_db.sync(path, "tok", get=_github()) and path.read_bytes() == b"new-db"


def test_same_stamp_is_not_downloaded_again(db):
    release_db.sync(db, "tok", get=_github())
    release_db._checked_at.clear()
    get = _github(body=b"should-not-land")
    assert release_db.sync(db, "tok", get=get) is None
    assert len(get.calls) == 1 and db.read_bytes() == b"new-db"


def test_checks_are_throttled(db):
    release_db.sync(db, "tok", get=_github())
    get = _github(stamp="2026-11-11T22:00:00Z")
    assert release_db.sync(db, "tok", get=get) is None
    assert get.calls == []


@pytest.mark.parametrize("github", [
    _github(release_status=404),                # release missing / token lacks access
    _github(name="screener_us.db"),             # asset mid-clobber or not uploaded yet
    _github(asset_status=502),                  # download failed
    _github(size=999),                          # truncated download
])
def test_failures_keep_the_db_on_disk(db, github):
    assert release_db.sync(db, "tok", get=github) is None
    assert db.read_bytes() == b"bundled"
    assert release_db.source(db) == "bundled"
    assert list(db.parent.glob("*.tmp")) == []


def test_network_error_keeps_the_db_on_disk(db):
    def get(url, **kw):
        raise requests.ConnectionError("down")
    assert release_db.sync(db, "tok", get=get) is None
    assert db.read_bytes() == b"bundled"


def test_token_prefers_secrets_then_env(monkeypatch):
    monkeypatch.setenv("STATION_PAT", "from-env")
    assert release_db.token({"STATION_PAT": "from-secrets"}) == "from-secrets"
    assert release_db.token({}) == "from-env"

    class _NoSecrets:                       # st.secrets raises without a secrets file
        def get(self, *a):
            raise FileNotFoundError
    assert release_db.token(_NoSecrets()) == "from-env"
