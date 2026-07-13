"""Cloud entrypoint — uk market. (Local dev: use serve.py with MARKET env.)"""
import os

os.environ["MARKET"] = "uk"
# Cloud containers are ephemeral; refresh runs in GitHub Actions instead.
os.environ.setdefault("DISABLE_SCHEDULER", "true")

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
with open(_APP, encoding="utf-8") as _f:
    exec(compile(_f.read(), _APP, "exec"))
