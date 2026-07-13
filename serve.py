"""
Mining Screener launcher (set MARKET env first).
Run:  python serve.py
Opens on http://localhost:8502
"""
import os, sys, asyncio

# Required for Tornado/uvicorn compatibility on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

# Local port/address (moved out of .streamlit/config.toml so that
# Streamlit Cloud can use its required default port 8501)
os.environ.setdefault("STREAMLIT_SERVER_PORT", "8502")
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Config is read from .streamlit/config.toml (useStarlette=true, port=8502)
from streamlit.web.bootstrap import run

print("Mining Screener starting on http://localhost:8502 ...", flush=True)
run(
    main_script_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    is_hello=False,
    args=[],
    flag_options={},
)
