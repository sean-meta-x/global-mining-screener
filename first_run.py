"""Bootstrap: fetch all tickers and seed the database."""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

from scheduler.jobs import run_daily_refresh
run_daily_refresh()
print("first_run complete.", flush=True)
