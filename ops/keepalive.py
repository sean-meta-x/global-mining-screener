"""Wake the fund's Streamlit sites before people open them (a sleeping app takes 60-70 s).

Streamlit Community Cloud puts an app to sleep after a stretch without visitors; the
sleeping page has a "Yes, get this app back up!" button, and a plain HTTP request does not
press it. So this opens each site in headless Chrome (the one preinstalled on GitHub's
Ubuntu runner — nothing is downloaded), presses the button when it is there, and moves on
without waiting for the app to finish booting. Read-only: it only loads public pages.

    python ops/keepalive.py            # all sites
    python ops/keepalive.py URL ...    # just these
Exits 0 even when a site is down: this is a convenience, not a health check.
"""
import sys
import time

SITES = [
    "https://global-mining-fund.streamlit.app/",
    "https://meta-research-lab.streamlit.app/",
    "https://meta-mining-screener.streamlit.app/",
    *[f"https://{m}-mining-screener.streamlit.app/" for m in ("ca", "us", "uk", "za", "id", "hk", "cn")],
]
WAKE_BUTTON = "Yes, get this app back up!"
LOOK_SECONDS = 20          # long enough to see either the app or the sleeping page


def visit(page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    deadline = time.time() + LOOK_SECONDS
    while time.time() < deadline:
        for frame in page.frames:
            try:
                button = frame.get_by_role("button", name=WAKE_BUTTON)
                if button.count():
                    button.first.click()
                    return "was asleep — woken"
            except Exception:
                pass
        try:
            text = " ".join(f.inner_text("body") for f in page.frames)
        except Exception:
            text = ""
        if "do not have access" in text:
            return "not public (sign-in page)"
        if len(text) > 300:
            return "awake"
        time.sleep(2)
    return f"no answer within {LOOK_SECONDS} s (booting?)"


def main(urls: list[str]) -> int:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for url in urls:
            page = browser.new_page()
            try:
                print(f"{url}: {visit(page, url)}", flush=True)
            except Exception as e:
                print(f"{url}: failed — {type(e).__name__}: {e}", flush=True)
            finally:
                page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or SITES))
