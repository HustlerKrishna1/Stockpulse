"""Capture screenshots of the running StockPulse AI app for the README."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

SHOTS = [
    ("01-terminal.png",   "Chart"),
    ("02-tv-terminal.png","TV Terminal"),
    ("03-verdict.png",    "Verdict"),
    ("04-hf-council.png", "HF Council"),
    ("05-chat.png",       "Chat"),
    ("06-tool-agent.png", "Tool Agent"),
    ("07-strategy.png",   "Strategy Lab"),
    ("08-ml-forecast.png","ML Forecast"),
    ("09-risk.png",       "Risk"),
    ("10-fundamentals.png","Fundamentals"),
    ("11-sec.png",        "SEC Filings"),
    ("12-insiders.png",   "Insiders"),
    ("13-watchlist.png",  "Watchlist"),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000},
                                        device_scale_factor=1.5)
        page = await ctx.new_page()
        await page.goto("http://localhost:8501", wait_until="domcontentloaded", timeout=60_000)
        # Let Streamlit finish rendering
        await page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30_000)
        await page.wait_for_timeout(8000)

        # First shot — landing view
        await page.screenshot(path=str(OUT / SHOTS[0][0]), full_page=False)
        print(f"saved {SHOTS[0][0]}")

        # Iterate tabs by clicking each one
        for fname, tab_label in SHOTS[1:]:
            try:
                # Tabs are buttons with role=tab
                tab = page.get_by_role("tab", name=tab_label).first
                await tab.click(timeout=5000)
                await page.wait_for_timeout(3500)
                await page.screenshot(path=str(OUT / fname), full_page=False)
                print(f"saved {fname}")
            except Exception as e:
                print(f"skip {fname}: {e}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
