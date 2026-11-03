#!/usr/bin/env python3
"""Fig. 1, dataflow: render scripts/paper/dataflow.html (a self-contained, hand-drawn SVG with no
scripts or external resources) to PNG with headless Chromium via Playwright.

Run in a throwaway environment so the project lockfile is untouched, e.g.
  uv run --no-project --with playwright==1.48.0 python scripts/paper/render_dataflow.py \
    --out reports/local_4060_20261101/figures/fig_dataflow.png
(Playwright 1.48.0 pairs with the cached chromium-1140 build.)
"""

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--html", default=str(HERE / "dataflow.html"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=2.0, help="device scale factor")
    a = ap.parse_args(argv)
    html = Path(a.html).read_text()
    if re.search(r"<script|<link|<img|https?://(?!www\.w3\.org/2000/svg)", html):
        raise SystemExit("dataflow.html must be self-contained (no scripts or external resources)")
    m = re.search(r'<svg[^>]*\bwidth="(\d+)"[^>]*\bheight="(\d+)"', html)
    width, height = int(m.group(1)), int(m.group(2))
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": height}, device_scale_factor=a.scale
        )
        page.set_content(html, wait_until="load")
        page.locator("svg").first.screenshot(path=a.out)
        browser.close()
    print(f"FIG_DATAFLOW_OK {a.out} ({width}x{height} css px, scale {a.scale:g})")


if __name__ == "__main__":
    main()
