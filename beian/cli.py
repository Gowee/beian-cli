"""CLI tool for querying ICP registration info from MIIT (beian.miit.gov.cn)."""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor

from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

# ---------------------------------------------------------------------------
# Gap detection algorithm – runs inside the browser page context via
# page.evaluate().  Reads bgImg + sildeImg from the DOM, computes per-column
# mean RGB, and finds the gap via max pairwise column color distance in a
# 65 px sliding window.  Returns the displacement (px) for the slider.
# ---------------------------------------------------------------------------

DETECT_GAP_JS = """\
async () => {
    const bgImg = document.getElementById('bgImg');
    const sildeImg = document.getElementById('sildeImg');
    if (!bgImg || !sildeImg) return {error: 'CAPTCHA images not loaded'};

    // Wait for both images to decode (no fixed sleep)
    for (let i = 0; i < 50; i++) {
        if (bgImg.complete && bgImg.naturalWidth > 0 &&
            sildeImg.complete && sildeImg.naturalWidth > 0) break;
        await new Promise(r => setTimeout(r, 100));
    }
    if (!bgImg.naturalWidth || !sildeImg.naturalWidth)
        return {error: 'CAPTCHA images failed to load'};

    const c = new OffscreenCanvas(bgImg.naturalWidth, bgImg.naturalHeight);
    const ctx = c.getContext('2d');
    ctx.drawImage(bgImg, 0, 0);
    const bgD = ctx.getImageData(0, 0, c.width, c.height);

    const sc = new OffscreenCanvas(sildeImg.naturalWidth, sildeImg.naturalHeight);
    const sctx = sc.getContext('2d');
    sctx.drawImage(sildeImg, 0, 0);
    const sD = sctx.getImageData(0, 0, sc.width, sc.height);

    const W = c.width, H = c.height, sW = sc.width, sH = sc.height;

    // Puzzle piece alpha mask
    const mask = [];
    for (let y = 0; y < sH; y++)
        for (let x = 0; x < sW; x++)
            if (sD.data[(y * sW + x) * 4 + 3] > 128) mask.push({x, y});
    if (!mask.length) return {error: 'No puzzle piece found in alpha mask'};

    const pMinX = Math.min(...mask.map(p => p.x));
    const pMaxX = Math.max(...mask.map(p => p.x));
    const pCX = Math.round((pMinX + pMaxX) / 2);

    // Per-column mean RGB (vertical centre 70%)
    const yStart = Math.floor(H * 0.15);
    const yEnd = Math.floor(H * 0.85);
    const colC = [];
    for (let x = 0; x < W; x++) {
        let rS = 0, gS = 0, bS = 0;
        for (let y = yStart; y < yEnd; y++) {
            const i = (y * W + x) * 4;
            rS += bgD.data[i];
            gS += bgD.data[i + 1];
            bS += bgD.data[i + 2];
        }
        const n = yEnd - yStart;
        colC.push({x, r: rS / n, g: gS / n, b: bS / n});
    }

    // Max pairwise column colour distance in sliding window
    const win = 65;
    const cands = [];
    for (let s = pMaxX + 5; s <= W - win; s++) {
        const cols = colC.slice(s, s + win);
        let maxDist = 0;
        for (let i = 0; i < cols.length; i++) {
            for (let j = i + 1; j < cols.length; j++) {
                const d = Math.abs(cols[i].r - cols[j].r)
                        + Math.abs(cols[i].g - cols[j].g)
                        + Math.abs(cols[i].b - cols[j].b);
                if (d > maxDist) maxDist = d;
            }
        }
        const leftC  = colC.slice(Math.max(0, s - 40), s);
        const rightC = colC.slice(s + win, Math.min(W, s + win + 40));
        const nbrs   = [...leftC, ...rightC];
        if (!nbrs.length) continue;
        const nR = nbrs.reduce((a, c) => a + c.r, 0) / nbrs.length;
        const nG = nbrs.reduce((a, c) => a + c.g, 0) / nbrs.length;
        const nB = nbrs.reduce((a, c) => a + c.b, 0) / nbrs.length;
        const wR = cols.reduce((a, c) => a + c.r, 0) / cols.length;
        const wG = cols.reduce((a, c) => a + c.g, 0) / cols.length;
        const wB = cols.reduce((a, c) => a + c.b, 0) / cols.length;
        const abnormal = Math.abs(wR - nR) + Math.abs(wG - nG) + Math.abs(wB - nB);
        cands.push({s, score: abnormal / (maxDist + 1)});
    }
    cands.sort((a, b) => b.score - a.score);
    if (!cands.length) return {error: 'No gap candidates found'};

    const gapCenter = cands[0].s + Math.round(win / 2);
    const displacement = gapCenter - pCX;
    return {displacement, gapCenter, puzzleCenterX: pCX, imgWidth: W};
}
"""

# JS that waits for CAPTCHA images to fully decode (event-based, no sleep)
WAIT_CAPTCHA_READY_JS = """\
async () => {
    const bgImg = document.getElementById('bgImg');
    const sildeImg = document.getElementById('sildeImg');
    if (!bgImg || !sildeImg) return false;
    // Wait until images have valid dimensions (decoded)
    for (let i = 0; i < 80; i++) {
        if (bgImg.complete && bgImg.naturalWidth > 0 &&
            sildeImg.complete && sildeImg.naturalWidth > 0) return true;
        await new Promise(r => setTimeout(r, 100));
    }
    return false;
}
"""

# ---------------------------------------------------------------------------
# Query type mapping
# ---------------------------------------------------------------------------
QUERY_TYPES = {
    "website":  1,   # 网站
    "app":      6,   # APP
    "miniprogram": 7, # 小程序
    "quickapp": 8,   # 快应用
}
QUERY_TYPE_LABELS = {
    1: "网站",
    6: "APP",
    7: "小程序",
    8: "快应用",
}

# ---------------------------------------------------------------------------
# Core automation
# ---------------------------------------------------------------------------

def query_beian(query: str, *, headless: bool = True, timeout_ms: int = 30_000,
                retries: int = 3, query_type: str = "website") -> dict:
    """Query MIIT for a single term. Returns one result dict."""
    results = query_beian_batch(
        [query], headless=headless, timeout_ms=timeout_ms,
        retries=retries, query_type=query_type,
    )
    return results[0]


def query_beian_batch(queries: list[str], *, headless: bool = True,
                      timeout_ms: int = 30_000, retries: int = 3,
                      query_type: str = "website",
                      screenshot_dir: str | None = None) -> list[dict]:
    """Query MIIT for multiple terms in one browser session."""
    service_type = QUERY_TYPES.get(query_type)
    if service_type is None:
        raise ValueError(f"Unknown query type: {query_type!r}. "
                         f"Choose from: {', '.join(QUERY_TYPES)}")

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-font-subpixel-positioning",
                "--disable-extensions",
            ],
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)

        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)
        page.route("**/*.{woff,woff2,ttf,otf}", lambda route: route.abort())
        page.route("**/analytics**", lambda route: route.abort())
        page.route("**/gtm**", lambda route: route.abort())

        try:
            for i, q in enumerate(queries):
                if i > 0:
                    # Navigate back to index for next query
                    page.goto("https://beian.miit.gov.cn/#/Integrated/recordQuery",
                              wait_until="networkidle")
                    page.wait_for_function(
                        """() => {
                            const el = document.getElementById('app');
                            if (!el || !el.__vue__) return false;
                            return !!el.__vue__.$children.find(
                                c => c.$el.classList.contains('Integrated'));
                        }""",
                        timeout=15_000,
                    )
                results.append(_do_query(page, q, retries, service_type,
                                         screenshot_dir=screenshot_dir))
        finally:
            ctx.close()
            browser.close()

    return results


def _do_query(page, query: str, retries: int, service_type: int,
              *, screenshot_dir: str | None = None) -> dict:
    # 1. Load the site – networkidle is reliable for this SPA
    page.goto("https://beian.miit.gov.cn/#/Integrated/recordQuery",
              wait_until="networkidle")

    # 1b. Wait for Vue to initialise
    page.wait_for_function(
        """() => {
            const el = document.getElementById('app');
            if (!el || !el.__vue__) return false;
            const integrated = el.__vue__.$children.find(
                c => c.$el.classList.contains('Integrated'));
            return !!integrated;
        }""",
        timeout=15_000,
    )

    # 2. Type query, set radio, and trigger search
    page.evaluate("""([query, serviceType]) => {
        const app = document.getElementById('app').__vue__;
        const integrated = app.$children.find(
            c => c.$el.classList.contains('Integrated'));
        integrated.inputname = query;
        integrated.searchRadioFlag = serviceType;
        integrated.searchA();
    }""", [query, service_type])

    # 4. Wait for CAPTCHA images to be ready
    page.wait_for_selector("#bgImg", state="attached", timeout=10_000)
    page.wait_for_function(WAIT_CAPTCHA_READY_JS, timeout=10_000)

    # 5. Solve CAPTCHA (with retries)
    for attempt in range(1, retries + 1):
        result = page.evaluate(DETECT_GAP_JS)
        if "error" in result:
            if attempt < retries:
                _refresh_captcha(page)
                continue
            return {"error": result["error"]}

        displacement = result["displacement"]

        def _match_check(r):
            return "image/checkImage" in r.url

        def _match_search(r):
            return ("icpAbbreviateInfo" in r.url
                    or "blackListDomain" in r.url)

        try:
            # Set up BOTH expect_response before triggering checkImg
            # checkImage response arrives first, then search API
            with page.expect_response(_match_check, timeout=5_000) as check_info:
                with page.expect_response(_match_search, timeout=8_000) as search_info:
                    page.evaluate("""(disp) => {
                        const app = document.getElementById('app').__vue__;
                        const integrated = app.$children.find(
                            c => c.$el.classList.contains('Integrated'));
                        integrated.puzzle = disp;
                        integrated.checkImg();
                        return true;
                    }""", displacement)

            # checkImage response — verify success
            check_body = check_info.value.json()
            if not check_body.get("success", False):
                # Wrong answer — fail fast
                if attempt < retries:
                    _wait_for_captcha_refresh(page)
                continue

            # CAPTCHA passed — search response captured
            result = _parse_api_response(search_info.value.json(), query, service_type)
            if screenshot_dir:
                _take_screenshot(page, query, screenshot_dir)
            return result

        except PwTimeout:
            if attempt < retries:
                _wait_for_captcha_refresh(page)
                continue
        except Exception:
            if attempt < retries:
                _wait_for_captcha_refresh(page)
                continue

    return {"error": f"Failed after {retries} attempts", "query": query}


def _sanitize_filename(query: str) -> str:
    """Convert query string to a safe filename."""
    name = re.sub(r"[^\w\u4e00-\u9fff]+", "_", query).strip("_")
    return name or "result"


def _take_screenshot(page, query: str, screenshot_dir: str):
    """Take a full-page screenshot and save to directory."""
    os.makedirs(screenshot_dir, exist_ok=True)
    path = os.path.join(screenshot_dir, f"{_sanitize_filename(query)}.png")
    page.screenshot(path=path, full_page=True)
    print(f"Screenshot: {path}")


def _refresh_captcha(page):
    """Refresh CAPTCHA and wait for new images (event-based)."""
    page.evaluate("""() => {
        const app = document.getElementById('app').__vue__;
        const integrated = app.$children.find(
            c => c.$el.classList.contains('Integrated'));
        integrated.getImg();
    }""")
    page.wait_for_function(WAIT_CAPTCHA_READY_JS, timeout=5_000)


def _wait_for_captcha_refresh(page):
    """After failed checkImg, Vue auto-refreshes CAPTCHA – wait for it."""
    try:
        page.wait_for_function(WAIT_CAPTCHA_READY_JS, timeout=5_000)
    except PwTimeout:
        pass


def _parse_api_response(resp: dict, query: str, service_type: int) -> dict:
    """Parse the raw API JSON response into our output format."""
    params = resp.get("params") or {}

    # Website type: params.list is the record array
    records = params.get("list") or []

    # Normalise fields
    normalised = []
    for r in records:
        normalised.append({
            "unitName":       r.get("unitName", ""),
            "nature":         r.get("natureName", ""),
            "serviceLicence": r.get("serviceLicence", ""),
            "mainLicence":    r.get("mainLicence", ""),
            "domain":         r.get("domain", ""),
            "updateDate":     r.get("updateRecordTime", ""),
        })

    return {
        "query": query,
        "queryType": QUERY_TYPE_LABELS.get(service_type, str(service_type)),
        "queryTime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": params.get("total", len(normalised)),
        "records": normalised,
    }


def _extract_results_from_dom(page, query: str, service_type: int) -> dict:
    """Fallback: extract results from DOM when API response wasn't captured."""
    # Check for explicit empty state
    empty = page.evaluate("""() => {
        const el = document.querySelector('.el-table__empty-text');
        return el ? el.innerText.trim() : null;
    }""")
    if empty:
        return {
            "query": query,
            "queryType": QUERY_TYPE_LABELS.get(service_type, str(service_type)),
            "queryTime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": 0,
            "records": [],
        }

    # Try table rows
    data = page.evaluate("""() => {
        const rows = document.querySelectorAll(
            '.el-table__body-wrapper table tbody tr');
        const results = [];
        for (const row of rows) {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 5) {
                const unitName = cells[1]?.innerText?.trim() || '';
                if (unitName.includes('function') || unitName.includes('var ')) continue;
                results.push({
                    unitName:       unitName,
                    nature:         cells[2]?.innerText?.trim() || '',
                    serviceLicence: cells[3]?.innerText?.trim() || '',
                    mainLicence:    '',
                    domain:         '',
                    updateDate:     cells[4]?.innerText?.trim() || '',
                });
            }
        }
        return results;
    }""")

    if not data:
        # Try detail page
        data = [page.evaluate("""() => {
            const tds = Array.from(document.querySelectorAll('td'));
            const map = {};
            for (let i = 0; i < tds.length; i++) {
                const text = tds[i].innerText.trim().replace(/[：:]/g, '');
                const next = tds[i + 1];
                if (next) map[text] = next.innerText.trim();
            }
            return {
                unitName:       map['主办单位名称'] || '',
                nature:         map['主办单位性质'] || '',
                mainLicence:    map['ICP备案/许可证号'] || '',
                serviceLicence: '',
                domain:         map['网站域名'] || '',
                updateDate:     map['审核通过日期'] || '',
            };
        }""")]
        # If detail page also empty, return no records
        if data and not data[0].get("unitName"):
            data = []

    return {
        "query": query,
        "queryType": QUERY_TYPE_LABELS.get(service_type, str(service_type)),
        "queryTime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(data),
        "records": data,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="beian",
        description="Query ICP registration info from MIIT (beian.miit.gov.cn)",
        epilog="Accepts: domain (baidu.com), unit name (北京百度网讯), or ICP number (京ICP证030173号)",
    )
    parser.add_argument("queries", nargs="+",
                        help="Domain, unit name, or ICP filing number (one or more)")
    parser.add_argument("--website", action="store_const", dest="query_type",
                        const="website", help="Query website ICP (default)")
    parser.add_argument("--app", action="store_const", dest="query_type",
                        const="app", help="Query app registration")
    parser.add_argument("--miniprogram", action="store_const", dest="query_type",
                        const="miniprogram", help="Query miniprogram registration")
    parser.add_argument("--quickapp", action="store_const", dest="query_type",
                        const="quickapp", help="Query quick app registration")
    parser.add_argument("--retries", type=int, default=3,
                        help="Max CAPTCHA solve attempts (default: 3)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Page timeout in seconds (default: 30)")
    parser.add_argument("--no-headless", action="store_true",
                        help="Run with visible browser (debug)")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw JSON only, no formatting")
    parser.add_argument("--screenshot", nargs="?", const=".", default=None,
                        metavar="DIR",
                        help="Save full-page screenshot (default: current dir)")
    args = parser.parse_args()
    if not args.query_type:
        args.query_type = "website"

    results = query_beian_batch(
        args.queries,
        headless=not args.no_headless,
        timeout_ms=args.timeout * 1000,
        retries=args.retries,
        query_type=args.query_type,
        screenshot_dir=args.screenshot,
    )

    if args.raw:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results):
            if i > 0:
                print("=" * 50)
            _print_result(r)

    sys.exit(0 if all("error" not in r for r in results) else 1)


def _print_result(result: dict):
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        return

    meta = [
        ("Query", result["query"]),
        ("Type", result.get("queryType", "网站")),
        ("Query time", result["queryTime"]),
        ("Records", str(result.get("total", len(result["records"])))),
    ]
    for label, val in meta:
        print(f"{label + ':':14s} {val}")
    print()

    if not result["records"]:
        print("  (no results)")
        print()
        return

    labels = {
        "mainLicence":    "ICP备案/许可证号",
        "unitName":       "主办单位名称",
        "nature":         "主办单位性质",
        "updateDate":     "审核通过日期",
        "serviceLicence": "ICP备案/许可证号",
        "domain":         "网站域名",
    }
    ordered = ["mainLicence", "unitName", "nature", "updateDate"]
    service = ["serviceLicence", "domain"]

    for i, rec in enumerate(result["records"], 1):
        if len(result["records"]) > 1:
            print(f"--- Record {i} ---")
        for k in ordered:
            if rec.get(k):
                print(f"  {labels[k]}：  {rec[k]}")
        if any(rec.get(k) for k in service):
            print()
            print("  ICP备案服务信息")
            for k in service:
                if rec.get(k):
                    print(f"    {labels[k]}：  {rec[k]}")
        print()


if __name__ == "__main__":
    main()
