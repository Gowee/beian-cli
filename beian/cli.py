"""CLI tool for querying ICP registration info from MIIT (beian.miit.gov.cn)."""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import requests

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
    "license": "增值电信业务经营许可证",
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
                      screenshot_dir: str | None = None,
                      verbose: bool = False) -> list[dict]:
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
        # Block non-essential resources
        page.route("**/*.{woff,woff2,ttf,otf}", lambda route: route.abort())
        page.route("**/analytics**", lambda route: route.abort())
        page.route("**/gtm**", lambda route: route.abort())
        page.route("**/latestMessage**", lambda route: route.abort())
        page.route("**/portalHomePage**", lambda route: route.abort())
        page.route("**/queryOneUpgradeNoticeInfo**", lambda route: route.abort())

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
                                         screenshot_dir=screenshot_dir,
                                         verbose=verbose))
        finally:
            ctx.close()
            browser.close()

    return results


def _do_query(page, query: str, retries: int, service_type: int,
              *, screenshot_dir: str | None = None,
              verbose: bool = False) -> dict:
    # 1. Load the site – networkidle is reliable for this SPA
    if verbose:
        print(f"    [{query[:12]}...] Loading page...", file=sys.stderr)
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
    if verbose:
        print(f"    [{query[:12]}...] Page ready, triggering search...", file=sys.stderr)

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
    if verbose:
        print(f"    [{query[:12]}...] CAPTCHA loaded", file=sys.stderr)

    # 5. Solve CAPTCHA (with retries)
    for attempt in range(1, retries + 1):
        if verbose:
            print(f"    [{query[:12]}...] CAPTCHA attempt {attempt}/{retries}...", file=sys.stderr, end=" ")
        result = page.evaluate(DETECT_GAP_JS)
        if "error" in result:
            if verbose:
                print(f"detection error: {result['error']}", file=sys.stderr)
            if attempt < retries:
                _refresh_captcha(page)
                continue
            return {"error": result["error"]}

        displacement = result["displacement"]
        if verbose:
            print(f"displacement={displacement}px", file=sys.stderr, end=" ")

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
                if verbose:
                    print("WRONG CAPTCHA", file=sys.stderr)
                if attempt < retries:
                    _wait_for_captcha_refresh(page)
                continue

            if verbose:
                print("OK", file=sys.stderr, end=" ")

            # CAPTCHA passed — search response captured
            result = _parse_api_response(search_info.value.json(), query, service_type)
            if verbose:
                print(f"-> {result.get('total', 0)} records", file=sys.stderr)
            if screenshot_dir:
                _take_screenshot(page, query, screenshot_dir)
            return result

        except PwTimeout:
            if verbose:
                print("TIMEOUT", file=sys.stderr)
            if attempt < retries:
                _wait_for_captcha_refresh(page)
                continue
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}", file=sys.stderr)
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
# ICP license query (tsm.miit.gov.cn)
# ---------------------------------------------------------------------------

def _edge_sharp(img_bytes: bytes) -> bytes:
    """Preprocess CAPTCHA: edge enhance + sharpen for ddddocr."""
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Edge enhance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_n = (np.abs(lap) / (np.abs(lap).max() + 1e-6) * 255).astype(np.uint8)
    lc = np.stack([lap_n] * 3, axis=-1)
    ee = cv2.addWeighted(img, 1.5, lc, -0.5, 0)
    # Sharpen
    blur = cv2.GaussianBlur(ee, (0, 0), 1.5)
    r = cv2.addWeighted(ee, 2.0, blur, -1.0, 0)
    _, buf = cv2.imencode(".png", r)
    return buf.tobytes()


def _parse_annual_report(html: str) -> dict | None:
    """Parse年报 JSP page HTML into key-value dict."""
    tds = re.findall(r'<td[^>]*>(.*?)</td>', html, re.DOTALL)
    cleaned = [re.sub(r'<[^>]+>', '', c).strip() for c in tds]
    if len(cleaned) < 9:
        return None
    # Table structure: 序号, 指标名称, 单位/值 repeating
    fields = {}
    for i in range(0, len(cleaned) - 2, 3):
        seq, name, val = cleaned[i], cleaned[i + 1], cleaned[i + 2]
        if name and val and name != "指标名称":
            fields[name] = val
    if not fields:
        return None
    return {
        "fillYear": fields.get("填报年度", ""),
        "enterpriseName": fields.get("企业名称", ""),
        "creditCode": fields.get("统一社会信用代码", ""),
        "legalPerson": fields.get("法定代表人", ""),
        "licenseNo": fields.get("许可证编码", ""),
        "address": fields.get("注册住所", ""),
        "region": fields.get("注册属地", ""),
        "registeredCapital": fields.get("注册资本", ""),
        "businessTypes": fields.get("许可证业务种类", ""),
        "enterpriseNature": fields.get("企业性质", ""),
        "stockStatus": fields.get("上市情况", ""),
        "servicePhone": fields.get("客户投诉服务电话", ""),
        "complaintCount": fields.get("用户投诉量", ""),
        "complaintReplyRate": fields.get("用户投诉回复率", ""),
    }


def query_license_batch(queries: list[str], *, retries: int = 10,
                        verbose: bool = False) -> list[dict]:
    """Query ICP license from tsm.miit.gov.cn for multiple company names."""
    import ddddocr

    ocr = ddddocr.DdddOcr(show_ad=False, beta=True)
    ocr.set_ranges("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    # Create session with connection pooling
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "Referer": "https://tsm.miit.gov.cn/dxxzsp/xkz/xkzgl/resource/qiyesearch.jsp",
        "X-Requested-With": "XMLHttpRequest",
    })

    if verbose:
        print("  [init] Session ready", file=sys.stderr)

    # Cache for授权_info and年报_info keyed by lic_id
    cache = {"auth": {}, "report": {}}

    results = []
    for i, query in enumerate(queries, 1):
        if verbose and len(queries) > 1:
            print(f"\n  [{i}/{len(queries)}] Querying: {query}", file=sys.stderr)
        result = _do_license_query(session, query, ocr, retries, cache, verbose)
        results.append(result)
    session.close()
    return results


def _do_license_query(session: requests.Session, company: str, ocr, retries: int,
                      cache: dict, verbose: bool = False) -> dict:
    """Solve CAPTCHA and query ICP license for one company.

    Retry logic: non-7-char OCR results refresh CAPTCHA without counting as
    a retry. 10 consecutive non-7-char results count as one failure.
    """
    for attempt in range(1, retries + 1):
        non7_count = 0
        while non7_count < 10:
            try:
                # Get CAPTCHA
                if verbose:
                    print(f"    [{company[:8]}...] CAPTCHA attempt {attempt}/{retries}...", file=sys.stderr, end=" ")
                resp = session.post(
                    "https://tsm.miit.gov.cn/dxxzsp/corpinfo/getCode",
                    data={"num": company}, timeout=10)
                captcha = resp.json()
                img_bytes = base64.b64decode(captcha["src"].split(",")[1])

                # OCR with edge_sharp preprocessing
                processed = _edge_sharp(img_bytes)
                code = ocr.classification(processed)

                if len(code) != 7:
                    non7_count += 1
                    if verbose:
                        print(f"OCR='{code}' (len={len(code)}, retrying)", file=sys.stderr)
                    continue

                # Submit
                if verbose:
                    print(f"OCR='{code}' -> submitting...", file=sys.stderr, end=" ")
                check = session.post(
                    "https://tsm.miit.gov.cn/dxxzsp/corpinfo/getcorpinfocount.wf",
                    data={"num": company, "type": "xuke", "code": code},
                    timeout=10).json()

                if check.get("flag") == "0":
                    if verbose:
                        print("WRONG CAPTCHA", file=sys.stderr)
                    break  # Wrong CAPTCHA, count this as one failure

                if verbose:
                    print("OK", file=sys.stderr)

                # Get results
                if verbose:
                    print(f"    [{company[:8]}...] Fetching license data...", file=sys.stderr)
                body = session.post(
                    "https://tsm.miit.gov.cn/dxxzsp/corpinfo/getcorpinfo.wf",
                    data={"num": company, "type": "xuke", "code": code,
                          "pageNum": 1, "pageSize": 100},
                    timeout=10).json()

                records = []
                for item in (body.get("listyj") or []):
                    ywzl_infos = item.get("ywzlInfos") or []
                    lic_id = item.get("lic_id", "")

                    # Fetch授权_info (cached by lic_id)
                    authorizations = []
                    if lic_id:
                        if lic_id in cache["auth"]:
                            authorizations = cache["auth"][lic_id]
                            if verbose:
                                print(f"    [{company[:8]}...] 授权 (cached)", file=sys.stderr)
                        else:
                            if verbose:
                                print(f"    [{company[:8]}...] Fetching 授权...", file=sys.stderr, end=" ")
                            try:
                                auth_body = session.post(
                                    "https://tsm.miit.gov.cn/dxxzsp/corpinfo/getshouquan.wf",
                                    params={"pageNum": 1, "pageSize": 100},
                                    data={"num": lic_id},
                                    timeout=10).json()
                                if isinstance(auth_body, list):
                                    authorizations = auth_body
                                cache["auth"][lic_id] = authorizations
                                if verbose:
                                    print(f"{len(authorizations)} records", file=sys.stderr)
                            except Exception:
                                cache["auth"][lic_id] = []
                                if verbose:
                                    print("error", file=sys.stderr)

                    # Fetch年报_info (cached by lic_id)
                    annual_report = None
                    if lic_id:
                        if lic_id in cache["report"]:
                            annual_report = cache["report"][lic_id]
                            if verbose:
                                print(f"    [{company[:8]}...] 年报 (cached)", file=sys.stderr)
                        else:
                            if verbose:
                                print(f"    [{company[:8]}...] Fetching 年报...", file=sys.stderr, end=" ")
                            fill_year = ""
                            try:
                                ar_body = session.post(
                                    "https://tsm.miit.gov.cn/dxxzsp/corpinfo/getreporty",
                                    data={"num": lic_id},
                                    timeout=10).json()
                                if ar_body and not ar_body.get("ssss"):
                                    fill_year = ar_body.get("FILL_YEAR", "")
                            except Exception:
                                pass
                            try:
                                ar_html = session.get(
                                    "https://tsm.miit.gov.cn/dxxzsp/xkz/xkzgl/resource/qiyereport.jsp",
                                    params={"num": lic_id, "type": "yreport"},
                                    timeout=10).text
                                annual_report = _parse_annual_report(ar_html)
                                if annual_report:
                                    annual_report["fillYear"] = fill_year
                                cache["report"][lic_id] = annual_report
                                if verbose:
                                    print(f"year={fill_year or 'N/A'}", file=sys.stderr)
                            except Exception:
                                cache["report"][lic_id] = None
                                if verbose:
                                    print("error", file=sys.stderr)

                    rec = {
                        "companyName": item.get("company_name", ""),
                        "legalPerson": item.get("faren", ""),
                        "licenseNo": item.get("license_no", ""),
                        "issuingAuthority": item.get("fzjg", ""),
                        "businessTypes": [
                            {
                                "scope": y.get("ywzl", ""),
                                "serviceArea": y.get("fgfw", ""),
                                "issueDate": y.get("fzrq", "") or "----------",
                                "expiryDate": y.get("jzrq", ""),
                            }
                            for y in ywzl_infos
                        ],
                        "authorizations": [
                            {
                                "parentCompany": a.get("name", ""),
                                "subsidiary": a.get("nameints", ""),
                                "scope": a.get("yewu", ""),
                                "licenseNo": a.get("num", ""),
                            }
                            for a in authorizations
                        ],
                    }
                    if annual_report:
                        rec["annualReport"] = annual_report
                    records.append(rec)

                return {
                    "query": company,
                    "queryType": "增值电信业务经营许可证",
                    "queryTime": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total": int(check.get("c", len(records))),
                    "records": records,
                }

            except Exception:
                break  # Network error, count as one failure

        if attempt == retries:
            return {"error": f"Failed after {retries} attempts", "query": company}

    return {"error": f"Failed after {retries} attempts", "query": company}


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
    parser.add_argument("--license", action="store_const", dest="query_type",
                        const="license", help="Query ICP license (增值电信业务经营许可证)")
    parser.add_argument("--retry", type=int, default=5,
                        help="Max CAPTCHA solve attempts (default: 5)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Page timeout in seconds (default: 30)")
    parser.add_argument("--no-headless", action="store_true",
                        help="Run with visible browser (debug)")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw JSON only, no formatting")
    parser.add_argument("--screenshot", nargs="?", const=".", default=None,
                        metavar="DIR",
                        help="Save full-page screenshot (default: current dir)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress (CAPTCHA solving, API calls)")
    args = parser.parse_args()
    if not args.query_type:
        args.query_type = "website"

    # ICP license uses different query path (no browser needed)
    if args.query_type == "license":
        results = query_license_batch(
            args.queries,
            retries=args.retry,
            verbose=args.verbose,
        )
    else:
        results = query_beian_batch(
            args.queries,
            headless=not args.no_headless,
            timeout_ms=args.timeout * 1000,
            retries=args.retry,
            query_type=args.query_type,
            screenshot_dir=args.screenshot,
            verbose=args.verbose,
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

    # ICP license format
    if result.get("queryType") == "增值电信业务经营许可证":
        _print_license_records(result)
        return

    # ICP filing format
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


def _print_license_records(result: dict):
    """Print ICP license records in vertical format for terminal readability."""
    for i, rec in enumerate(result["records"], 1):
        if len(result["records"]) > 1:
            print(f"  === 许可证 {i} ===")
        print(f"  公司名称：      {rec['companyName']}")
        print(f"  法定代表人：    {rec['legalPerson']}")
        print(f"  许可证号：      {rec['licenseNo']}")
        print(f"  发证机关：      {rec['issuingAuthority']}")

        if rec["businessTypes"]:
            print(f"  业务种类：")
            for j, bt in enumerate(rec["businessTypes"], 1):
                print(f"    [{j}] {bt['scope']}")
                print(f"        覆盖范围：{bt['serviceArea']}")
                print(f"        发证日期：{bt['issueDate']}")
                print(f"        有效期至：{bt['expiryDate']}")

        if rec.get("authorizations"):
            print(f"  授权信息：")
            for j, auth in enumerate(rec["authorizations"], 1):
                print(f"    [{j}] 持证公司：{auth['parentCompany']}")
                print(f"        授权子公司：{auth['subsidiary']}")
                print(f"        许可证号：{auth['licenseNo']}")
                print(f"        授权业务及范围：{auth['scope'][:100]}...")

        ar = rec.get("annualReport")
        if ar:
            print(f"  年报公示（{ar.get('fillYear', '')}）：")
            for k, v in [
                ("企业名称", ar.get("enterpriseName", "")),
                ("统一社会信用代码", ar.get("creditCode", "")),
                ("法定代表人", ar.get("legalPerson", "")),
                ("许可证编码", ar.get("licenseNo", "")),
                ("注册住所", ar.get("address", "")),
                ("注册属地", ar.get("region", "")),
                ("注册资本", ar.get("registeredCapital", "")),
                ("企业性质", ar.get("enterpriseNature", "")),
                ("上市情况", ar.get("stockStatus", "")),
                ("客户投诉服务电话", ar.get("servicePhone", "")),
                ("用户投诉量", ar.get("complaintCount", "")),
                ("用户投诉回复率", ar.get("complaintReplyRate", "")),
            ]:
                if v:
                    print(f"    {k}：{v}")
        print()


if __name__ == "__main__":
    main()
