"""
SHL Product Catalog Scraper
============================
Scrapes all assessments from https://www.shl.com/products/product-catalog/
and saves them to shl_assessments.json in the format:
{
  "id": "...",
  "name": "...",
  "url": "...",
  "description": "...",
  "categories": [...],
  "job_levels": [...],
  "languages": [...],
  "duration": "...",
  "remote_support": bool,
  "adaptive_support": bool,
  "embedding_text": "..."
}

Requirements (install once):
    pip install playwright beautifulsoup4
    playwright install chromium

Run:
    python shl_scraper.py
"""

import asyncio
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL      = "https://www.shl.com"
CATALOG_URL   = "https://www.shl.com/products/product-catalog/"
# Filter params: type=1 (Pre-packaged Job Solutions), adjust as needed
CATALOG_PARAMS = "?action_doFilteringForm=Search&f=1&type=1"
PAGE_SIZE     = 12          # SHL uses 12 items per page
OUTPUT_FILE   = "shl_assessments.json"
DELAY_SECONDS = 1.5         # polite delay between requests
# ──────────────────────────────────────────────────────────────────────────────


def make_slug(name: str) -> str:
    """Create a URL-style ID from the assessment name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_embedding_text(item: dict) -> str:
    """Combine key fields into a searchable embedding string."""
    parts = [item.get("name", "")]
    parts += item.get("categories", [])
    parts += item.get("job_levels", [])
    parts += item.get("languages", [])
    if item.get("duration"):
        parts.append(item["duration"])
    # Add key words from description
    desc = item.get("description", "")
    # Keep only meaningful words (>3 chars) from description
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", desc) if w.lower() not in
             {"that", "this", "with", "from", "have", "will", "been", "their",
              "which", "test", "measures", "skills", "knowledge", "ability"}]
    parts.extend(words[:20])
    return " ".join(p for p in parts if p)


async def get_total_count(page) -> int:
    """Scrape the total number of assessments listed."""
    try:
        text = await page.inner_text(".custom-select__count, .catalog-count, [class*='count']")
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


async def parse_catalog_page(html: str) -> list[dict]:
    """Parse catalog listing page and return list of {name, url} dicts."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # SHL uses a table with rows for each assessment
    rows = soup.select("table tbody tr, .product-catalogue__row, [class*='catalogue__row']")
    
    if not rows:
        # Fallback: look for any links to /view/ pages
        links = soup.select("a[href*='/product-catalog/view/']")
        for link in links:
            href = link.get("href", "")
            name = link.get_text(strip=True)
            if name and href:
                results.append({
                    "name": name,
                    "url": urljoin(BASE_URL, href)
                })
        return results

    for row in rows:
        # First <td> usually has the link
        link = row.select_one("a[href*='/product-catalog/view/']")
        if not link:
            continue
        name = link.get_text(strip=True)
        href = link.get("href", "")
        if name and href:
            results.append({
                "name": name,
                "url": urljoin(BASE_URL, href)
            })

    return results


def parse_detail_page(html: str, url: str) -> dict:
    """
    Parse an individual assessment detail page.

    Page structure (from live HTML inspection):
    ─────────────────────────────────────────────
    <h1>  →  Assessment name

    Content is a sequence of <h4> labels followed by a <p> (or plain text):
        #### Description        → next <p> = full description
        #### Job levels         → next <p> text, comma-separated
        #### Languages          → next <p> text, comma-separated
        #### Assessment length  → next <p> = "Approximate Completion Time in minutes = 49"

    Test Type block:
        The label "Test Type:" is plain text; immediately after it are single-letter
        spans (C, P, A, B, K, S, …).  A legend at the bottom of the page maps
        each letter to its full name:
            A → Ability & Aptitude
            B → Biodata & Situational Judgement
            C → Competencies
            D → Development & 360
            E → Assessment Exercises
            K → Knowledge & Skills
            P → Personality & Behavior
            S → Simulations

    Remote Testing block:
        "Remote Testing:" label followed by an <img> when supported (no img = not supported).
        There is NO "yes/no" text — presence of the image is the signal.

    Adaptive/IRT block (same pattern as Remote Testing):
        "Adaptive/IRT:" label followed by an <img> when supported.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Letter → Category legend (static, but parsed for robustness) ──────────
    LETTER_MAP = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations",
    }

    # ── Helper: get the <p> that immediately follows an <h4> by text ──────────
    def get_paragraph_after_h4(label: str) -> str:
        """Return stripped text of the first <p> after the <h4> whose text
        contains `label` (case-insensitive)."""
        for h4 in soup.find_all("h4"):
            if label.lower() in h4.get_text(strip=True).lower():
                # Walk forward siblings until we find a <p>
                for sib in h4.next_siblings:
                    if hasattr(sib, "name"):
                        if sib.name == "p":
                            return sib.get_text(strip=True)
                        if sib.name == "h4":
                            break   # hit the next section header
        return ""

    # ── Helper: split comma-separated values, clean trailing commas ───────────
    def split_csv(text: str) -> list[str]:
        return [v.strip().rstrip(",") for v in text.split(",") if v.strip().rstrip(",")]

    # ── Name ──────────────────────────────────────────────────────────────────
    h1 = soup.find("h1")
    name = h1.get_text(strip=True) if h1 else ""

    # ── Description ───────────────────────────────────────────────────────────
    # The <h4>Description</h4> is followed directly by a <p> with the full text.
    # The meta description is truncated (ends in "…"), so prefer the <p>.
    description = get_paragraph_after_h4("Description")
    if not description:
        # Fallback to meta og:description
        og = soup.find("meta", attrs={"property": "og:description"})
        if og:
            description = og.get("content", "").strip()

    # ── Job Levels ────────────────────────────────────────────────────────────
    job_levels_raw = get_paragraph_after_h4("Job levels")
    job_levels = split_csv(job_levels_raw)

    # ── Languages ─────────────────────────────────────────────────────────────
    languages_raw = get_paragraph_after_h4("Languages")
    languages = split_csv(languages_raw)

    # ── Duration ──────────────────────────────────────────────────────────────
    # Format: "Approximate Completion Time in minutes = 49"
    duration = ""
    length_text = get_paragraph_after_h4("Assessment length")
    if length_text:
        match = re.search(r"=\s*(\d+)", length_text)
        if match:
            duration = f"{match.group(1)} minutes"
        else:
            # Fallback: grab any number + min pattern
            match = re.search(r"(\d+)\s*min", length_text, re.IGNORECASE)
            if match:
                duration = match.group(0).strip()

    # ── Categories (Test Type letters → full names) ────────────────────────────
    # The "Test Type:" label is a plain-text node. The letter badges are
    # individual text nodes / spans directly after it in the same container.
    categories = []
    for node in soup.find_all(string=re.compile(r"Test Type:", re.IGNORECASE)):
        parent = node.parent
        # Collect single uppercase letters from the same container and its
        # immediate next siblings until the next recognisable label.
        container = parent.parent if parent else None
        if container:
            letters = re.findall(r"\b([A-Z])\b", container.get_text())
            categories = [LETTER_MAP[l] for l in letters if l in LETTER_MAP]
        break   # only process first match

    # ── Remote Testing ─────────────────────────────────────────────────────────
    # "Remote Testing:" is a plain-text node. An <img> immediately after it
    # signals "yes"; no image (or the section is absent) means "no".
    remote_support = False
    for node in soup.find_all(string=re.compile(r"Remote Testing:", re.IGNORECASE)):
        parent = node.parent
        # Look for an <img> within the same parent or its next sibling
        img = parent.find("img") or (
            parent.find_next_sibling() and parent.find_next_sibling().find("img")
        )
        if img:
            remote_support = True
        break

    # ── Adaptive / IRT ────────────────────────────────────────────────────────
    adaptive_support = False
    for node in soup.find_all(string=re.compile(r"Adaptive/IRT:|Adaptive Testing:", re.IGNORECASE)):
        parent = node.parent
        img = parent.find("img") or (
            parent.find_next_sibling() and parent.find_next_sibling().find("img")
        )
        if img:
            adaptive_support = True
        break

    # ── Build item ────────────────────────────────────────────────────────────
    # Note: remote_support and adaptive_support are NOT included — they depend
    # on a coloured dot/icon on the page which cannot be reliably scraped as
    # text. Use the official catalog JSON (which has remote/adaptive fields
    # derived from the source) instead.
    item = {
        "id":               make_slug(name) if name else "",
        "name":             name,
        "url":              url,
        "description":      description,
        "categories":       categories,
        "job_levels":       job_levels,
        "languages":        languages,
        "duration":         duration,
    }
    item["embedding_text"] = build_embedding_text(item)
    return item


async def scrape_all():
    print("🚀 Starting SHL catalog scraper...")
    all_assessments = []
    seen_urls = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        page = await context.new_page()

        # ── Step 1: Discover all assessment links from catalog pages ──────────
        start = 0
        catalog_links = []

        while True:
            url = f"{CATALOG_URL}{CATALOG_PARAMS}&start={start}"
            print(f"  📄 Catalog page start={start}: {url}")
            
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)  # extra JS settle time

            html = await page.content()
            links = await parse_catalog_page(html)

            if not links:
                print(f"  ⚠️  No links found at start={start}. Stopping pagination.")
                break

            new_links = [l for l in links if l["url"] not in seen_urls]
            if not new_links:
                print(f"  ✅ No new links found at start={start}. Done paginating.")
                break

            catalog_links.extend(new_links)
            for l in new_links:
                seen_urls.add(l["url"])

            print(f"     Found {len(new_links)} new assessments (total so far: {len(catalog_links)})")

            # Check if there are more pages
            has_next = await page.query_selector("a[rel='next'], .pagination__next:not([disabled]), [class*='next']:not([disabled])")
            if not has_next:
                print("  ✅ No 'next page' button found. Pagination complete.")
                break

            start += PAGE_SIZE
            await asyncio.sleep(DELAY_SECONDS)

        print(f"\n📋 Total assessments found: {len(catalog_links)}")

        # ── Step 2: Visit each detail page ────────────────────────────────────
        for i, link in enumerate(catalog_links, 1):
            print(f"  [{i}/{len(catalog_links)}] Scraping: {link['name']}")
            try:
                await page.goto(link["url"], wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1500)
                html = await page.content()
                item = parse_detail_page(html, link["url"])

                # Fallback name from catalog listing
                if not item["name"]:
                    item["name"] = link["name"]
                    item["id"]   = make_slug(link["name"])

                all_assessments.append(item)
                print(f"     ✓ {item['name']} | {item['duration']} | remote={item['remote_support']}")

            except Exception as e:
                print(f"     ✗ Error: {e}")
                # Still save what we have
                all_assessments.append({
                    "id":               make_slug(link["name"]),
                    "name":             link["name"],
                    "url":              link["url"],
                    "description":      "",
                    "categories":       [],
                    "job_levels":       [],
                    "languages":        [],
                    "duration":         "",
                    "remote_support":   False,
                    "adaptive_support": False,
                    "embedding_text":   link["name"],
                    "_error":           str(e),
                })

            await asyncio.sleep(DELAY_SECONDS)

        await browser.close()

    # ── Step 3: Save to JSON ──────────────────────────────────────────────────
    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_assessments, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Saved {len(all_assessments)} assessments → {output_path.resolve()}")
    return all_assessments


if __name__ == "__main__":
    asyncio.run(scrape_all())