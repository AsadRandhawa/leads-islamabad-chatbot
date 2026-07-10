"""
Step 1 of ingestion: crawl leads.edu.pk starting from the Islamabad Campus
page and save each page's clean markdown to data/raw/ as a JSON file.

Usage:
    python ingest/scrape.py
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from firecrawl import Firecrawl
from firecrawl.types import ScrapeOptions

load_dotenv()

START_URL = "https://leads.edu.pk/islamabad-campus/"

# Crawl the whole site (LLU only has one domain, campuses are just pages/
# sub-menus on it) but cap the page count. Firecrawl respects robots.txt
# automatically.
CRAWL_LIMIT = 150

# Only keep pages whose URL looks Islamabad-campus-relevant, plus a fixed
# allowlist of shared/general pages every campus bot should know about.
ISLAMABAD_URL_HINTS = ["islamabad"]
SHARED_URL_HINTS = [
    "about-us",
    "programs",
    "admission",
    "faculty",
    "department",
    "policies",
    "contact",
    "fee",
    "scholarship",
]

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def is_relevant(url: str) -> bool:
    url_lower = url.lower()
    return any(h in url_lower for h in ISLAMABAD_URL_HINTS + SHARED_URL_HINTS)


def slugify(url: str) -> str:
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-")
    return slug[:150] or "index"


def main():
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise SystemExit("Set FIRECRAWL_API_KEY in your .env file first.")

    app = Firecrawl(api_key=api_key)

    print(f"Crawling from {START_URL} (limit={CRAWL_LIMIT}) ...")
    result = app.crawl(
        START_URL,
        limit=CRAWL_LIMIT,
        crawl_entire_domain=True,  # follow sibling/parent links too, not just
                                    # child paths under /islamabad-campus/
        scrape_options=ScrapeOptions(formats=["markdown"]),
    )

    status = getattr(result, "status", None) or (
        result.get("status") if isinstance(result, dict) else None
    )
    print(f"Crawl status: {status}")

    pages = result.data if hasattr(result, "data") else result["data"]
    print(f"Firecrawl returned {len(pages)} pages total.")
    print("URLs found:")
    for page in pages:
        metadata = page.metadata if hasattr(page, "metadata") else page.get("metadata", {})
        url = getattr(metadata, "source_url", None) or metadata.get("sourceURL", "(unknown)")
        print(f"  - {url}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kept = 0
    for page in pages:
        metadata = page.metadata if hasattr(page, "metadata") else page.get("metadata", {})
        source_url = getattr(metadata, "source_url", None) or metadata.get("sourceURL", "")
        markdown = page.markdown if hasattr(page, "markdown") else page.get("markdown", "")

        if not source_url or not markdown:
            continue
        if not is_relevant(source_url):
            continue

        title = getattr(metadata, "title", None) or (
            metadata.get("title") if isinstance(metadata, dict) else None
        ) or source_url

        out_path = RAW_DIR / f"{slugify(source_url)}.json"
        out_path.write_text(
            json.dumps(
                {"url": source_url, "title": title, "markdown": markdown},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        kept += 1

    print(f"Saved {kept} relevant pages to {RAW_DIR}")
    print("Next: review data/raw/*.json for junk content, then run "
          "ingest/build_index.py")


if __name__ == "__main__":
    main()
