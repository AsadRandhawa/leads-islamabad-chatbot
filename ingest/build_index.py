"""
Step 2 of ingestion: clean the raw scraped pages (strip spam/junk/nav
boilerplate), chunk them heading-aware, and load them into a local Chroma
collection.

Usage:
    python ingest/build_index.py
"""

import json
import re
from pathlib import Path

import chromadb

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "leads_islamabad"

CHUNK_SIZE = 800        # characters per chunk (within a section)
CHUNK_OVERLAP = 120
MIN_CHUNK_LEN = 40      # drop anything shorter than this after spam redaction

HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)

# Terms that flagged injected spam content on the live site when this
# project was first put together. Sentences matching these get surgically
# redacted from a chunk (not the whole chunk dropped) so legitimate content
# sharing a chunk boundary with spam isn't lost. Extend if you spot more
# junk in data/raw.
SPAM_PATTERNS = [
    r"casino",
    r"betting",
    r"\bbonus code\b",
    r"free spins",
    r"gambl",
]
SPAM_RE = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)


def clean_markdown(md: str) -> str:
    # Drop lines that are pure nav/menu repeats (very short, all-caps or
    # single links) — crude but effective for WordPress menu boilerplate.
    lines = [ln for ln in md.splitlines() if len(ln.strip()) > 3]
    text = "\n".join(lines)

    # Strip markdown images entirely — alt text/URLs are pure noise for
    # embeddings and generation.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Collapse markdown links to their visible text — keeps "BS Computer
    # Science" instead of "[BS Computer Science](https://...)" so link-heavy
    # sections (like program listings) read as clean text and embed well.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def redact_spam(chunk: str) -> tuple[str, bool]:
    """Remove only the sentence(s)/lines containing spam patterns, keeping
    the rest of the chunk intact. Returns (cleaned_chunk, was_redacted)."""
    if not SPAM_RE.search(chunk):
        return chunk, False
    # Split into sentence-ish pieces (keep the delimiter attached) and drop
    # any piece that matches a spam pattern.
    pieces = re.split(r"(?<=[.\n])", chunk)
    kept = [p for p in pieces if not SPAM_RE.search(p)]
    cleaned = "".join(kept).strip()
    return cleaned, True


def split_into_sections(text: str):
    """Split text on markdown headings so each section stays topically
    coherent (e.g. '## Academic Programs' becomes its own section instead
    of bleeding into the fee table or the next unrelated block)."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group().strip()
        sections.append((heading, text[start:end].strip()))
    return sections


def chunk_section(heading, section_text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Char-window chunk a single section, keeping the heading attached to
    every resulting piece so context survives even when a section is long
    enough to need splitting."""
    if len(section_text) <= size:
        return [section_text]

    chunks = []
    start = 0
    while start < len(section_text):
        piece = section_text[start:start + size].strip()
        if piece:
            if heading and not piece.startswith(heading):
                piece = f"{heading}\n{piece}"
            chunks.append(piece)
        start += size - overlap
    return chunks


def build_chunks(cleaned_text: str, page_title: str):
    """Heading-aware chunking: split by section first, sub-chunk only if a
    section is too long, and prefix every chunk with the page title so short
    fragments (like a bare program list) still carry enough context to
    embed and rank well."""
    chunks = []
    for heading, section_text in split_into_sections(cleaned_text):
        if not section_text.strip():
            continue
        for piece in chunk_section(heading, section_text):
            chunks.append(f"Page: {page_title}\n{piece}".strip())
    return chunks


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.json")):
        raise SystemExit("No scraped pages found. Run ingest/scrape.py first.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Fresh build each run so re-running never duplicates entries.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, docs, metadatas = [], [], []
    redacted_count = 0
    dropped_count = 0
    page_count = 0

    for path in RAW_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        cleaned = clean_markdown(data["markdown"])
        page_count += 1
        # A page's URL slug isn't a reliable Islamabad signal on its own —
        # some genuinely Islamabad-specific posts have URLs like
        # "/🎓-admissions-now-open-for-fall-2026-intake/" with no
        # "islamabad" substring at all. Fall back to checking the URL/title
        # OR each individual chunk's own text.
        url_or_title_islamabad = (
            "islamabad" in data["url"].lower() or "islamabad" in data["title"].lower()
        )

        for i, raw_chunk in enumerate(build_chunks(cleaned, data["title"])):
            chunk, was_redacted = redact_spam(raw_chunk)
            if was_redacted:
                redacted_count += 1
            if len(chunk) < MIN_CHUNK_LEN:
                dropped_count += 1
                continue

            is_islamabad = url_or_title_islamabad or "islamabad" in chunk.lower()
            campus = "islamabad" if is_islamabad else "university-wide"

            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metadatas.append({
                "url": data["url"],
                "title": data["title"],
                "campus": campus,
            })

    if not docs:
        raise SystemExit("Nothing to index after cleaning — check data/raw contents.")

    # Chroma batches writes; keep it simple with one add() call since this
    # dataset is small (a single campus site).
    collection.add(ids=ids, documents=docs, metadatas=metadatas)

    print(f"Indexed {len(docs)} chunks from {page_count} pages.")
    if redacted_count:
        print(f"Redacted spam sentences (casino/gambling injection) from "
              f"{redacted_count} chunks — rest of each chunk was kept.")
    if dropped_count:
        print(f"Dropped {dropped_count} chunks that were empty/too short "
              f"after cleaning.")
    print(f"Chroma DB written to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
