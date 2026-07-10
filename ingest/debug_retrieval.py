"""
Debug helper: shows how a query ranks against ALL chunks (not just top-k),
and separately dumps every chunk that came from a specific page so you can
see exactly where content landed after chunking.

Usage:
    python ingest/debug_retrieval.py "what programs are offered at Islamabad campus?"
    python ingest/debug_retrieval.py --page islamabad-campus
"""

import sys
from pathlib import Path

import chromadb

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "leads_islamabad"


def show_ranking(collection, query: str, n: int = 25):
    print(f'\n=== Ranking for query: "{query}" (top {n}) ===\n')
    results = collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        preview = doc[:120].replace("\n", " ")
        print(f"#{rank:2d}  distance={dist:.4f}  campus={meta.get('campus'):16s} "
              f"{meta['url']}")
        print(f"      {preview}...\n")


def show_page_chunks(collection, page_slug: str):
    print(f'\n=== All chunks whose id starts with "{page_slug}" ===\n')
    all_data = collection.get(include=["documents", "metadatas"])
    matched = [
        (id_, doc, meta)
        for id_, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
        if page_slug in id_
    ]
    if not matched:
        print(f"No chunks found with id containing '{page_slug}'. "
              f"Check the exact filename stem in data/raw/.")
        return
    for id_, doc, meta in sorted(matched, key=lambda x: x[0]):
        print(f"--- {id_} ({meta.get('campus')}) ---")
        print(doc)
        print()


def main():
    if not CHROMA_DIR.exists():
        raise SystemExit("No index found. Run ingest/build_index.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    args = sys.argv[1:]
    if not args:
        raise SystemExit(
            'Usage:\n'
            '  python ingest/debug_retrieval.py "your query here"\n'
            '  python ingest/debug_retrieval.py --page islamabad-campus'
        )

    if args[0] == "--page":
        show_page_chunks(collection, args[1])
    else:
        query = " ".join(args)
        show_ranking(collection, query)


if __name__ == "__main__":
    main()
