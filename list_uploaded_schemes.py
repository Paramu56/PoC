import argparse
import os
from typing import List, Set

import chromadb

from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH


def list_schemes(db_path: str, collection_name: str) -> None:
    print(f"Connecting to Chroma collection '{collection_name}' at: {os.path.abspath(db_path)}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    # Fetch all metadatas; in practice you may want to page, but this should be fine for a single PDF.
    print("Fetching all metadatas from collection...")
    result = collection.get(include=["metadatas"])
    metas: List[dict] = result.get("metadatas") or []

    schemes: Set[str] = set()
    for m in metas:
        name = (m or {}).get("scheme_name")
        if name:
            schemes.add(str(name))

    print(f"Unique schemes in collection: {len(schemes)}")
    for s in sorted(schemes):
        print(f"- {s}")


def main() -> None:
    ap = argparse.ArgumentParser(description="List unique scheme names stored in the Chroma collection.")
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Chroma persistent directory path.")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    args = ap.parse_args()

    list_schemes(db_path=args.db_path, collection_name=args.collection)


if __name__ == "__main__":
    main()

