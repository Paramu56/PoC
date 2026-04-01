import argparse
import os
from typing import Set

import chromadb

from ingest_karnataka_schemes import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_PATH,
    extract_pages,
    parse_page_into_scheme_chunks,
)


def schemes_from_pdf(pdf_path: str) -> Set[str]:
    pages = extract_pages(pdf_path)
    names: Set[str] = set()
    for page_num, text in pages:
        for sc in parse_page_into_scheme_chunks(page_num, text):
            if sc.scheme_name:
                names.add(sc.scheme_name.strip())
    return names


def schemes_from_db(db_path: str, collection_name: str) -> Set[str]:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    result = collection.get(include=["metadatas"])
    metas = result.get("metadatas") or []

    names: Set[str] = set()
    for m in metas:
        name = (m or {}).get("scheme_name")
        if name:
            names.add(str(name).strip())
    return names


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare schemes in the PDF vs schemes stored in the Chroma collection."
    )
    ap.add_argument(
        "--pdf",
        required=True,
        help="Path to the schemes PDF (same file you used for ingestion).",
    )
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Chroma persistent directory path.")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    args = ap.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    print(f"Reading schemes from PDF: {os.path.abspath(pdf_path)}")
    pdf_schemes = schemes_from_pdf(pdf_path)
    print(f"  - PDF schemes detected: {len(pdf_schemes)}")

    print(f"Reading schemes from DB collection '{args.collection}' at: {os.path.abspath(args.db_path)}")
    db_schemes = schemes_from_db(db_path=args.db_path, collection_name=args.collection)
    print(f"  - DB schemes detected: {len(db_schemes)}")

    only_in_pdf = sorted(pdf_schemes - db_schemes)
    only_in_db = sorted(db_schemes - pdf_schemes)

    print()
    print("Schemes present in PDF but NOT in DB:")
    if not only_in_pdf:
        print("  (none)")
    else:
        for s in only_in_pdf:
            print(f"  - {s}")

    print()
    print("Schemes present in DB but NOT in PDF:")
    if not only_in_db:
        print("  (none)")
    else:
        for s in only_in_db:
            print(f"  - {s}")


if __name__ == "__main__":
    main()

