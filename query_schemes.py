import argparse
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL


def _format_hit(doc: str, meta: Dict[str, Any], distance: Optional[float]) -> str:
    scheme = meta.get("scheme_name", "UNKNOWN_SCHEME")
    page = meta.get("page", "?")
    source = meta.get("source", "")
    category = meta.get("category")
    tags = meta.get("tags")

    header = f"{scheme}  (page {page})"
    if category:
        header += f" | Category: {category}"
    if tags:
        header += f" | Tags: {tags}"
    if distance is not None:
        header += f" | distance={distance:.4f}"

    body = doc.strip()
    if len(body) > 1200:
        body = body[:1200].rstrip() + " …"

    src_line = f"Source: {source}" if source else ""
    return "\n".join([header, src_line, body]).strip()


def query_once(
    *,
    db_path: str,
    collection_name: str,
    model_name: str,
    question: str,
    n_results: int,
) -> None:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name, embedding_function=ef)

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs: List[str] = (results.get("documents") or [[]])[0]
    metas: List[dict] = (results.get("metadatas") or [[]])[0]
    dists: List[float] = (results.get("distances") or [[]])[0]

    if not docs:
        print("No results found (collection may be empty).")
        return

    print()
    print(f"Question: {question}")
    print(f"Top {len(docs)} matches:")
    for i in range(len(docs)):
        dist = dists[i] if i < len(dists) else None
        print()
        print(f"[{i + 1}]")
        print(_format_hit(docs[i], metas[i] or {}, dist))


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the schemes ChromaDB collection (local SentenceTransformer embeddings).")
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Chroma persistent directory path.")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name or local path.")
    ap.add_argument("-n", "--n-results", type=int, default=5, help="Number of matches to return.")
    ap.add_argument("--question", help="Ask a single question and exit.")
    args = ap.parse_args()

    print(f"DB path: {os.path.abspath(args.db_path)}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.model}")

    if args.question:
        query_once(
            db_path=args.db_path,
            collection_name=args.collection,
            model_name=args.model,
            question=args.question,
            n_results=args.n_results,
        )
        return

    print("\nType your question and press Enter. Type 'quit' to exit.")
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break
        query_once(
            db_path=args.db_path,
            collection_name=args.collection,
            model_name=args.model,
            question=q,
            n_results=args.n_results,
        )


if __name__ == "__main__":
    main()

