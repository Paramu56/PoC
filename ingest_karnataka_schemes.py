import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from graph_knowledge import (
    DATA_DIR,
    DEFAULT_CYPHER_PATH,
    GRAPH_COMPILED_PATH,
    compile_cypher_file,
    save_compiled_graph,
)


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "schemes-db"
DEFAULT_DB_PATH = "./chroma_schemes_db"
# Preferred PDF in project data folder (see data/README.txt)
DEFAULT_PDF_IN_DATA = os.path.join(DATA_DIR, "Karnataka Schemes.pdf")
LEGACY_DOWNLOADS_BASENAME = "karnataka-schemes-list.pdf"


@dataclass(frozen=True)
class SchemeChunk:
    scheme_name: str
    text: str
    page: int
    category: Optional[str] = None
    level_for: Optional[str] = None
    tags: Optional[str] = None
    ingest_mode: str = "scheme_blocks"  # or "page_fallback"


def _clean_page_text(text: str) -> str:
    # Remove footer markers like: "-- 1 of 24 --"
    text = re.sub(r"\n\s*--\s*\d+\s+of\s+\d+\s*--\s*\n", "\n\n", text, flags=re.IGNORECASE)
    # Normalize whitespace a bit (keep newlines as they help parsing).
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    print(f"[1/5] Reading PDF and extracting text per page: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = _clean_page_text(raw)
        if cleaned:
            pages.append((i + 1, cleaned))
        if (i + 1) % 5 == 0:
            print(f"  - processed {i + 1}/{len(reader.pages)} pages...")
    print(f"  - extracted text from {len(pages)}/{len(reader.pages)} pages")
    return pages


def _parse_scheme_blocks_from_lines(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Heuristic parser:
    - A scheme starts at a non-empty line whose *next non-empty* line starts with 'Category:'.
    - Capture lines until the next scheme start.
    Returns: list of (scheme_name, block_lines).
    """
    def next_nonempty_idx(start: int) -> Optional[int]:
        for j in range(start, len(lines)):
            if lines[j].strip():
                return j
        return None

    starts: List[int] = []
    for i in range(len(lines)):
        if not lines[i].strip():
            continue
        j = next_nonempty_idx(i + 1)
        if j is None:
            continue
        if lines[j].lstrip().startswith("Category:"):
            starts.append(i)

    blocks: List[Tuple[str, List[str]]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        scheme_name = lines[start].strip()
        block = [ln.rstrip() for ln in lines[start:end] if ln.strip()]
        blocks.append((scheme_name, block))
    return blocks


def parse_page_into_scheme_chunks(page_num: int, page_text: str) -> List[SchemeChunk]:
    lines = page_text.split("\n")
    blocks = _parse_scheme_blocks_from_lines(lines)
    chunks: List[SchemeChunk] = []

    for scheme_name, block_lines in blocks:
        block_text = "\n".join(block_lines).strip()
        category = None
        level_for = None
        tags = None

        for ln in block_lines:
            if ln.startswith("Category:"):
                category = ln[len("Category:") :].strip() or None
            elif ln.startswith("Level:"):
                level_for = ln[len("Level:") :].strip() or None
            elif ln.startswith("Tags:"):
                tags = ln[len("Tags:") :].strip() or None

        chunks.append(
            SchemeChunk(
                scheme_name=scheme_name,
                text=block_text,
                page=page_num,
                category=category,
                level_for=level_for,
                tags=tags,
            )
        )

    return chunks


def parse_page_fallback_chunks(page_num: int, page_text: str) -> List[SchemeChunk]:
    """
    When the PDF does not follow 'title + Category:' blocks, ingest whole-page text
    so nothing is lost. Scheme label is synthetic for citation (page still correct).
    """
    title = f"Karnataka Schemes (page {page_num})"
    return [
        SchemeChunk(
            scheme_name=title,
            text=page_text.strip(),
            page=page_num,
            ingest_mode="page_fallback",
        )
    ]


def chunk_text(text: str, max_chars: int = 2400, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunker (works fine for English scheme descriptions).
    Keeps overlap to help retrieval continuity.
    """
    t = text.strip()
    if len(t) <= max_chars:
        return [t]

    out: List[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        # try to end on a boundary
        window = t[start:end]
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "), window.rfind(" "))
        if cut > 400:
            end = start + cut
        out.append(t[start:end].strip())
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return out


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def ingest(
    pdf_path: str,
    db_path: str,
    collection_name: str,
    model_name: str,
    *,
    fresh: bool = False,
) -> None:
    print(f"[0/5] Initializing Chroma collection '{collection_name}' at: {os.path.abspath(db_path)}")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    chroma_client = chromadb.PersistentClient(path=db_path)

    if fresh:
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"  - removed old collection '{collection_name}' (only new PDF data will remain)")
        except Exception:
            # Collection may not exist yet on first run
            pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    try:
        existing = collection.count()
    except Exception:
        existing = None
    if existing is not None and not fresh:
        print(f"  - existing chunks already in collection: {existing}")
    elif fresh:
        print("  - collection is empty; ingesting from scratch")
    print(f"  - embedding model: {model_name}")

    pages = extract_pages(pdf_path)
    print("[2/5] Detecting schemes on each page...")
    all_scheme_chunks: List[SchemeChunk] = []
    for page_num, text in pages:
        all_scheme_chunks.extend(parse_page_into_scheme_chunks(page_num, text))

    if not all_scheme_chunks:
        print("  - no 'Category:' blocks found; falling back to per-page chunks")
        for page_num, text in pages:
            all_scheme_chunks.extend(parse_page_fallback_chunks(page_num, text))
    if not all_scheme_chunks:
        raise SystemExit("No text could be extracted from the PDF.")

    unique_schemes_set = {sc.scheme_name for sc in all_scheme_chunks}
    print(f"  - scheme entries to embed: {len(all_scheme_chunks)}")
    print(f"  - unique scheme labels: {len(unique_schemes_set)}")

    print("[3/5] Chunking schemes and preparing records...")
    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for sc in all_scheme_chunks:
        sub_chunks = chunk_text(sc.text)
        for chunk_idx, chunk in enumerate(sub_chunks):
            doc_id = _stable_id(os.path.abspath(pdf_path), sc.scheme_name, str(sc.page), str(chunk_idx))
            documents.append(chunk)
            metadatas.append(
                {
                    "scheme_name": sc.scheme_name,
                    "page": sc.page,
                    "category": sc.category,
                    "level_for": sc.level_for,
                    "tags": sc.tags,
                    "source": os.path.abspath(pdf_path),
                    "chunk_index": chunk_idx,
                    "ingest_mode": sc.ingest_mode,
                }
            )
            ids.append(doc_id)

    print(f"  - prepared chunks: {len(documents)}")

    print("[4/5] Writing to Chroma..." + (" (full replace)" if fresh else " (idempotent upsert)"))
    # Prefer upsert (add if new, replace if exists).
    # If upsert isn't supported, delete existing ids then add (still idempotent).
    try:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        mode = "upsert"
    except Exception:
        try:
            existing_rows = collection.get(ids=ids, include=[])
            existing_ids = set(existing_rows.get("ids") or [])
        except Exception:
            existing_ids = set()
        if existing_ids:
            print(f"  - upsert not available; deleting {len(existing_ids)} existing chunks before add...")
            collection.delete(ids=list(existing_ids))
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        mode = "delete+add"

    final_count = None
    try:
        final_count = collection.count()
    except Exception:
        pass

    print("[5/5] Done.")
    print(f"Source: {os.path.abspath(pdf_path)}")
    print(f"Collection: {collection_name}")
    print(f"DB path: {os.path.abspath(db_path)}")
    print(f"Embedding model: {model_name}")
    print(f"Detected schemes (unique names): {len(unique_schemes_set)}")
    print(f"Written chunks: {len(documents)} (mode={mode})")
    if final_count is not None:
        print(f"Collection count now: {final_count}")


def _find_schemes_pdf(explicit_pdf: Optional[str]) -> str:
    """
    Resolution order:
    1) --pdf path if provided
    2) data/Karnataka Schemes.pdf (recommended)
    3) any single .pdf under data/
    4) Downloads/karnataka-schemes-list.pdf (legacy)
    5) any single match in Downloads starting with karnataka-schemes-list
    """
    if explicit_pdf and explicit_pdf.strip():
        return explicit_pdf.strip().strip('"')

    if os.path.isfile(DEFAULT_PDF_IN_DATA):
        return DEFAULT_PDF_IN_DATA

    data_pdfs: List[str] = []
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            if name.lower().endswith(".pdf"):
                data_pdfs.append(os.path.join(DATA_DIR, name))
    if len(data_pdfs) == 1:
        return data_pdfs[0]
    if len(data_pdfs) > 1:
        print(f"Multiple PDFs in {DATA_DIR}. Put your main file as:")
        print(f"  {DEFAULT_PDF_IN_DATA}")
        print("Or pass: --pdf <path>")
        for p in sorted(data_pdfs)[:15]:
            print(f" - {p}")
        raise SystemExit(2)

    downloads_dir = os.path.join(os.environ.get("USERPROFILE") or os.path.expanduser("~"), "Downloads")
    legacy = os.path.join(downloads_dir, LEGACY_DOWNLOADS_BASENAME)
    if os.path.isfile(legacy):
        return legacy

    candidates: List[str] = []
    try:
        for name in os.listdir(downloads_dir):
            if name.lower().startswith("karnataka-schemes-list"):
                full_path = os.path.join(downloads_dir, name)
                if os.path.isfile(full_path):
                    candidates.append(full_path)
    except Exception:
        candidates = []

    pdf_candidates = [p for p in candidates if p.lower().endswith(".pdf")]
    preferred = pdf_candidates if pdf_candidates else candidates
    if len(preferred) == 1:
        return preferred[0]

    print("PDF not found. Do one of the following:")
    print(f"  1) Copy your file to: {DEFAULT_PDF_IN_DATA}")
    print("  2) Or run: python ingest_karnataka_schemes.py --pdf \"C:\\path\\to\\file.pdf\"")
    if preferred:
        print("Found multiple candidates in Downloads:")
        for p in sorted(preferred)[:10]:
            print(f" - {p}")
        print("Re-run with --pdf <full_path>")
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Karnataka schemes list PDF into Chroma (SentenceTransformers).")
    ap.add_argument(
        "--pdf",
        required=False,
        help=f"Path to schemes PDF (default: {DEFAULT_PDF_IN_DATA} if present)",
    )
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Chroma persistent directory path.")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name or local path.")
    ap.add_argument(
        "--cypher",
        default=DEFAULT_CYPHER_PATH,
        help=f"Path to schemes.cypher for graph rules (default: {DEFAULT_CYPHER_PATH})",
    )
    ap.add_argument(
        "--no-cypher",
        action="store_true",
        help="Skip compiling schemes.cypher to graph_compiled.json",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the Chroma collection first so ONLY this PDF is indexed (removes all old chunks).",
    )
    args = ap.parse_args()

    pdf_path = _find_schemes_pdf(args.pdf)
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    ingest(
        pdf_path=pdf_path,
        db_path=args.db_path,
        collection_name=args.collection,
        model_name=args.model,
        fresh=args.fresh,
    )

    if not args.no_cypher and os.path.isfile(args.cypher):
        print(f"[graph] Compiling Cypher: {args.cypher}")
        compiled = compile_cypher_file(args.cypher)
        save_compiled_graph(compiled, GRAPH_COMPILED_PATH)
        print(f"[graph] Wrote {compiled.get('count', 0)} statements -> {GRAPH_COMPILED_PATH}")
    elif not args.no_cypher:
        print(f"[graph] No Cypher file at {args.cypher} (optional). Add data/schemes.cypher to enable graph context.")


if __name__ == "__main__":
    main()

