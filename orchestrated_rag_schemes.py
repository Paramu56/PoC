import argparse
import os
import time
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai.errors import ClientError

from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


def _build_retriever(db_path: str, collection_name: str, model_name: str):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name, embedding_function=ef)
    return collection


def _retrieve(
    collection,
    question: str,
    n_results: int,
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs: List[str] = (results.get("documents") or [[]])[0]
    metas: List[dict] = (results.get("metadatas") or [[]])[0]
    dists: List[float] = (results.get("distances") or [[]])[0]
    return docs, metas, dists


def _format_context_block(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        scheme = meta.get("scheme_name", "UNKNOWN_SCHEME")
        page = meta.get("page", "?")
        category = meta.get("category")
        tags = meta.get("tags")
        source = meta.get("source", "")

        header = f"[{i}] Scheme: {scheme} (page {page})"
        if category:
            header += f" | Category: {category}"
        if tags:
            header += f" | Tags: {tags}"

        lines.append(header)
        if source:
            lines.append(f"Source: {source}")
        lines.append(doc.strip())
        lines.append("")  # blank line
    return "\n".join(lines).strip()


def _init_gemini_client() -> genai.Client:
    # Read API key from standard environment variable.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Set it to your Gemini API key before running this script."
        )
    return genai.Client(api_key=api_key)


def _gemini_generate_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents: str,
    max_attempts: int = 5,
    initial_wait_s: float = 2.0,
) -> Any:
    """
    Retry Gemini calls for transient high-demand / quota-per-minute spikes.
    """
    wait_s = initial_wait_s
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except ClientError as e:
            last_error = e
            msg = str(e)
            transient = any(
                token in msg
                for token in [
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "experiencing high demand",
                    "RetryInfo",
                    "Please retry in",
                ]
            )
            if not transient or attempt == max_attempts:
                raise
            print(f"  - Gemini busy/rate-limited (attempt {attempt}/{max_attempts}). Retrying in {wait_s:.1f}s...")
            time.sleep(wait_s)
            wait_s = min(wait_s * 2, 30.0)

    if last_error:
        raise last_error
    raise RuntimeError("Gemini call failed unexpectedly.")


def _ask_gemini_for_refinements(
    client: genai.Client,
    llm_model: str,
    question: str,
    context_summary: str,
    max_refinements: int = 3,
) -> Dict[str, Any]:
    """
    Ask Gemini whether current context is sufficient.
    If not, ask it to propose refined search queries for Chroma.
    """
    prompt = dedent(
        f"""
        You are helping orchestrate retrieval for a question about Karnataka government schemes.

        USER QUESTION:
        {question}

        CURRENT CONTEXT SUMMARY (from database):
        {context_summary}

        TASK:
        1. Decide if the existing context is enough to answer the question accurately.
        2. If NOT enough, propose up to {max_refinements} short, focused search queries that could retrieve better context.
           Each query should be a single sentence suitable for semantic search over scheme descriptions.

        Respond ONLY in JSON with this structure:
        {{
          "sufficient": true/false,
          "reason": "<short explanation>",
          "refined_queries": ["...", "..."]  // empty list if sufficient is true
        }}
        """
    )

    # Return the raw GenerateContentResponse; we'll parse it downstream.
    resp = _gemini_generate_with_retry(client, model=llm_model, contents=prompt)
    return resp


def _safe_extract_json_from_gemini(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gemini returns a structured dict; but content is still text. We'll try to parse JSON-like text safely.
    """
    import json

    # google-genai v1 returns a GenerateContentResponse object.
    # Prefer .text; fall back to candidates/parts if needed.
    text = ""
    try:
        if hasattr(raw, "text") and isinstance(raw.text, str):
            text = raw.text
        elif hasattr(raw, "candidates"):
            cands = getattr(raw, "candidates") or []
            if cands:
                content = getattr(cands[0], "content", None)
                if content and getattr(content, "parts", None):
                    part0 = content.parts[0]
                    text = getattr(part0, "text", "") or ""
    except Exception:
        pass

    if not text:
        return {"sufficient": True, "reason": "No guidance from model.", "refined_queries": []}

    # Try to extract JSON from surrounding text.
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
    else:
        snippet = text

    try:
        return json.loads(snippet)
    except Exception:
        # Fallback: assume it's enough and don't refine.
        return {"sufficient": True, "reason": "Could not parse JSON from model output.", "refined_queries": []}


def _ask_gemini_final_answer(
    client: genai.Client,
    llm_model: str,
    question: str,
    context_block: str,
) -> str:
    prompt = dedent(
        f"""
        You are an expert assistant on Karnataka government schemes.

        Use ONLY the provided context to answer the user's question.
        If the answer is not clearly supported by the context, say so and explain what is missing.

        Always:
        - mention the scheme names you are using
        - where possible, cite page numbers from the context in parentheses, e.g., "(page 12)"
        - answer in clear, simple language for citizens

        CONTEXT:
        {context_block}

        USER QUESTION:
        {question}
        """
    )

    resp = _gemini_generate_with_retry(client, model=llm_model, contents=prompt)
    return resp.text


def run_orchestrated_rag(
    *,
    db_path: str,
    collection_name: str,
    model_name: str,
    llm_model: str,
    question: str,
    initial_k: int = 6,
    refined_k: int = 4,
) -> None:
    collection = _build_retriever(db_path=db_path, collection_name=collection_name, model_name=model_name)
    client = _init_gemini_client()

    # Step 1: initial retrieval
    print("[1/3] Initial retrieval from Chroma...")
    docs, metas, dists = _retrieve(collection, question=question, n_results=initial_k)
    if not docs:
        print("No results from Chroma. The collection might be empty.")
        return

    initial_context_block = _format_context_block(docs, metas)

    # Step 2: ask Gemini if context is sufficient and get refined queries if needed
    print("[2/3] Asking Gemini if more context is needed...")
    summary_text = "\n".join(
        f"- {m.get('scheme_name', 'UNKNOWN')} (page {m.get('page', '?')})"
        for m in metas
    )
    try:
        raw_refine = _ask_gemini_for_refinements(
            client, llm_model=llm_model, question=question, context_summary=summary_text
        )
    except ClientError as e:
        raise SystemExit(
            f"Gemini refinement call failed for model '{llm_model}'.\n"
            f"Error: {e}\n"
            "Try a different model with --llm-model (for example: gemini-2.0-flash)."
        )
    refine_info = _safe_extract_json_from_gemini(raw_refine)

    sufficient = bool(refine_info.get("sufficient", True))
    refined_queries: List[str] = list(refine_info.get("refined_queries") or [])

    print(f"  - Gemini says sufficient: {sufficient}")
    print(f"  - Reason: {refine_info.get('reason', '')}")
    if not sufficient and refined_queries:
        print("  - Refined queries suggested:")
        for q in refined_queries:
            print(f"    * {q}")

    # Step 3: optional refined retrieval
    all_docs = list(docs)
    all_metas = list(metas)

    if not sufficient and refined_queries:
        print("[2b/3] Performing refined retrievals...")
        for rq in refined_queries:
            more_docs, more_metas, _ = _retrieve(collection, question=rq, n_results=refined_k)
            all_docs.extend(more_docs)
            all_metas.extend(more_metas)

    # Remove duplicates by (scheme_name, page, text)
    seen = set()
    uniq_docs: List[str] = []
    uniq_metas: List[Dict[str, Any]] = []
    for d, m in zip(all_docs, all_metas):
        key = (m.get("scheme_name"), m.get("page"), d.strip())
        if key in seen:
            continue
        seen.add(key)
        uniq_docs.append(d)
        uniq_metas.append(m)

    merged_context_block = _format_context_block(uniq_docs, uniq_metas)

    # Final answer
    print("[3/3] Asking Gemini for final answer...")
    try:
        answer = _ask_gemini_final_answer(
            client, llm_model=llm_model, question=question, context_block=merged_context_block
        )
    except ClientError as e:
        raise SystemExit(
            f"Gemini final-answer call failed for model '{llm_model}'.\n"
            f"Error: {e}\n"
            "Try a different model with --llm-model (for example: gemini-2.0-flash)."
        )

    print("\n=== ANSWER ===")
    print(answer.strip())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-step orchestrated RAG over schemes DB with Gemini (retrieval + refinement + answer)."
    )
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Chroma persistent directory path.")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name or local path.")
    ap.add_argument(
        "--llm-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model id for refinement + answer steps (default: {DEFAULT_GEMINI_MODEL}).",
    )
    ap.add_argument(
        "--question",
        required=False,
        help="User question to answer. If omitted, you will be prompted at runtime.",
    )
    ap.add_argument("--initial-k", type=int, default=6, help="Initial number of chunks to retrieve.")
    ap.add_argument("--refined-k", type=int, default=4, help="Chunks per refined query.")
    args = ap.parse_args()

    question = args.question
    if not question:
        print("Enter your question about Karnataka schemes.")
        question = input("Question: ").strip()
        if not question:
            raise SystemExit("No question provided.")

    run_orchestrated_rag(
        db_path=args.db_path,
        collection_name=args.collection,
        model_name=args.model,
        llm_model=args.llm_model,
        question=question,
        initial_k=args.initial_k,
        refined_k=args.refined_k,
    )


if __name__ == "__main__":
    main()

