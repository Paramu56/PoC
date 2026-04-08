import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

from scheme_choice_label import extract_scheme_title_line

import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai.errors import ClientError, ServerError

from graph_knowledge import graph_addon_for_metadatas
from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# Minimum seconds between Gemini API calls (reduces RPM burst 429s on free/low tiers).
_GEMINI_LAST_CALL_MONO: float = 0.0


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
    """
    Build the RAG context string for Gemini. Each chunk is truncated so large PDF slices
    do not blow past token-per-minute quotas (429). Override with GEMINI_CONTEXT_CHUNK_CHARS.
    """
    try:
        max_chunk = int(os.environ.get("GEMINI_CONTEXT_CHUNK_CHARS", "2500"))
    except ValueError:
        max_chunk = 2500
    max_chunk = max(400, min(max_chunk, 48000))

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
        body = doc.strip()
        if len(body) > max_chunk:
            body = body[:max_chunk] + "\n[… truncated for API size; lower GEMINI_CONTEXT_CHUNK_CHARS is stricter …]"
        lines.append(body)
        lines.append("")  # blank line
    return "\n".join(lines).strip()


def ranked_unique_schemes_by_retrieval(
    collection,
    question: str,
    *,
    n_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Deduplicate retrieved chunks by scheme_name; keep the strongest match (lowest Chroma distance).
    Sorted best-first (ascending distance). Cosine space: lower distance = more similar.
    """
    docs, metas, dists = _retrieve(collection, question=question, n_results=n_results)
    best: Dict[str, Dict[str, Any]] = {}
    for doc, meta, dist in zip(docs, metas, dists):
        name = str(meta.get("scheme_name") or "Unknown").strip() or "Unknown"
        dist_f = float(dist)
        if name not in best or dist_f < float(best[name]["distance"]):
            best[name] = {
                "scheme_name": name,
                "distance": dist_f,
                "page": meta.get("page"),
                "snippet": (doc[:480] + "…") if len(doc) > 480 else doc,
                # Longer slice so UI can find a real scheme title (not "1. Age:…")
                "text_for_label": doc[:4500] if len(doc) > 4500 else doc,
                "meta": dict(meta),
            }
    ranked = sorted(best.values(), key=lambda x: float(x["distance"]))
    return ranked


def pdf_scheme_choices_from_answer_context(
    uniq_docs: List[str],
    uniq_metas: List[Dict[str, Any]],
    ranked_schemes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    One row per scheme that actually appears in the orchestrated RAG context (uniq_metas order),
    not the separate broad ranked_unique_schemes_by_retrieval list. This keeps the PDF picker
    aligned with what Gemini saw (typically a small set, similar to schemes discussed in the answer).
    """
    ranked_map = {r["scheme_name"]: r for r in ranked_schemes if r.get("scheme_name")}
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for d, m in zip(uniq_docs, uniq_metas):
        sn = (m.get("scheme_name") or "").strip()
        if not sn or sn in seen:
            continue
        seen.add(sn)
        base = ranked_map.get(sn)
        row: Dict[str, Any] = {
            "scheme_name": sn,
            "page": m.get("page"),
            "snippet": (d[:480] + "…") if len(d) > 480 else d,
            "text_for_label": d[:4500] if len(d) > 4500 else d,
            "meta": dict(m),
        }
        if base:
            row["distance"] = base.get("distance")
        out.append(row)
    return out


def _gemini_response_text(resp: Any) -> str:
    text = ""
    try:
        if hasattr(resp, "text") and isinstance(resp.text, str):
            text = resp.text
        elif hasattr(resp, "candidates"):
            cands = getattr(resp, "candidates") or []
            if cands:
                content = getattr(cands[0], "content", None)
                if content and getattr(content, "parts", None):
                    part0 = content.parts[0]
                    text = getattr(part0, "text", "") or ""
    except Exception:
        pass
    return text or ""


def _json_object_from_model_text(text: str) -> Dict[str, Any]:
    if not (text or "").strip():
        return {}
    t = text.strip()
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b != -1 and b > a:
        t = t[a : b + 1]
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def extract_scheme_names_from_answer_text(
    client: genai.Client,
    llm_model: str,
    answer_text: str,
    fallback_llm_model: Optional[str],
) -> List[str]:
    """
    Second Gemini call: JSON list of scheme names as actually discussed in the final answer prose.
    """
    prompt = dedent(
        f"""
        You extract Karnataka government scheme names from an assistant answer.

        ANSWER TEXT:
        {answer_text}

        Respond with ONLY valid JSON (no markdown code fences):
        {{"schemes": ["first scheme name", "second scheme name", "..."]}}

        Rules:
        - Put **each scheme in its own JSON string** — never one comma-separated string; use one array element per scheme.
        - Include every scheme the answer explicitly names or clearly recommends (if it discusses four schemes, the array must have four strings).
        - Order by first mention in the answer.
        - If the answer does not name any specific scheme, return {{"schemes": []}}.
        - Do not use generic entries like "Karnataka schemes" or "government welfare".
        """
    )
    try:
        resp = _gemini_generate_with_retry(client, model=llm_model, contents=prompt)
    except (ClientError, ServerError) as e:
        if (
            fallback_llm_model
            and fallback_llm_model != llm_model
            and _gemini_fallback_might_help(e)
        ):
            resp = _gemini_generate_with_retry(client, model=fallback_llm_model, contents=prompt)
        else:
            raise
    data = _json_object_from_model_text(_gemini_response_text(resp))
    schemes = data.get("schemes")
    return _flatten_extracted_scheme_names(schemes)


def _flatten_extracted_scheme_names(schemes: Any) -> List[str]:
    """Normalize Gemini output: list of strings, or one string with commas, or stray types."""
    out: List[str] = []
    if schemes is None:
        return out
    if isinstance(schemes, str):
        schemes = re.split(r"[,;\n]\s*", schemes)
    if not isinstance(schemes, list):
        return out
    for s in schemes:
        if not isinstance(s, str):
            continue
        for part in re.split(r"[,;\n]\s*", s.strip()):
            p = part.strip()
            if not p:
                continue
            if p.lower() in ("none", "n/a", "na"):
                continue
            out.append(p)
    # Dedupe preserving order
    seen: set = set()
    uniq: List[str] = []
    for x in out:
        k = _norm_label(x)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def match_extracted_names_to_choices(
    extracted_names: List[str],
    context_choices: List[Dict[str, Any]],
    *,
    min_fuzzy_ratio: float = 0.42,
) -> List[Dict[str, Any]]:
    """
    Map each extracted name onto a distinct context row (one picker row per scheme in the answer).
    Uses tiered thresholds; if nothing clears the bar, assigns the next unused context row so
    counts stay aligned with extraction.
    """

    def _score_name_against_row(exn: str, ch: Dict[str, Any]) -> float:
        best = 0.0
        haystack = [
            ch.get("scheme_name", ""),
            extract_scheme_title_line(ch.get("text_for_label") or ""),
            (ch.get("snippet") or "")[:500],
        ]
        for cand in haystack:
            cn = _norm_label(cand)
            if not cn:
                continue
            if exn == cn:
                return 1.0
            if exn in cn or cn in exn:
                best = max(best, 0.92)
            elif len(exn) > 5 and exn in cn:
                best = max(best, 0.88)
            else:
                best = max(best, SequenceMatcher(None, exn, cn).ratio())
        return best

    if not extracted_names or not context_choices:
        return []
    out: List[Dict[str, Any]] = []
    used: set = set()
    thresholds = (min_fuzzy_ratio, max(0.22, min_fuzzy_ratio * 0.55), 0.18)

    for ex in extracted_names:
        exn = _norm_label(ex)
        if len(exn) < 2:
            continue
        best_i: Optional[int] = None
        best_score = 0.0
        for i, ch in enumerate(context_choices):
            if i in used:
                continue
            sc = _score_name_against_row(exn, ch)
            if sc > best_score:
                best_score = sc
                best_i = i
        picked: Optional[int] = None
        for th in thresholds:
            if best_i is not None and best_score >= th:
                picked = best_i
                break
        if picked is not None:
            used.add(picked)
            out.append(context_choices[picked])
            continue
        # No score passed: still consume one context row in order (keeps N names ≈ N rows)
        for i, ch in enumerate(context_choices):
            if i in used:
                continue
            used.add(i)
            out.append(ch)
            break
    return out


def retrieve_context_for_scheme(
    collection,
    scheme_name: str,
    *,
    n_results: int = 8,
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    """
    Semantic retrieval biased toward a scheme. May include other schemes' chunks — do not use alone for PDF.
    """
    q = f"{scheme_name} Karnataka government scheme documents eligibility how to apply"
    return _retrieve(collection, question=q, n_results=n_results)


def get_documents_for_scheme_only(
    collection,
    scheme_name: str,
    *,
    limit: int = 48,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    All indexed chunks whose metadata scheme_name equals this scheme (exact match).
    Sorted by PDF page number, then document id. Use this for PDF text so only the chosen scheme appears.
    """
    target = (scheme_name or "").strip()
    if not target:
        return [], []

    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    pairs: List[Tuple[str, Dict[str, Any]]] = []
    try:
        got = collection.get(
            where={"scheme_name": target},
            limit=limit,
            include=["documents", "metadatas"],
        )
        raw_docs = (got or {}).get("documents") or []
        raw_metas = (got or {}).get("metadatas") or []
        if raw_docs and raw_metas and len(raw_docs) == len(raw_metas):
            pairs = list(zip(raw_docs, raw_metas))
    except Exception:
        pass

    if not pairs:
        q = f"{target} Karnataka government scheme"
        more_docs, more_metas, _ = _retrieve(collection, question=q, n_results=min(limit * 3, 120))
        for d, m in zip(more_docs, more_metas):
            if (m.get("scheme_name") or "").strip() == target:
                pairs.append((d, m))
        seen = set()
        uniq_pairs: List[Tuple[str, Dict[str, Any]]] = []
        for d, m in pairs:
            key = (m.get("page"), hash((d or "")[:200]))
            if key in seen:
                continue
            seen.add(key)
            uniq_pairs.append((d, m))
        pairs = uniq_pairs

    def _page_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        m = item[1]
        try:
            p = int(m.get("page") or 0)
        except (TypeError, ValueError):
            p = 0
        return (p, (item[0] or "")[:40])

    pairs.sort(key=_page_key)
    for d, m in pairs:
        docs.append(d)
        metas.append(m)

    return docs, metas


def get_gemini_client() -> genai.Client:
    """Return a Gemini client. Raises RuntimeError if GEMINI_API_KEY is missing (for programmatic callers)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it to your Gemini API key in the environment."
        )
    return genai.Client(api_key=api_key)


def _init_gemini_client() -> genai.Client:
    try:
        return get_gemini_client()
    except RuntimeError as e:
        raise SystemExit(str(e))


def _gemini_pace_if_configured() -> None:
    """Space out requests; set GEMINI_CALL_SPACING_S=0 to disable."""
    global _GEMINI_LAST_CALL_MONO
    try:
        spacing = float(os.environ.get("GEMINI_CALL_SPACING_S", "6.0"))
    except ValueError:
        spacing = 6.0
    if spacing <= 0:
        return
    now = time.monotonic()
    if _GEMINI_LAST_CALL_MONO > 0:
        gap = spacing - (now - _GEMINI_LAST_CALL_MONO)
        if gap > 0:
            time.sleep(gap)


def _gemini_mark_call_done() -> None:
    global _GEMINI_LAST_CALL_MONO
    _GEMINI_LAST_CALL_MONO = time.monotonic()


def _gemini_error_is_transient(exc: BaseException) -> bool:
    """
    Only true overload / quota / short server blips. Do not use loose substring
    matching on str(exc): the SDK embeds full JSON details and false positives
    caused endless 'rate-limited' backoff on unrelated 4xx errors.
    """
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    st = (status or "").upper() if isinstance(status, str) else ""

    if isinstance(exc, ServerError):
        return code in (500, 502, 503, 504) or st in ("UNAVAILABLE", "DEADLINE_EXCEEDED")

    if isinstance(exc, ClientError):
        if code == 429:
            return True
        if st in ("RESOURCE_EXHAUSTED", "UNAVAILABLE"):
            return True
        return False

    return False


def _gemini_fallback_might_help(exc: BaseException) -> bool:
    """
    After a failed Gemini call, trying a fallback model usually does NOT help when the
    failure is quota (429): the same API key shares limits across models. Only fall back
    for errors that suggest the primary model id is wrong or unavailable for this project.
    """
    if isinstance(exc, ServerError):
        code = getattr(exc, "code", None)
        return code in (500, 502, 503, 504)

    if not isinstance(exc, ClientError):
        return False

    code = getattr(exc, "code", None)
    status = (getattr(exc, "status", None) or "").upper()

    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return False
    if code in (401, 403):
        return False
    if code in (400, 404):
        return True
    if status in ("NOT_FOUND", "INVALID_ARGUMENT", "FAILED_PRECONDITION"):
        return True
    return False


def _gemini_generate_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents: str,
    max_attempts: Optional[int] = None,
    initial_wait_s: Optional[float] = None,
) -> Any:
    """
    Retry Gemini calls only for HTTP 429 / RESOURCE_EXHAUSTED and 5xx blips.
    Other ClientError (400 invalid model, 401 key, 403 permission, etc.) fail fast.
    """
    if initial_wait_s is None:
        try:
            initial_wait_s = float(os.environ.get("GEMINI_429_INITIAL_WAIT_S", "4.0"))
        except ValueError:
            initial_wait_s = 4.0
    if max_attempts is None:
        try:
            max_attempts = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
        except ValueError:
            max_attempts = 5
    max_attempts = max(1, min(int(max_attempts), 12))
    wait_s = float(initial_wait_s)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            _gemini_pace_if_configured()
            out = client.models.generate_content(model=model, contents=contents)
            _gemini_mark_call_done()
            return out
        except (ClientError, ServerError) as e:
            last_error = e
            if not _gemini_error_is_transient(e) or attempt == max_attempts:
                raise
            code = getattr(e, "code", "?")
            status = getattr(e, "status", "") or ""
            print(
                f"  - Gemini transient error HTTP {code} {status} "
                f"(attempt {attempt}/{max_attempts}). Backing off {wait_s:.1f}s..."
            )
            time.sleep(wait_s)
            wait_s = min(wait_s * 2, 120.0)

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


def _strip_optional_json_fence(block: str) -> str:
    s = (block or "").strip()
    if not s.startswith("```"):
        return s
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, re.IGNORECASE)
    return m.group(1).strip() if m else s


def _split_final_reply_and_scheme_list(full_text: str) -> Tuple[str, List[str]]:
    """
    Final model reply = citizen-facing prose, then a trailing JSON object with key "schemes".
    Avoids a separate Gemini call for PDF picker alignment (saves quota / RPM).
    """
    text = full_text or ""
    if not text.strip():
        return "", []
    lines = text.splitlines()
    for n_tail in range(1, min(14, len(lines) + 1)):
        tail = "\n".join(lines[-n_tail:]).strip()
        if not tail:
            continue
        if not tail.lstrip().startswith("{") and not tail.lstrip().startswith("```"):
            continue
        tail = _strip_optional_json_fence(tail)
        data = _json_object_from_model_text(tail)
        schemes = data.get("schemes")
        if schemes is None or not isinstance(schemes, list):
            continue
        answer = "\n".join(lines[:-n_tail]).rstrip()
        return answer, _flatten_extracted_scheme_names(schemes)
    return text.strip(), []


def _ask_gemini_final_answer_with_scheme_list(
    client: genai.Client,
    llm_model: str,
    question: str,
    context_block: str,
) -> Tuple[str, List[str]]:
    prompt = dedent(
        f"""
        You are an expert assistant on Karnataka government schemes.

        Use ONLY the provided context to answer the user's question.
        If the answer is not clearly supported by the context, say so and explain what is missing.

        The context may include a KNOWLEDGE GRAPH section (compiled from schemes.cypher). Use it for
        exclusions, relationships, and eligibility rules when they apply to the schemes mentioned.

        Always:
        - mention the scheme names you are using
        - where possible, cite page numbers from the context in parentheses, e.g., "(page 12)"
        - answer in clear, simple language for citizens

        CONTEXT:
        {context_block}

        USER QUESTION:
        {question}

        After the answer, add ONE final block (can be multiple lines) that contains ONLY valid JSON
        (no markdown code fences, no commentary) with this exact shape:
        {{"schemes": ["<first scheme name mentioned>", "<second>", "..."]}}
        Rules for that JSON:
        - One string per scheme you named in the answer; order by first mention.
        - If you did not name specific schemes, use {{"schemes": []}}.
        - No comma-joined scheme names inside a single string.
        """
    )

    resp = _gemini_generate_with_retry(client, model=llm_model, contents=prompt)
    raw = _gemini_response_text(resp)
    return _split_final_reply_and_scheme_list(raw)


def _ask_gemini_final_answer(
    client: genai.Client,
    llm_model: str,
    question: str,
    context_block: str,
) -> str:
    answer, _ = _ask_gemini_final_answer_with_scheme_list(
        client, llm_model=llm_model, question=question, context_block=context_block
    )
    return answer


def run_orchestrated_rag(
    *,
    db_path: str,
    collection_name: str,
    model_name: str,
    llm_model: str,
    question: str,
    initial_k: int = 6,
    refined_k: int = 4,
    use_refinement_llm: bool = False,
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
    summary_text = "\n".join(
        f"- {m.get('scheme_name', 'UNKNOWN')} (page {m.get('page', '?')})"
        for m in metas
    )
    env_skip_refine = os.environ.get("GEMINI_SKIP_REFINEMENT_LLM", "").strip().lower() in ("1", "true", "yes")
    env_force_refine = os.environ.get("GEMINI_FORCE_REFINEMENT_LLM", "").strip().lower() in ("1", "true", "yes")
    skip_refine = env_skip_refine or (not use_refinement_llm and not env_force_refine)
    if skip_refine:
        print("[2/3] Skipping refinement LLM (default off; set --refinement-llm or GEMINI_FORCE_REFINEMENT_LLM=1).")
        class _StaticTextResp:
            text = '{"sufficient": true, "reason": "skipped", "refined_queries": []}'

        raw_refine = _StaticTextResp()
    else:
        print("[2/3] Asking Gemini if more context is needed...")
        try:
            raw_refine = _ask_gemini_for_refinements(
                client, llm_model=llm_model, question=question, context_summary=summary_text
            )
        except (ClientError, ServerError) as e:
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
    graph_addon = graph_addon_for_metadatas(uniq_metas)
    if graph_addon:
        merged_context_block = f"{merged_context_block}\n\n{graph_addon}"
        print("  - appended knowledge-graph context from graph_compiled.json")

    # Final answer
    print("[3/3] Asking Gemini for final answer...")
    try:
        answer = _ask_gemini_final_answer(
            client, llm_model=llm_model, question=question, context_block=merged_context_block
        )
    except (ClientError, ServerError) as e:
        raise SystemExit(
            f"Gemini final-answer call failed for model '{llm_model}'.\n"
            f"Error: {e}\n"
            "Try a different model with --llm-model (for example: gemini-2.0-flash)."
        )

    print("\n=== ANSWER ===")
    print(answer.strip())


def run_orchestrated_rag_result(
    *,
    db_path: str,
    collection_name: str,
    model_name: str,
    llm_model: str,
    question: str,
    initial_k: int = 6,
    refined_k: int = 4,
    fallback_llm_model: Optional[str] = "gemini-pro-latest",
    ranked_n: int = 24,
    use_refinement_llm: bool = False,
) -> Dict[str, Any]:
    """
    Same pipeline as run_orchestrated_rag (retrieve → optional Gemini refinement → optional refined retrieve → graph → final answer),
    but returns structured data for UIs instead of printing.

    ranked_schemes: broad retrieval (debug). pdf_scheme_choices_context_only: schemes in RAG context.
    Scheme names for the PDF picker are parsed from the final LLM reply (trailing JSON); no extra Gemini call.

    use_refinement_llm: default False to limit quota (one generate_content per run unless env forces otherwise).
    """
    collection = _build_retriever(db_path=db_path, collection_name=collection_name, model_name=model_name)
    client = get_gemini_client()

    ranked_schemes = ranked_unique_schemes_by_retrieval(collection, question, n_results=ranked_n)

    docs, metas, _ = _retrieve(collection, question=question, n_results=initial_k)
    if not docs:
        return {
            "answer": "No matching content in the local scheme database. Ingest a PDF and rebuild the Chroma index.",
            "ranked_schemes": ranked_schemes,
            "pdf_scheme_choices": [],
            "citations": [],
            "refine": {"sufficient": True, "reason": "No chunks to assess.", "refined_queries": []},
            "debug": {"empty_retrieval": True, "retrieved_chunks": 0, "graph_appended": False},
            "answer_extracted_scheme_names": [],
        }

    summary_text = "\n".join(
        f"- {m.get('scheme_name', 'UNKNOWN')} (page {m.get('page', '?')})" for m in metas
    )

    def _refine_with_model(model_id: str):
        return _ask_gemini_for_refinements(
            client, llm_model=model_id, question=question, context_summary=summary_text
        )

    env_skip_refine = os.environ.get("GEMINI_SKIP_REFINEMENT_LLM", "").strip().lower() in ("1", "true", "yes")
    env_force_refine = os.environ.get("GEMINI_FORCE_REFINEMENT_LLM", "").strip().lower() in ("1", "true", "yes")
    skip_refine = env_skip_refine or (not use_refinement_llm and not env_force_refine)
    if skip_refine:
        class _StaticTextResp:
            text = (
                '{"sufficient": true, "reason": "Refinement LLM skipped (default off or GEMINI_SKIP_REFINEMENT_LLM).", '
                '"refined_queries": []}'
            )

        raw_refine = _StaticTextResp()
    else:
        try:
            raw_refine = _refine_with_model(llm_model)
        except (ClientError, ServerError) as e:
            if (
                fallback_llm_model
                and fallback_llm_model != llm_model
                and _gemini_fallback_might_help(e)
            ):
                raw_refine = _ask_gemini_for_refinements(
                    client, llm_model=fallback_llm_model, question=question, context_summary=summary_text
                )
            else:
                raise

    refine_info = _safe_extract_json_from_gemini(raw_refine)
    sufficient = bool(refine_info.get("sufficient", True))
    refined_queries: List[str] = list(refine_info.get("refined_queries") or [])

    all_docs = list(docs)
    all_metas = list(metas)
    if not sufficient and refined_queries:
        for rq in refined_queries:
            more_docs, more_metas, _ = _retrieve(collection, question=rq, n_results=refined_k)
            all_docs.extend(more_docs)
            all_metas.extend(more_metas)

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
    graph_addon = graph_addon_for_metadatas(uniq_metas)
    if graph_addon:
        merged_context_block = f"{merged_context_block}\n\n{graph_addon}"

    def _final_with_model(model_id: str) -> Tuple[str, List[str]]:
        return _ask_gemini_final_answer_with_scheme_list(
            client, llm_model=model_id, question=question, context_block=merged_context_block
        )

    try:
        answer, extracted_names = _final_with_model(llm_model)
        used_final = llm_model
    except (ClientError, ServerError) as e:
        if (
            fallback_llm_model
            and fallback_llm_model != llm_model
            and _gemini_fallback_might_help(e)
        ):
            answer, extracted_names = _final_with_model(fallback_llm_model)
            used_final = fallback_llm_model
        else:
            raise

    citations: List[Dict[str, Any]] = []
    for m in uniq_metas[:12]:
        citations.append(
            {
                "scheme_name": m.get("scheme_name"),
                "page": m.get("page"),
                "source": m.get("source"),
            }
        )

    context_choices = pdf_scheme_choices_from_answer_context(uniq_docs, uniq_metas, ranked_schemes)

    pdf_scheme_choices = context_choices
    if extracted_names:
        matched = match_extracted_names_to_choices(extracted_names, context_choices)
        if matched:
            pdf_scheme_choices = matched

    return {
        "answer": answer.strip(),
        "ranked_schemes": ranked_schemes,
        "pdf_scheme_choices": pdf_scheme_choices,
        "answer_extracted_scheme_names": extracted_names,
        "pdf_scheme_choices_context_only": context_choices,
        "citations": citations,
        "refine": {
            "sufficient": sufficient,
            "reason": refine_info.get("reason", ""),
            "refined_queries": refined_queries,
        },
        "debug": {
            "empty_retrieval": False,
            "retrieved_chunks": len(uniq_docs),
            "graph_appended": bool(graph_addon),
            "llm_model_used": used_final,
            "answer_extraction_count": len(extracted_names),
            "pdf_choices_context_count": len(context_choices),
            "pdf_choices_after_answer_filter": len(pdf_scheme_choices),
            "schemes_extracted_in_final_llm_call": True,
            "gemini_skip_refinement_llm": skip_refine,
            "gemini_use_refinement_llm_param": use_refinement_llm,
        },
    }


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
    ap.add_argument(
        "--refinement-llm",
        action="store_true",
        help="Run the refinement Gemini step (extra API call; default is off to reduce quota use).",
    )
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
        use_refinement_llm=args.refinement_llm,
    )


if __name__ == "__main__":
    main()

