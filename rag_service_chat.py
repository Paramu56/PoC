from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL
from orchestrated_rag_schemes import (
    DEFAULT_GEMINI_MODEL,
    _build_retriever,
    _format_context_block,
    _gemini_generate_with_retry,
    _gemini_response_text,
    get_documents_for_scheme_only,
    get_gemini_client,
    match_extracted_names_to_choices,
    run_orchestrated_rag_result,
)
from scheme_choice_label import extract_scheme_title_line, is_fallback_scheme_metadata_name
from user_profile import compose_rag_question

_NUM_SCHEME_LINE = re.compile(r"^\s*(\d+)[.)]\s*(.+)$")


def _is_false_positive_numbered_line(rest: str) -> bool:
    """
    Lines like '1.20 Lakh' or '2.5%' are not scheme titles — avoid treating as '1.' / '2.' schemes.
    """
    r = rest.strip()
    if len(r) < 6:
        return True
    # Decimal amounts: "20 Lakh", "5% interest" after "1." from "1.20"
    if re.match(r"^\d+[.,]\d", r):
        return True
    if re.match(r"^\d+\s*(lakh|crore|million|%|rupees?|₹|years?\s*old)\b", r, re.I):
        return True
    return False


def extract_numbered_scheme_titles_from_answer(answer: str) -> List[str]:
    """
    Parse the assistant's numbered list (1. Scheme A / 2. Scheme B) in answer order.
    Ignores generic 'Karnataka Schemes (page N)' lines and obvious non-title bullets.
    """
    by_num: Dict[int, str] = {}
    text = (answer or "").replace("\u00a0", " ").replace("．", ".").replace("（", "(").replace("）", ")")
    for ln in text.splitlines():
        m = _NUM_SCHEME_LINE.match(ln.strip())
        if not m:
            continue
        num = int(m.group(1))
        rest = m.group(2).strip()
        if num < 1 or num > 32:
            continue
        if _is_false_positive_numbered_line(rest):
            continue
        if re.match(r"^Karnataka Schemes \(page", rest, re.I):
            continue
        if re.match(
            r"^(Note:|If you|Missing|Please|Reference|Benefits?:|Eligibility:|Documents?:)\b",
            rest,
            re.I,
        ):
            continue
        if re.match(
            r"^(Age|Educational|Income|Annual|Must|Should|The\s+applicant|Family|Below|Above|Resident|"
            r"Domicile|BPL|Aadhaar|Bank|Caste|Certificate)\b",
            rest,
            re.I,
        ):
            continue
        if num not in by_num:
            by_num[num] = rest
    if not by_num:
        return []
    return [by_num[k] for k in sorted(by_num.keys())]


def _norm_blob(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _pdf_row_from_ranked(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scheme_name": r.get("scheme_name") or "",
        "page": r.get("page"),
        "snippet": r.get("snippet"),
        "text_for_label": r.get("text_for_label"),
        "meta": dict(r.get("meta") or {}),
    }


def _dedupe_scheme_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for ch in rows:
        key = (
            (ch.get("scheme_name") or "").strip(),
            str(ch.get("page") or ""),
            ((ch.get("snippet") or "")[:48]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ch)
    return out


def _score_title_against_row(title: str, exn: str, ch: Dict[str, Any]) -> float:
    """
    How well an answer title (e.g. numbered list item) matches a retrieval row, using chunk text
    — not only metadata scheme_name (often 'Karnataka Schemes (page N)').
    """
    blob = _norm_blob((ch.get("text_for_label") or "") + "\n" + (ch.get("snippet") or ""))
    if not blob:
        return 0.0
    if len(exn) >= 12 and exn in blob:
        return 0.98
    head = exn[: min(56, len(exn))]
    if len(head) >= 12 and head in blob:
        return 0.93
    hits = 0
    total = 0
    for tok in re.findall(r"[a-z0-9]+", exn):
        if len(tok) < 4:
            continue
        total += 1
        if tok in blob:
            hits += 1
    if total > 0:
        r = hits / total
        if r >= 0.45:
            return 0.5 + 0.45 * r
    mined = _norm_blob(extract_scheme_title_line((ch.get("text_for_label") or "")[:3500]))
    if mined and len(mined) > 8 and (exn in mined or mined in exn):
        return 0.9
    sm = SequenceMatcher(None, exn, blob[: min(len(blob), 520)]).ratio()
    base = sm * 0.82
    if is_fallback_scheme_metadata_name((ch.get("scheme_name") or "").strip()):
        base *= 0.72
    return base


def align_extracted_titles_to_scheme_rows(
    extracted_names: List[str],
    context_choices: List[Dict[str, Any]],
    ranked_schemes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Map each answer title (in list order) to one retrieval row. Uses merged candidates from
    tight RAG context + broad ranked list so we match on real chunk text, not only metadata labels.
    """
    pool: List[Dict[str, Any]] = _dedupe_scheme_rows(list(context_choices) + [_pdf_row_from_ranked(r) for r in ranked_schemes])
    if not pool or not extracted_names:
        return []

    used: set[int] = set()
    out: List[Dict[str, Any]] = []

    for title in extracted_names:
        exn = _norm_blob(title)
        if len(exn) < 4:
            continue
        scored: List[Tuple[float, int, bool]] = []
        for i, ch in enumerate(pool):
            if i in used:
                continue
            sc = _score_title_against_row(title, exn, ch)
            generic = is_fallback_scheme_metadata_name((ch.get("scheme_name") or "").strip())
            scored.append((sc, i, generic))
        if not scored:
            break
        scored.sort(key=lambda t: -t[0])
        best_sc, best_i, best_gen = scored[0]
        chosen_i = best_i
        # If the top match is only a generic page-label row, prefer a close non-generic match.
        if best_gen:
            for sc, i, gen in scored[1:24]:
                if not gen and sc >= max(0.26, best_sc - 0.14):
                    chosen_i = i
                    best_sc = sc
                    break
        if best_sc < 0.24:
            # Do not stop early: a weak match must still consume a row so "scheme N" in chat
            # stays aligned with `pdf_scheme_choices` (otherwise pick-by-number fails for later items).
            weak_i, weak_gen = scored[0][1], scored[0][2]
            if weak_gen:
                for sc, i, gen in scored[1:32]:
                    if not gen:
                        weak_i = i
                        break
            chosen_i = weak_i
        used.add(chosen_i)
        out.append(pool[chosen_i])

    return out


def _extract_scheme_names_from_answer(answer: str, context_choices: List[Dict[str, Any]]) -> List[str]:
    text = (answer or "").lower()
    out: List[str] = []
    for row in context_choices:
        cand = (row.get("scheme_name") or "").strip()
        if cand and cand.lower() in text and cand not in out:
            out.append(cand)
            continue
        title = extract_scheme_title_line(row.get("text_for_label") or row.get("snippet") or "")
        if title and title.lower() in text and cand and cand not in out:
            out.append(cand)
    if out:
        return out
    # Fallback to first context schemes.
    seen = set()
    for row in context_choices:
        c = (row.get("scheme_name") or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= 6:
            break
    return out


def _build_selected_scheme_prompt(question: str, selected_scheme_name: str, context_block: str) -> str:
    return (
        "You are helping a citizen with ONE selected Karnataka scheme.\n\n"
        f"SELECTED SCHEME (DO NOT SWITCH): {selected_scheme_name}\n\n"
        "Hard rules:\n"
        "- Answer ONLY about the selected scheme.\n"
        "- If the user asks about other schemes, say they must start a new discovery search.\n"
        "- If context is missing, clearly state what is missing.\n"
        "- Include page references where available.\n\n"
        f"CONTEXT (selected scheme only):\n{context_block}\n\n"
        f"USER QUESTION:\n{question}"
    )


def discover_schemes(
    *,
    user_message: str,
    profile: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    llm_model: str = DEFAULT_GEMINI_MODEL,
    initial_k: int = 6,
    refined_k: int = 4,
    ranked_n: int = 24,
) -> Dict[str, Any]:
    """
    Same Gemini + RAG pipeline as poc_streamlit / run_orchestrated_rag_result (not LLMGateway),
    then chat-specific alignment of numbered scheme titles to retrieval rows.
    """
    question = compose_rag_question(user_message, profile or {})
    result = run_orchestrated_rag_result(
        db_path=db_path,
        collection_name=collection_name,
        model_name=model_name,
        llm_model=llm_model,
        question=question,
        initial_k=initial_k,
        refined_k=refined_k,
        ranked_n=ranked_n,
        use_refinement_llm=False,
    )

    answer = result.get("answer") or ""
    context_choices = result.get("pdf_scheme_choices_context_only") or []
    ranked_schemes = result.get("ranked_schemes") or []

    numbered = extract_numbered_scheme_titles_from_answer(answer)
    if len(numbered) >= 2:
        extracted_names = numbered
    elif len(numbered) == 1 and not re.search(r"^\s*2[.)]\s", answer or "", re.MULTILINE):
        extracted_names = numbered
    else:
        extracted_names = list(result.get("answer_extracted_scheme_names") or [])
        if not extracted_names:
            extracted_names = _extract_scheme_names_from_answer(answer, context_choices)

    pdf_choices = list(result.get("pdf_scheme_choices") or [])
    if extracted_names:
        aligned = align_extracted_titles_to_scheme_rows(extracted_names, context_choices, ranked_schemes)
        if len(aligned) == len(extracted_names):
            pdf_choices = aligned
        else:
            matched = match_extracted_names_to_choices(extracted_names, context_choices)
            if matched:
                pdf_choices = matched

    out: Dict[str, Any] = dict(result)
    out["answer_extracted_scheme_names"] = extracted_names
    out["pdf_scheme_choices"] = pdf_choices
    dbg = dict(result.get("debug") or {})
    dbg["chat_scheme_alignment"] = {
        "used_numbered_titles": bool(numbered),
        "pdf_choices_final": len(pdf_choices),
    }
    out["debug"] = dbg
    return out


def answer_for_selected_scheme(
    *,
    user_message: str,
    selected_scheme_name: str,
    profile: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    llm_model: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    question = compose_rag_question(user_message, profile or {})
    collection = _build_retriever(db_path=db_path, collection_name=collection_name, model_name=model_name)
    docs, metas = get_documents_for_scheme_only(collection, selected_scheme_name, limit=60)
    if not docs:
        return {
            "answer": (
                f"I could not find indexed content for `{selected_scheme_name}` in the local database. "
                "Please run a fresh search to choose another scheme."
            ),
            "pdf_scheme_choices": [],
            "answer_extracted_scheme_names": [selected_scheme_name],
            "citations": [],
        }

    uniq_docs: List[str] = []
    uniq_metas: List[Dict[str, Any]] = []
    seen = set()
    for d, m in zip(docs, metas):
        key = (m.get("scheme_name"), m.get("page"), d.strip())
        if key in seen:
            continue
        seen.add(key)
        uniq_docs.append(d)
        uniq_metas.append(m)

    context_block = _format_context_block(uniq_docs[:24], uniq_metas[:24])
    prompt = _build_selected_scheme_prompt(question, selected_scheme_name, context_block)
    client = get_gemini_client()
    resp = _gemini_generate_with_retry(client, model=llm_model, contents=prompt)
    answer = (_gemini_response_text(resp) or "").strip()

    citations: List[Dict[str, Any]] = []
    for m in uniq_metas[:12]:
        citations.append(
            {
                "scheme_name": m.get("scheme_name"),
                "page": m.get("page"),
                "source": m.get("source"),
            }
        )

    return {
        "answer": answer,
        "pdf_scheme_choices": [
            {
                "scheme_name": selected_scheme_name,
                "page": uniq_metas[0].get("page"),
                "snippet": (uniq_docs[0][:480] + "…") if len(uniq_docs[0]) > 480 else uniq_docs[0],
                "text_for_label": uniq_docs[0][:4500] if len(uniq_docs[0]) > 4500 else uniq_docs[0],
                "meta": dict(uniq_metas[0]),
            }
        ],
        "answer_extracted_scheme_names": [selected_scheme_name],
        "citations": citations,
        "debug": {"selected_scheme_locked": True, "llm_model_used": llm_model},
    }

