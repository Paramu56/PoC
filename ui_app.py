import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL
from orchestrated_rag_schemes import (
    _ask_gemini_final_answer,
    _ask_gemini_for_refinements,
    _build_retriever,
    _format_context_block,
    _init_gemini_client,
    _retrieve,
    _safe_extract_json_from_gemini,
)


DEFAULT_LLM_MODEL = "gemini-flash-latest"

app = Flask(__name__)


def _profile_to_text(profile: Dict[str, Any]) -> str:
    rows = []
    order = [
        "name",
        "gender",
        "age",
        "caste",
        "residence",
        "marital_status",
        "disability_percentage",
        "employment_status",
        "occupation",
        "minority",
        "below_poverty_line",
        "economic_distress",
    ]
    for key in order:
        value = (profile.get(key) or "").strip()
        if value:
            rows.append(f"- {key.replace('_', ' ').title()}: {value}")
    return "\n".join(rows)


def answer_question(question: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("Question is required.")

    collection = _build_retriever(
        db_path=DEFAULT_DB_PATH,
        collection_name=DEFAULT_COLLECTION,
        model_name=DEFAULT_MODEL,
    )
    client = _init_gemini_client()

    profile_text = _profile_to_text(profile)
    composed_question = question
    if profile_text:
        composed_question = (
            f"User profile:\n{profile_text}\n\n"
            f"Question:\n{question}\n\n"
            "Please consider profile constraints while selecting schemes."
        )

    # Step 1: initial retrieval
    docs, metas, _ = _retrieve(collection, question=composed_question, n_results=6)
    if not docs:
        return {"answer": "No matching schemes found in the database.", "debug": {}}

    # Step 2: ask Gemini whether refinement is needed
    summary_text = "\n".join(f"- {m.get('scheme_name', 'UNKNOWN')} (page {m.get('page', '?')})" for m in metas)
    raw_refine = _ask_gemini_for_refinements(
        client,
        llm_model=DEFAULT_LLM_MODEL,
        question=composed_question,
        context_summary=summary_text,
    )
    refine_info = _safe_extract_json_from_gemini(raw_refine)
    sufficient = bool(refine_info.get("sufficient", True))
    refined_queries: List[str] = list(refine_info.get("refined_queries") or [])

    all_docs = list(docs)
    all_metas = list(metas)
    if not sufficient and refined_queries:
        for rq in refined_queries[:3]:
            more_docs, more_metas, _ = _retrieve(collection, question=rq, n_results=4)
            all_docs.extend(more_docs)
            all_metas.extend(more_metas)

    # Deduplicate
    seen = set()
    uniq_docs = []
    uniq_metas = []
    for d, m in zip(all_docs, all_metas):
        key = (m.get("scheme_name"), m.get("page"), d.strip())
        if key in seen:
            continue
        seen.add(key)
        uniq_docs.append(d)
        uniq_metas.append(m)

    context_block = _format_context_block(uniq_docs, uniq_metas)
    answer = _ask_gemini_final_answer(
        client,
        llm_model=DEFAULT_LLM_MODEL,
        question=composed_question,
        context_block=context_block,
    )

    citations = []
    for m in uniq_metas[:12]:
        citations.append(
            {
                "scheme_name": m.get("scheme_name"),
                "page": m.get("page"),
                "source": m.get("source"),
            }
        )

    return {
        "answer": answer.strip(),
        "citations": citations,
        "debug": {
            "sufficient": sufficient,
            "refined_queries": refined_queries,
            "retrieved_chunks": len(uniq_docs),
        },
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/ask")
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    profile = data.get("profile") or {}

    try:
        result = answer_question(question=question, profile=profile)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Usage:
    #   set GEMINI_API_KEY=...
    #   python ui_app.py
    app.run(host="127.0.0.1", port=8000, debug=True)

