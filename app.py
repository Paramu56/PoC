import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL
from orchestrated_rag_schemes import run_orchestrated_rag_result
from user_profile import compose_rag_question


DEFAULT_LLM_MODEL = "gemini-flash-latest"
FALLBACK_LLM_MODEL = "gemini-pro-latest"

app = Flask(__name__)


def answer_question(question: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("Question is required.")

    composed_question = compose_rag_question(question.strip(), profile)

    result = run_orchestrated_rag_result(
        db_path=DEFAULT_DB_PATH,
        collection_name=DEFAULT_COLLECTION,
        model_name=DEFAULT_MODEL,
        llm_model=DEFAULT_LLM_MODEL,
        question=composed_question,
        initial_k=6,
        refined_k=4,
        fallback_llm_model=FALLBACK_LLM_MODEL,
    )
    refine = result.get("refine") or {}
    dbg = result.get("debug") or {}
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "debug": {
            "sufficient": refine.get("sufficient"),
            "refined_queries": refine.get("refined_queries"),
            "retrieved_chunks": dbg.get("retrieved_chunks"),
            "graph_context_appended": dbg.get("graph_appended"),
            "llm_model_used": dbg.get("llm_model_used"),
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
    # Set GEMINI_API_KEY in your terminal before running.
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY is not set. The UI will error until you set it.")
    app.run(host="127.0.0.1", port=8000, debug=True)

