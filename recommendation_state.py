from __future__ import annotations

from typing import Any, Dict, List

from scheme_choice_label import label_for_rank_row


def empty_recommendation_state() -> Dict[str, Any]:
    return {
        "ranked_schemes": [],
        "selected_index": -1,
        "comparison": {},
        "last_answer": "",
    }


def update_from_rag_result(state: Dict[str, Any], rag_result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    out["ranked_schemes"] = rag_result.get("pdf_scheme_choices") or rag_result.get("ranked_schemes") or []
    out["last_answer"] = rag_result.get("answer") or ""
    if not out["ranked_schemes"]:
        out["selected_index"] = -1
    elif out.get("selected_index", -1) < 0:
        out["selected_index"] = 0
    return out


def recommendation_lines(state: Dict[str, Any], limit: int = 8) -> List[str]:
    rows = list((state or {}).get("ranked_schemes") or [])[:limit]
    return [label_for_rank_row(r, index_1based=i + 1) for i, r in enumerate(rows)]

