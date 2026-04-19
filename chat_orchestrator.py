from __future__ import annotations

from typing import Any, Dict, List

from chat_intents import classify_intent
from profile_state import next_profile_question, update_profile_from_message
from rag_service_chat import (
    align_extracted_titles_to_scheme_rows,
    answer_for_selected_scheme,
    discover_schemes,
    extract_numbered_scheme_titles_from_answer,
)
from recommendation_state import update_from_rag_result
from scheme_choice_label import is_fallback_scheme_metadata_name, label_for_rank_row


def _rebuild_available_from_titles_and_ranked(
    titles: List[str],
    ranked: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """When stored choices are shorter than the numbered list in the last answer, remap titles onto ranked rows."""
    if not titles or not ranked:
        return []
    aligned = align_extracted_titles_to_scheme_rows(titles, [], ranked)
    if not aligned:
        return []
    out: List[Dict[str, str]] = []
    for i, row in enumerate(aligned):
        display_name = ""
        if i < len(titles) and (titles[i] or "").strip():
            display_name = titles[i].strip()
        if not display_name:
            display_name = label_for_rank_row(row, index_1based=None)
        target = (row.get("scheme_name") or "").strip() or display_name
        if is_fallback_scheme_metadata_name(target) and display_name:
            target = display_name
        out.append({"display_name": display_name, "target_scheme_name": target})
    return out


def _safe_selected_scheme(rec_state: Dict[str, Any]) -> str:
    rows = rec_state.get("available_schemes") or []
    idx = int(rec_state.get("selected_index", -1))
    if 0 <= idx < len(rows):
        return rows[idx].get("target_scheme_name") or rows[idx].get("display_name") or ""
    return ""


def _titles_from_last_assistant_message(chat_history: List[Dict[str, str]]) -> List[str]:
    """Recover numbered scheme titles from the latest assistant reply (source of truth for what the user saw)."""
    for m in reversed(chat_history or []):
        if m.get("role") == "assistant":
            return extract_numbered_scheme_titles_from_answer(m.get("content") or "")
    return []


def _is_fresh_discovery_request(text: str) -> bool:
    low = (text or "").strip().lower()
    triggers = (
        "show schemes",
        "find schemes",
        "new search",
        "search again",
        "other schemes",
        "different schemes",
        "all schemes",
        "start over",
    )
    return any(t in low for t in triggers)


def handle_turn(
    *,
    user_message: str,
    session_state: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    profile = update_profile_from_message(session_state.get("profile") or {}, user_message)
    rec_state = dict(session_state.get("recommendations") or {})
    if "available_schemes" not in rec_state:
        rec_state["available_schemes"] = []
    chat_history: List[Dict[str, str]] = list(session_state.get("chat_history") or [])

    intent = classify_intent(user_message)
    assistant_text = ""
    rag_result: Dict[str, Any] | None = None
    action: Dict[str, Any] = {}

    if intent.intent == "smalltalk":
        assistant_text = (
            "Tell me your situation, and I will suggest relevant Karnataka schemes. "
            "You can also ask me to compare schemes."
        )
    elif intent.intent == "select_scheme":
        rows = list(rec_state.get("available_schemes") or [])
        parsed_titles = _titles_from_last_assistant_message(chat_history)
        stored_titles = list(rec_state.get("scheme_titles_from_answer") or [])
        if len(parsed_titles) >= len(stored_titles):
            titles = parsed_titles
        else:
            titles = stored_titles or parsed_titles
        if (
            not (0 <= intent.index < len(rows))
            and titles
            and 0 <= intent.index < len(titles)
            and rec_state.get("ranked_schemes")
        ):
            rebuilt = _rebuild_available_from_titles_and_ranked(
                titles,
                list(rec_state.get("ranked_schemes") or []),
            )
            if rebuilt and intent.index < len(rebuilt) and len(rebuilt) >= len(titles):
                rec_state["available_schemes"] = rebuilt
                rows = rebuilt
        if 0 <= intent.index < len(rows):
            rec_state["selected_index"] = intent.index
            row = rows[intent.index]
            chosen = ""
            if intent.index < len(titles) and (titles[intent.index] or "").strip():
                chosen = titles[intent.index].strip()
            if not chosen:
                chosen = (row.get("display_name") or "").strip()
            if not chosen:
                chosen = (row.get("target_scheme_name") or "").strip()
            if is_fallback_scheme_metadata_name(chosen) and intent.index < len(titles) and (titles[intent.index] or "").strip():
                chosen = titles[intent.index].strip()
            if not chosen:
                chosen = f"scheme {intent.index + 1}"
            assistant_text = f"Selected scheme {intent.index + 1}: `{chosen}`. Say 'generate pdf' when you are ready."
        else:
            assistant_text = "I could not find that scheme number. Please choose from the numbered list in chat."
    elif intent.intent == "compare_schemes":
        rec_state["comparison"] = {"scheme_a": intent.scheme_a, "scheme_b": intent.scheme_b}
        assistant_text = (
            f"I will compare **{intent.scheme_a}** and **{intent.scheme_b}** using current evidence. "
            "If needed, ask one more detailed question and I will refresh recommendations."
        )
    elif intent.intent == "generate_pdf":
        selected = _safe_selected_scheme(rec_state)
        if selected:
            assistant_text = f"Generating PDF for **{selected}**. Please wait..."
            action = {"type": "generate_pdf", "scheme_name": selected}
        else:
            assistant_text = "Please select a scheme first (for example: 'select scheme 1')."
    elif intent.intent == "unlock_scheme":
        rec_state["selected_index"] = -1
        assistant_text = "Selection cleared. You are now back in discovery mode."
    else:
        selected = _safe_selected_scheme(rec_state)
        use_selected_scope = bool(selected and not _is_fresh_discovery_request(intent.payload or user_message))
        previous_available = list(rec_state.get("available_schemes") or [])
        previous_selected_index = int(rec_state.get("selected_index", -1))
        if use_selected_scope:
            rag_result = answer_for_selected_scheme(
                user_message=intent.payload or user_message,
                selected_scheme_name=selected,
                profile=profile,
                db_path=cfg["db_path"],
                collection_name=cfg["collection_name"],
                model_name=cfg["model_name"],
                llm_model=cfg["llm_model"],
            )
        else:
            rag_result = discover_schemes(
                user_message=intent.payload or user_message,
                profile=profile,
                db_path=cfg["db_path"],
                collection_name=cfg["collection_name"],
                model_name=cfg["model_name"],
                llm_model=cfg["llm_model"],
                initial_k=cfg["initial_k"],
                refined_k=cfg["refined_k"],
                ranked_n=cfg["ranked_n"],
            )
        rec_state = update_from_rag_result(rec_state, rag_result)
        extracted = rag_result.get("answer_extracted_scheme_names") or []
        choices = rag_result.get("pdf_scheme_choices") or []
        available: List[Dict[str, str]] = []
        for i, row in enumerate(choices):
            display_name = ""
            if i < len(extracted) and (extracted[i] or "").strip():
                display_name = extracted[i].strip()
            if not display_name:
                display_name = label_for_rank_row(row, index_1based=None)
            target = (row.get("scheme_name") or "").strip() or display_name
            available.append({"display_name": display_name, "target_scheme_name": target})

        rec_state["scheme_titles_from_answer"] = extracted

        if use_selected_scope and previous_available:
            # Keep original discovery list so user can switch to another number any time.
            rec_state["available_schemes"] = previous_available
            rec_state["selected_index"] = previous_selected_index if previous_selected_index >= 0 else 0
        else:
            rec_state["available_schemes"] = available
            rec_state["selected_index"] = 0 if available else -1

        if use_selected_scope and previous_available:
            # Ensure selection points to the currently locked scheme in the preserved list.
            chosen = selected.strip().lower()
            for i, row in enumerate(rec_state["available_schemes"]):
                tn = (row.get("target_scheme_name") or "").strip().lower()
                dn = (row.get("display_name") or "").strip().lower()
                if tn == chosen or dn == chosen:
                    rec_state["selected_index"] = i
                    break
        else:
            rec_state["selected_index"] = 0 if available else -1
        assistant_text = rag_result.get("answer") or "I could not generate an answer."
        next_q = next_profile_question(profile)
        if next_q:
            assistant_text = f"{assistant_text}\n\n{next_q}"

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": assistant_text})

    return {
        "profile": profile,
        "chat_history": chat_history,
        "recommendations": rec_state,
        "rag_result": rag_result or {},
        "recommendation_lines": [],
        "action": action,
    }

