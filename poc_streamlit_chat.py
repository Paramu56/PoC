"""
Chat-first Karnataka schemes PoC (parallel path).

This file is intentionally separate from poc_streamlit.py so the baseline launcher
continues to behave exactly as before.
"""

from __future__ import annotations

import concurrent.futures
import copy
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List

import streamlit as st
from google.genai.errors import ClientError, ServerError

from chat_orchestrator import handle_turn
from graph_scheme_documents import load_scheme_documents_from_cypher
from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL
from orchestrated_rag_schemes import (
    DEFAULT_GEMINI_MODEL,
    _build_retriever,
    get_documents_for_scheme_only,
)
from poc_nearby import build_geocode_query, load_centres, nearest_centre, nominatim_geocode
from poc_pdf import build_scheme_kit_pdf, strip_markdown_like
from scheme_choice_label import scheme_heading_title
from scheme_source_pdf import extract_scheme_text_from_source_pdf

_CONTENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CONTENT_DIR not in sys.path:
    sys.path.insert(0, _CONTENT_DIR)


def _ensure_state() -> None:
    if "chat_state" not in st.session_state:
        st.session_state["chat_state"] = {
            "profile": {},
            "chat_history": [
                {
                    "role": "assistant",
                    "content": "Tell me your situation and I will suggest relevant Karnataka schemes.",
                }
            ],
            "recommendations": {},
            "rag_result": {},
            "recommendation_lines": [],
            "pdf_meta": {},
        }


def _status_message_chat(elapsed: float) -> str:
    if elapsed < 15:
        return "Working… (retrieval and model can take a few seconds)"
    if elapsed < 30:
        return "Still working… (15s+) — large context and the API can be slow."
    if elapsed < 45:
        return "Still working… (30s+) — hang tight; Chroma or Gemini may be busy."
    if elapsed < 60:
        return "Still working… (45s+) — if this never finishes, check API quota or network."
    return (
        f"Still working… ({int(elapsed)}s elapsed). Safe to wait; avoid refreshing the page."
    )


def _status_message_pdf(elapsed: float) -> str:
    if elapsed < 15:
        return "Building PDF… (gathering scheme text and centre info)"
    if elapsed < 30:
        return "Still building PDF… (15s+) — source PDF or geocoding can take a moment."
    if elapsed < 45:
        return "Still building PDF… (30s+) — large schemes or slow network."
    return f"Still building PDF… ({int(elapsed)}s elapsed)."


def _run_with_status_every_15s(
    fn: Callable[..., Any],
    message_fn: Callable[[float], str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Run a blocking function in a worker thread and refresh st.info text every ~15 seconds.
    Streamlit's st.spinner cannot change its label while the main thread is blocked.
    """
    placeholder = st.empty()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        start = time.monotonic()
        last_bucket = -1
        while not future.done():
            elapsed = time.monotonic() - start
            bucket = int(elapsed // 15.0)
            if bucket != last_bucket:
                last_bucket = bucket
                placeholder.info(message_fn(elapsed))
            time.sleep(0.35)
        try:
            return future.result()
        finally:
            placeholder.empty()


def _build_pdf_from_chat_state(state: Dict[str, Any], cfg: Dict[str, Any], pdf_inputs: Dict[str, str]) -> Dict[str, Any]:
    rec = state.get("recommendations") or {}
    rows = rec.get("available_schemes") or []
    idx = int(rec.get("selected_index", -1))
    if not (0 <= idx < len(rows)):
        raise RuntimeError("No selected scheme. Use 'select scheme 1' first.")

    selected = rows[idx]
    scheme_name = (selected.get("target_scheme_name") or "").strip()
    if not scheme_name:
        raise RuntimeError("Selected scheme could not be resolved.")

    col = _build_retriever(
        db_path=cfg["db_path"],
        collection_name=cfg["collection_name"],
        model_name=cfg["model_name"],
    )
    docs_map = load_scheme_documents_from_cypher()
    documents = docs_map.get(scheme_name, [])

    title_disp = (selected.get("display_name") or "").strip() or scheme_name

    source_body, _ = extract_scheme_text_from_source_pdf(
        scheme_name,
        alternate_names=[title_disp] if title_disp and title_disp != scheme_name else [],
    )
    scheme_details_plain = ""
    if (source_body or "").strip():
        scheme_details_plain = strip_markdown_like(source_body).strip()
    else:
        docs_r, metas_r = get_documents_for_scheme_only(col, scheme_name)
        first_snip = docs_r[0] if docs_r else ""
        if not (selected.get("display_name") or "").strip():
            title_disp = scheme_heading_title(scheme_name, first_snip)
        picked_details: List[str] = []
        seen_snips = set()
        for d, m in zip(docs_r, metas_r):
            meta_name = (m.get("scheme_name") or "").strip().lower()
            if meta_name and meta_name != scheme_name.strip().lower():
                continue
            clean = strip_markdown_like(d or "").strip()
            clean = clean.replace("\r", "\n")
            clean = re.sub(r"\bPage\s+\d+\s*:\s*", "", clean, flags=re.IGNORECASE)
            if not clean:
                continue
            sig = clean[:220]
            if sig in seen_snips:
                continue
            seen_snips.add(sig)
            picked_details.append(clean)
            if len(picked_details) >= 12:
                break
        if not picked_details:
            fallback_text = (selected.get("text_for_label") or selected.get("snippet") or "").strip()
            if fallback_text:
                picked_details = [strip_markdown_like(fallback_text)]
        scheme_details_plain = "\n\n".join(picked_details)

    addr_block = build_geocode_query(
        line1=pdf_inputs.get("address_line", ""),
        pincode=pdf_inputs.get("pincode", ""),
        city=pdf_inputs.get("city", ""),
    )
    centres = load_centres()
    centre_name = "See sevasindhu.karnataka.gov.in / Bangalore One for official centres"
    centre_address = "Add data/bangalore_one_centres.csv or check official listings."
    km_hint = None
    if centres:
        q = build_geocode_query(
            line1=pdf_inputs.get("address_line", ""),
            pincode=pdf_inputs.get("pincode", ""),
            city=pdf_inputs.get("city", ""),
        )
        coords = nominatim_geocode(q) if q.strip() else None
        if coords is None and (pdf_inputs.get("pincode") or "").strip():
            coords = nominatim_geocode(f"{pdf_inputs['pincode'].strip()}, {pdf_inputs.get('city', 'Bengaluru')}, Karnataka, India")
        if coords:
            hit = nearest_centre(coords[0], coords[1], centres)
            if hit:
                c, dkm = hit
                centre_name = c.name
                centre_address = f"{c.address}. Area: {c.area}. {c.notes}"
                km_hint = dkm
        else:
            c0 = centres[0]
            centre_name = c0.name
            centre_address = f"{c0.address}. Area: {c0.area}. (Fallback: could not geocode your address.)"

    pdf_bytes = build_scheme_kit_pdf(
        scheme_name=scheme_name,
        applicant_name=pdf_inputs.get("applicant_name", ""),
        applicant_phone=pdf_inputs.get("applicant_phone", ""),
        address_block=addr_block,
        suggested_centre_name=centre_name,
        suggested_centre_address=centre_address,
        suggested_centre_km=km_hint,
        scheme_details_plain=scheme_details_plain,
        documents_required=documents,
        scheme_title_display=title_disp,
    )
    return {"scheme": scheme_name, "bytes": pdf_bytes}


def main() -> None:
    st.set_page_config(page_title="Karnataka schemes chat PoC", layout="wide")
    st.title("Karnataka schemes - chat PoC")
    st.caption("New parallel chat path. Baseline PoC remains unchanged.")

    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Set GEMINI_API_KEY in your environment before running this app.")
        st.stop()

    with st.sidebar:
        st.subheader("Index settings")
        db_path = st.text_input("Chroma path", value=DEFAULT_DB_PATH)
        collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)
        model_name = st.text_input("Embedding model", value=DEFAULT_MODEL)
        st.subheader("LLM")
        llm_model = st.text_input(
            "Gemini model",
            value=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            help="Free tier limits requests per model per day (e.g. 20). If you see 429, switch model "
            "(separate quota) or wait / enable billing — see https://ai.google.dev/gemini-api/docs/rate-limits",
        )
        with st.expander("Retrieval sizes"):
            initial_k = st.number_input("Initial k", min_value=2, max_value=30, value=6)
            refined_k = st.number_input("Refined k", min_value=2, max_value=20, value=4)
            ranked_n = st.number_input("Ranked list size", min_value=8, max_value=48, value=24)
        st.subheader("PDF details")
        applicant_name = st.text_input("Applicant name", key="chat_pdf_name")
        applicant_phone = st.text_input("Phone (optional)", key="chat_pdf_phone")
        pincode = st.text_input("PIN code", value="560001", key="chat_pdf_pin")
        city = st.text_input("City", value="Bengaluru", key="chat_pdf_city")
        address_line = st.text_input("Address line", placeholder="Street / area", key="chat_pdf_addr")

    _ensure_state()
    if "chat_ui_error" not in st.session_state:
        st.session_state["chat_ui_error"] = None

    cfg = {
        "db_path": db_path.strip(),
        "collection_name": collection_name.strip(),
        "model_name": model_name.strip(),
        "llm_model": llm_model.strip(),
        "initial_k": int(initial_k),
        "refined_k": int(refined_k),
        "ranked_n": int(ranked_n),
    }

    st.subheader("Chat")
    err_msg = st.session_state.get("chat_ui_error")
    if err_msg:
        st.error(err_msg)
    for m in st.session_state["chat_state"]["chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    pdf_meta = st.session_state["chat_state"].get("pdf_meta") or {}
    if pdf_meta.get("bytes"):
        safe = "".join(c for c in (pdf_meta.get("scheme") or "scheme") if c.isalnum() or c in " -_")[:60].strip() or "scheme"
        st.download_button(
            "Download PDF kit",
            data=pdf_meta["bytes"],
            file_name=f"karnataka_scheme_kit_{safe.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

    user_input = st.chat_input("Ask about schemes, compare schemes, select scheme 1, or generate pdf")
    if user_input:
        st.session_state["chat_ui_error"] = None
        try:
            state_snap = copy.deepcopy(st.session_state["chat_state"])
            new_state = _run_with_status_every_15s(
                handle_turn,
                _status_message_chat,
                user_message=user_input,
                session_state=state_snap,
                cfg=cfg,
            )
            action = new_state.get("action") or {}
            if action.get("type") == "generate_pdf":
                pdf_inputs = {
                    "applicant_name": (applicant_name or "").strip(),
                    "applicant_phone": (applicant_phone or "").strip(),
                    "pincode": (pincode or "").strip(),
                    "city": (city or "").strip(),
                    "address_line": (address_line or "").strip(),
                }
                try:
                    new_state["pdf_meta"] = _run_with_status_every_15s(
                        _build_pdf_from_chat_state,
                        _status_message_pdf,
                        new_state,
                        cfg,
                        pdf_inputs,
                    )
                    new_state["chat_history"].append(
                        {"role": "assistant", "content": "PDF is ready. Use the download button below the chat."}
                    )
                except Exception as e:
                    st.session_state["chat_ui_error"] = f"PDF generation failed: {e}"
                    new_state["chat_history"].append({"role": "assistant", "content": f"PDF generation failed: {e}"})
            st.session_state["chat_state"] = new_state
        except (ClientError, ServerError) as e:
            st.session_state["chat_ui_error"] = f"LLM API error: {e}"
        except Exception as e:
            st.session_state["chat_ui_error"] = f"Chat orchestration failed: {e}"
        st.rerun()


if __name__ == "__main__":
    main()
