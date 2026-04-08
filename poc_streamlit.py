"""
PoC UI: Gemini-orchestrated RAG (retrieve → refinement → graph → final answer), ranked scheme picker,
PDF kit, optional Bangalore One suggestion. Profile fields match Flask `templates/index.html`.

Run from this folder (use python -m if `streamlit` is not on PATH):
  pip install -r requirements-poc.txt
  set GEMINI_API_KEY=...
  python -m streamlit run poc_streamlit.py

Or: run_poc_streamlit.bat

Requires: ingested Chroma DB (ingest_karnataka_schemes.py) and GEMINI_API_KEY.

Note: `.streamlit/config.toml` disables the file watcher so Streamlit does not traverse
`transformers` in site-packages (avoids torchvision import noise when using SentenceTransformers).
"""

from __future__ import annotations

import os
import re
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, List

import streamlit as st
from google.genai.errors import ClientError, ServerError

_CONTENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CONTENT_DIR not in sys.path:
    sys.path.insert(0, _CONTENT_DIR)

from graph_scheme_documents import load_scheme_documents_from_cypher
from ingest_karnataka_schemes import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL
from orchestrated_rag_schemes import (
    DEFAULT_GEMINI_MODEL,
    _build_retriever,
    get_documents_for_scheme_only,
    run_orchestrated_rag_result,
)
from scheme_choice_label import label_for_rank_row, scheme_heading_title
from poc_nearby import build_geocode_query, load_centres, nearest_centre, nominatim_geocode
from poc_pdf import build_scheme_kit_pdf, strip_markdown_like
from user_profile import compose_rag_question


def _resolve_pdf_scheme_target(
    selected_row: Dict[str, Any],
    extracted_names: List[str],
    choice_idx: int,
    ranked_rows: List[Dict[str, Any]],
) -> str:
    """
    Resolve the actual scheme_name to fetch PDF text for.
    Prefer the selected extracted label, then fuzzy-match it to known scheme_name values.
    """
    candidates: List[str] = []
    for r in ranked_rows or []:
        n = (r.get("scheme_name") or "").strip()
        if n and n not in candidates:
            candidates.append(n)
    chosen = (selected_row.get("scheme_name") or "").strip()
    shown = ""
    if choice_idx < len(extracted_names):
        shown = (extracted_names[choice_idx] or "").strip()
    if not shown:
        return chosen

    # Exact/containment before fuzzy.
    shown_l = shown.lower()
    for c in candidates:
        cl = c.lower()
        if shown_l == cl or shown_l in cl or cl in shown_l:
            return c
    best = chosen
    best_sc = 0.0
    for c in candidates:
        sc = SequenceMatcher(None, shown_l, c.lower()).ratio()
        if sc > best_sc:
            best_sc = sc
            best = c
    return best if best_sc >= 0.45 else chosen


def _profile_form() -> Dict[str, Any]:
    """Collect the same keys as Flask Scheme Finder (`app.py` / `index.html`), except Name (collected above)."""
    st.markdown("**Optional profile** — helps eligibility-aware answers (same choices as the Flask UI).")
    r1 = st.columns(3)
    with r1[0]:
        gender = st.selectbox("Gender", ("", "Male", "Female", "Transgender"), key="pf_gender")
    with r1[1]:
        age = st.text_input("Age", key="pf_age", placeholder="Optional")
    with r1[2]:
        caste = st.selectbox(
            "Caste",
            ("", "General", "SC", "ST", "OBC", "PVTG", "DNT"),
            key="pf_caste",
        )

    r2 = st.columns(3)
    with r2[0]:
        residence = st.selectbox("Residence", ("", "Rural", "Urban"), key="pf_residence")
    with r2[1]:
        marital_status = st.selectbox(
            "Marital status",
            ("", "Married", "Never married", "Widowed", "Divorced", "Separated"),
            key="pf_marital",
        )
    with r2[2]:
        disability_percentage = st.text_input("Disability %", key="pf_disability", placeholder="0–100, optional")

    r3 = st.columns(3)
    with r3[0]:
        employment_status = st.selectbox(
            "Employment status",
            ("", "Unemployed", "Employed", "Self-Employed/Entrepreneur"),
            key="pf_employment",
        )
    with r3[1]:
        occupation = st.selectbox(
            "Occupation",
            (
                "",
                "Artisans/Spinners/Weavers",
                "Artists",
                "Construction Worker",
                "Ex-Serviceman",
                "Farmer",
                "Fisherman",
                "Health Worker",
                "Khadi Artisan",
                "Safai Karamchari",
                "Sportsperson",
                "Street vendor",
                "Student",
                "Teacher/Faculty",
                "Unorganized worker",
                "Government employee",
            ),
            key="pf_occupation",
        )
    with r3[2]:
        minority = st.selectbox("Minority", ("", "Yes", "No"), key="pf_minority")

    r4 = st.columns(3)
    with r4[0]:
        below_poverty_line = st.selectbox("Below poverty line", ("", "Yes", "No"), key="pf_bpl")
    with r4[1]:
        economic_distress = st.selectbox("Economic distress", ("", "Yes", "No"), key="pf_distress")

    return {
        "name": "",
        "gender": (gender or "").strip(),
        "age": (age or "").strip(),
        "caste": (caste or "").strip(),
        "residence": (residence or "").strip(),
        "marital_status": (marital_status or "").strip(),
        "disability_percentage": (disability_percentage or "").strip(),
        "employment_status": (employment_status or "").strip(),
        "occupation": (occupation or "").strip(),
        "minority": (minority or "").strip(),
        "below_poverty_line": (below_poverty_line or "").strip(),
        "economic_distress": (economic_distress or "").strip(),
    }


def main() -> None:
    st.set_page_config(page_title="Karnataka schemes PoC", layout="wide")
    st.title("Karnataka schemes — PoC assistant")
    st.caption(
        "Orchestrated RAG with Gemini (refinement + final answer), Chroma retrieval ranking, "
        "PDF kit, sample Bangalore One geodesic hint. Demo only — confirm with official sources."
    )

    with st.sidebar:
        st.subheader("Index settings")
        db_path = st.text_input("Chroma path", value=DEFAULT_DB_PATH)
        collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)
        model_name = st.text_input("Embedding model", value=DEFAULT_MODEL)
        st.subheader("Gemini")
        st.caption(
            "Default **one** `generate_content` per run (refinement off). Spacing **6s** between calls "
            "(**GEMINI_CALL_SPACING_S**). Context per chunk capped (**GEMINI_CONTEXT_CHUNK_CHARS**, default 2500) "
            "to reduce 429s. Fallback runs only for wrong-model errors, not 429."
        )
        llm_model = st.text_input("Gemini model", value=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
        fallback_llm = st.text_input("Fallback Gemini model", value="gemini-pro-latest")
        use_refinement_llm = st.checkbox(
            "Refinement LLM (adds 1 extra Gemini call)",
            value=False,
            help="Optional follow-up retrieval step. Leave off if you see 429 RESOURCE_EXHAUSTED.",
        )
        with st.expander("Retrieval sizes"):
            initial_k = st.number_input("Initial k", min_value=2, max_value=30, value=6)
            refined_k = st.number_input("Refined k per query", min_value=2, max_value=20, value=4)
            ranked_n = st.number_input("Ranked list (dedup schemes)", min_value=8, max_value=48, value=24)

    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Set **GEMINI_API_KEY** in your environment before running this app. Orchestration requires Gemini.")
        st.stop()

    situation = st.text_area(
        "Your question",
        placeholder="Example: I am an artisan. What help can I get in Karnataka?",
        height=100,
        key="situation_q",
    )

    user_name = st.text_input(
        "Your name",
        key="user_name",
        help="Used on the generated PDF and sent as part of your profile to the assistant.",
    )

    with st.expander("Optional profile (same choices as Flask Scheme Finder)", expanded=False):
        profile = _profile_form()
    profile["name"] = (user_name or "").strip()

    st.subheader("Contact & location (PDF cover sheet & nearest sample centre)")
    c1, c2, c3 = st.columns(3)
    with c1:
        applicant_phone = st.text_input("Phone (optional)", key="pdf_phone")
    with c2:
        pincode = st.text_input("PIN code", placeholder="560001", key="pdf_pin")
    with c3:
        city = st.text_input("City", value="Bengaluru", key="pdf_city")

    address_line = st.text_input("Address line", placeholder="Street / area (e.g. MG Road)", key="pdf_addr")

    if st.button("Run orchestration & find schemes", type="primary"):
        st.session_state.pop("orch", None)
        st.session_state.pop("pdf_meta", None)
        if not situation.strip():
            st.warning("Please enter a question.")
            return
        loc = build_geocode_query(line1=address_line, pincode=pincode, city=city)
        composed = compose_rag_question(situation.strip(), profile, location_line=loc)
        try:
            result = run_orchestrated_rag_result(
                db_path=db_path.strip(),
                collection_name=collection_name.strip(),
                model_name=model_name.strip(),
                llm_model=llm_model.strip(),
                question=composed,
                initial_k=int(initial_k),
                refined_k=int(refined_k),
                fallback_llm_model=fallback_llm.strip() or None,
                ranked_n=int(ranked_n),
                use_refinement_llm=use_refinement_llm,
            )
        except RuntimeError as e:
            st.error(str(e))
            return
        except (ClientError, ServerError) as e:
            st.error(f"Gemini API error: {e}")
            return
        except Exception as e:
            st.error(f"Orchestration failed: {e}")
            return

        st.session_state["orch"] = {
            "result": result,
            "composed_question": composed,
            "_collection_cfg": {
                "db_path": db_path.strip(),
                "collection_name": collection_name.strip(),
                "model_name": model_name.strip(),
            },
        }

    orch = st.session_state.get("orch")
    if orch:
        result: Dict[str, Any] = orch["result"]
        answer = result.get("answer") or ""
        refine = result.get("refine") or {}
        debug = result.get("debug") or {}
        citations = result.get("citations") or []
        extracted_scheme_names: List[str] = result.get("answer_extracted_scheme_names") or []
        # PDF picker: schemes from orchestration context only (same evidence as Gemini), not broad retrieval
        choices: List[Dict[str, Any]] = result.get("pdf_scheme_choices") or []
        if not choices:
            choices = result.get("ranked_schemes") or []

        st.subheader("Orchestrated answer (Gemini)")
        st.markdown(answer)

        with st.expander("Refinement step (Gemini JSON)"):
            st.json(
                {
                    "sufficient": refine.get("sufficient"),
                    "reason": refine.get("reason"),
                    "refined_queries": refine.get("refined_queries"),
                }
            )
        with st.expander("Debug & citations"):
            st.json(
                {
                    "debug": debug,
                    "citations": citations,
                    "answer_extracted_scheme_names": result.get("answer_extracted_scheme_names"),
                    "pdf_choices_after_filter": len(result.get("pdf_scheme_choices") or []),
                    "pdf_choices_context_only": len(result.get("pdf_scheme_choices_context_only") or []),
                    "broad_ranked_count": len(result.get("ranked_schemes") or []),
                }
            )

        if not choices:
            st.warning("No schemes available for PDF. Check Chroma ingest.")
        else:
            st.subheader("Choose scheme for PDF kit")
            st.caption("Each option uses the same scheme name as the extraction step.")

            def _pdf_picker_label(i: int) -> str:
                if i < len(extracted_scheme_names) and (extracted_scheme_names[i] or "").strip():
                    name = extracted_scheme_names[i].strip()
                    pg = choices[i].get("page")
                    suffix = f" — evidence PDF p. {pg}" if pg is not None else ""
                    return f"{i + 1}. {name}{suffix}"
                return label_for_rank_row(choices[i], index_1based=i + 1)

            choice_idx = st.radio(
                "Scheme",
                list(range(len(choices))),
                format_func=_pdf_picker_label,
                label_visibility="collapsed",
            )
            selected = choices[choice_idx]

            if st.button("Generate PDF kit"):
                cfg = orch.get("_collection_cfg") or {}
                try:
                    col = _build_retriever(
                        db_path=cfg.get("db_path", DEFAULT_DB_PATH),
                        collection_name=cfg.get("collection_name", DEFAULT_COLLECTION),
                        model_name=cfg.get("model_name", DEFAULT_MODEL),
                    )
                except Exception as e:
                    st.error(f"Collection error: {e}")
                    return

                scheme_name = _resolve_pdf_scheme_target(
                    selected_row=selected,
                    extracted_names=extracted_scheme_names,
                    choice_idx=choice_idx,
                    ranked_rows=(result.get("ranked_schemes") or []),
                )
                docs_map = load_scheme_documents_from_cypher()
                documents = docs_map.get(scheme_name, [])

                docs_r, metas_r = get_documents_for_scheme_only(col, scheme_name)
                first_snip = docs_r[0] if docs_r else ""
                if choice_idx < len(extracted_scheme_names) and (extracted_scheme_names[choice_idx] or "").strip():
                    title_disp = extracted_scheme_names[choice_idx].strip()
                else:
                    title_disp = scheme_heading_title(scheme_name, first_snip)
                # Keep PDF details focused on the selected scheme only and avoid noisy page headings.
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
                    if len(picked_details) >= 8:
                        break
                if not picked_details:
                    fallback_text = (selected.get("text_for_label") or selected.get("snippet") or "").strip()
                    if fallback_text:
                        picked_details = [strip_markdown_like(fallback_text)]
                scheme_details_plain = "\n\n".join(picked_details)
                addr_block = build_geocode_query(
                    line1=address_line,
                    pincode=pincode,
                    city=city,
                )
                centres = load_centres()
                centre_name = "See sevasindhu.karnataka.gov.in / Bangalore One for official centres"
                centre_address = "Add data/bangalore_one_centres.csv or check official listings."
                km_hint = None
                if centres:
                    q = build_geocode_query(line1=address_line, pincode=pincode, city=city)
                    coords = nominatim_geocode(q) if q.strip() else None
                    if coords is None and pincode.strip():
                        coords = nominatim_geocode(f"{pincode.strip()}, {city}, Karnataka, India")
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
                try:
                    pdf_bytes = build_scheme_kit_pdf(
                        scheme_name=scheme_name,
                        applicant_name=profile.get("name") or "",
                        applicant_phone=applicant_phone,
                        address_block=addr_block,
                        suggested_centre_name=centre_name,
                        suggested_centre_address=centre_address,
                        suggested_centre_km=km_hint,
                        scheme_details_plain=scheme_details_plain,
                        documents_required=documents,
                        scheme_title_display=title_disp,
                    )
                except Exception as e:
                    st.error(f"PDF build failed: {e}")
                    return

                st.session_state["pdf_meta"] = {"scheme": scheme_name, "bytes": pdf_bytes}
                st.success("PDF ready - download below.")

            meta = st.session_state.get("pdf_meta")
            if meta and meta.get("bytes"):
                pdf_bytes = meta["bytes"]
                pdf_scheme = meta.get("scheme") or "scheme"
                if pdf_scheme != choices[choice_idx]["scheme_name"]:
                    st.info(
                        "Tip: PDF below is for a different scheme than your current radio selection — generate again to refresh."
                    )
                safe = "".join(c for c in pdf_scheme if c.isalnum() or c in " -_")[:60].strip() or "scheme"
                st.download_button(
                    "Download PDF kit",
                    data=pdf_bytes,
                    file_name=f"karnataka_scheme_kit_{safe.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                )


if __name__ == "__main__":
    main()
