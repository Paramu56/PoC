"""
Build a plain, citizen-facing PDF (ASCII-friendly) using fpdf2.
No markdown symbols; only applicant, chosen scheme details, documents, and Bangalore One address.
"""

from __future__ import annotations

import re
import textwrap
from io import BytesIO
from typing import List, Optional

from fpdf import FPDF
from fpdf.enums import WrapMode, XPos, YPos


def _normalize_for_pdf(text: str) -> str:
    """
    Replace common Unicode glyphs with ASCII-friendly equivalents before latin-1 encoding.
    This avoids '?' artifacts in generated PDFs.
    """
    if not text:
        return ""
    t = str(text)
    replacements = {
        "\u20b9": "Rs ",  # rupee sign
        "\u2022": "- ",   # bullet
        "\u25cf": "- ",
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\xa0": " ",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    return t


def _latin1_safe(text: str) -> str:
    if not text:
        return ""
    return _normalize_for_pdf(text).encode("latin-1", errors="replace").decode("latin-1")


def _wrap_long_tokens(s: str, max_chars: int = 52) -> str:
    """Break very long tokens so lines do not overflow the right margin."""
    s = (s or "").strip()
    if not s:
        return s
    out: List[str] = []
    for part in re.split(r"(\s+)", s):
        if not part:
            continue
        if part.isspace():
            out.append(part)
            continue
        if len(part) <= max_chars:
            out.append(part)
            continue
        out.append("\n".join(textwrap.wrap(part, width=max_chars, break_long_words=True, replace_whitespace=False)))
    return "".join(out)


def _reflow_extracted_text(text: str) -> str:
    """
    Some extracted PDF text comes one-word-per-line.
    Collapse single newlines into spaces, keep paragraph gaps.
    """
    if not text:
        return ""
    t = _normalize_for_pdf(text).replace("\r\n", "\n").replace("\r", "\n")
    # Preserve page boundaries as paragraph breaks.
    t = re.sub(r"\s+(Page\s+\d+\s*:)", r"\n\n\1", t, flags=re.IGNORECASE)
    blocks = re.split(r"\n\s*\n+", t)
    merged: List[str] = []
    for block in blocks:
        line = re.sub(r"[\n]+", " ", block)
        line = re.sub(r" +", " ", line).strip()
        # Remove repeated markdown-like separators left from extraction.
        line = re.sub(r"\s*[|]+\s*", " ", line)
        if line:
            merged.append(line)
    return "\n\n".join(merged)


def _paragraphs(text: str) -> List[str]:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    return parts


def _write_kv(pdf: FPDF, *, width: float, label: str, value: str) -> None:
    """
    Render key-value blocks with label on one line and value below it.
    This avoids long `Label: <very long token>` strings overflowing right edge.
    """
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(
        width,
        5.5,
        _latin1_safe(label),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 10)
    # Extra hard-wrap protects against overflow from long address tokens.
    v = _wrap_long_tokens(value or "", max_chars=34)
    pdf.multi_cell(
        width,
        5.5,
        _latin1_safe(v),
        wrapmode=WrapMode.CHAR,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(0.8)


def strip_markdown_like(text: str) -> str:
    """Remove common markdown / markup so printed PDF reads as plain text."""
    if not text:
        return ""
    lines = text.splitlines()
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^#+\s*", "", s)
        s = re.sub(r"^[-*+]\s+", "", s)
        s = re.sub(r"^\d+\.\s+", "", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
        s = re.sub(r"\*([^*]+)\*", r"\1", s)
        s = re.sub(r"__([^_]+)__", r"\1", s)
        s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
        s = re.sub(r"`([^`]*)`", r"\1", s)
        s = re.sub(r"```\w*", "", s)
        if s:
            out.append(s)
    return "\n".join(out)


class _KitPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "PoC demo - verify details with official Karnataka sources before applying.", align="C")


def build_scheme_kit_pdf(
    *,
    scheme_name: str,
    applicant_name: str,
    applicant_phone: str,
    address_block: str,
    suggested_centre_name: str,
    suggested_centre_address: str,
    suggested_centre_km: Optional[float],
    scheme_details_plain: str,
    documents_required: List[str],
    scheme_title_display: str = "",
) -> bytes:
    """
    Single-scheme PDF only: applicant name, scheme name, plain details, documents, Bangalore One address.
    """
    pdf = _KitPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    text_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Karnataka scheme - application kit (PoC)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        text_w,
        5,
        "Demonstration only. Confirm scheme rules and forms on official government portals.",
    )
    pdf.ln(3)

    title = (scheme_title_display or scheme_name).strip() or scheme_name
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(text_w, 7, _latin1_safe(f"Scheme: {title}"))
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Applicant", ln=True)
    _write_kv(pdf, width=text_w, label="Name", value=(applicant_name or "(not provided)"))
    phone_line = applicant_phone or "(not provided)"
    if phone_line != "(not provided)":
        phone_line = _wrap_long_tokens(phone_line, max_chars=34)
    _write_kv(pdf, width=text_w, label="Phone", value=phone_line)
    addr = address_block or "(not provided)"
    if addr != "(not provided)":
        addr = _wrap_long_tokens(addr, max_chars=42)
    _write_kv(pdf, width=text_w, label="Address", value=addr)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Details for this scheme", ln=True)
    pdf.set_font("Helvetica", "", 10)
    details = _reflow_extracted_text(strip_markdown_like(scheme_details_plain))[:9000]
    if details.strip():
        for para in _paragraphs(details):
            pdf.multi_cell(text_w, 5.5, _latin1_safe(para), wrapmode=WrapMode.WORD)
            pdf.ln(1.2)
    else:
        pdf.multi_cell(text_w, 6, "(No scheme text available from the local index for this selection.)")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Documents typically required", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if documents_required:
        for i, d in enumerate(documents_required, start=1):
            item = _reflow_extracted_text(d)
            pdf.multi_cell(text_w, 6, _latin1_safe(f"{i}. {item}"), wrapmode=WrapMode.WORD)
    else:
        pdf.multi_cell(
            text_w,
            6,
            "No document list is linked in the knowledge graph for this scheme. Check official notifications.",
        )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Bangalore One centre (suggested)", ln=True)
    centre_name = _reflow_extracted_text(suggested_centre_name) or "(not provided)"
    _write_kv(pdf, width=text_w, label="Centre name", value=centre_name)
    centre_addr = _wrap_long_tokens(_reflow_extracted_text(suggested_centre_address), max_chars=34)
    _write_kv(pdf, width=text_w, label="Address", value=centre_addr or "(not provided)")
    if suggested_centre_km is not None:
        pdf.multi_cell(
            text_w,
            6,
            f"Approx. straight-line distance from your address (geocoded): {suggested_centre_km:.1f} km",
        )

    bio = BytesIO()
    pdf.output(bio)
    return bio.getvalue()
