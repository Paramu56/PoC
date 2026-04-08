"""
Backup of currently-working PDF generator.
Saved before formatting tweaks so we can roll back quickly if needed.
"""

from __future__ import annotations

import re
import textwrap
from io import BytesIO
from typing import List, Optional

from fpdf import FPDF


def _latin1_safe(text: str) -> str:
    if not text:
        return ""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _wrap_unbroken(s: str, width: int = 44) -> str:
    """Insert breaks so fpdf multi_cell can wrap long digit strings (phone) within margins."""
    s = (s or "").strip()
    if not s:
        return s
    if " " in s or len(s) <= width:
        return s
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=True, replace_whitespace=False))


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
    pdf = _KitPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    text_w = getattr(pdf, "epw", None) or (pdf.w - pdf.l_margin - pdf.r_margin)

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
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(text_w, 6, _latin1_safe(f"Name: {applicant_name or '(not provided)'}"))
    phone_line = applicant_phone or "(not provided)"
    if phone_line != "(not provided)":
        phone_line = _wrap_unbroken(phone_line)
    pdf.multi_cell(text_w, 6, _latin1_safe("Phone: " + phone_line))
    addr = address_block or "(not provided)"
    if addr != "(not provided)":
        addr = _wrap_unbroken(addr, width=72)
    pdf.multi_cell(text_w, 6, _latin1_safe(f"Address: {addr}"))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Details for this scheme", ln=True)
    pdf.set_font("Helvetica", "", 10)
    details = strip_markdown_like(scheme_details_plain)[:12000]
    if details.strip():
        pdf.multi_cell(text_w, 5, _latin1_safe(details))
    else:
        pdf.multi_cell(text_w, 6, "(No scheme text available from the local index for this selection.)")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Documents typically required", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if documents_required:
        for i, d in enumerate(documents_required, start=1):
            pdf.multi_cell(text_w, 6, _latin1_safe(f"{i}. {d}"))
    else:
        pdf.multi_cell(
            text_w,
            6,
            "No document list is linked in the knowledge graph for this scheme. Check official notifications.",
        )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Bangalore One centre (suggested)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(text_w, 6, _latin1_safe(f"Centre name: {suggested_centre_name}"))
    pdf.multi_cell(text_w, 6, _latin1_safe(f"Address: {suggested_centre_address}"))
    if suggested_centre_km is not None:
        pdf.multi_cell(
            text_w,
            6,
            f"Approx. straight-line distance from your address (geocoded): {suggested_centre_km:.1f} km",
        )

    bio = BytesIO()
    pdf.output(bio)
    return bio.getvalue()
