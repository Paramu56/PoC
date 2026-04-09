"""
Extract exactly one scheme's text from the original Karnataka Schemes PDF.

The published PDF uses numbered scheme headings like:
  "3.  Gruha  Jyothi  Yojana  (Free  Electricity)"
  "23.  Shrama  Shakthi  Loan  Scheme  (Artisan  Support)"
not the ingest heuristic (title + next line "Category:"). This module splits on those
headings and returns the single section that best matches the chosen scheme name.
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from pypdf import PdfReader

from ingest_karnataka_schemes import DEFAULT_PDF_IN_DATA, _clean_page_text

# Lines like "2.  Currently, ..." or "2. Installments:" are sub-steps, not scheme titles.
# Numbered bullets inside a scheme body (1. Age:, 3. Loan:, etc.) must not be treated as a new scheme.
# Real headings look like "47. Chief Minister's ... (CMEGP)" with a title word first (Chief, Gruha, Shakti, ...).
_SUBLIST_FIRST_WORDS = frozenset(
    {
        "loan",
        "age",
        "subsidy",
        "project",
        "apply",
        "documents",
        "educational",
        "resident",
        "existing",
        "individuals",
        "prepare",
        "balance",
        "full",  # "Full Benefits:" inline
        "required",
        "exclusions",
        "linkages",
        "details",
        "portal",
        "selection",
        "post-selection",
        "note",
        "important",
        "steps",
        "procedure",
        "benefits",
        "eligibility",
        "category",
        "categories",
        "duration",
        "amount",
        "interest",
        "repayment",
        "installments",
        "annual",
        "renewal",
        "ensure",
        "applicants",
        "mass",
        "selection",
    }
)

_BAD_HEADER_FIRST = frozenset(
    {
        "installments",
        "additional",
        "annual",
        "beneficiaries",
        "post-selection",
        "distance",
        "must",
        "health",
        "refills",
        "submit",
        "selection",
        "obc",
        "verification",
        "family",
        "interest",
        "the",
        "also",
        "currently",
        "visit",
        "apply",
        "check",
        "for",
        "no",
        "select",
        "register",
        "log",
        "to",
        "if",
        "continuing",
        "expert",
        "note",
        "full",
        "post",
    }
)


def _norm_key(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _extract_pages_quiet(pdf_path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = _clean_page_text(raw)
        if cleaned:
            out.append((i + 1, cleaned))
    return out


def _is_scheme_header_line(line: str) -> bool:
    s = line.strip()
    m = re.match(r"^\s*(\d{1,3})\.\s+(.+)$", s)
    if not m:
        return False
    rest = m.group(2).strip()
    if len(rest) < 12:
        return False
    # Body bullets like "● Details:" or "- Full Benefits:"
    rest = re.sub(r"^[●•·▪▫\-\*]\s*", "", rest)
    parts = rest.split()
    if not parts:
        return False
    fw = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", parts[0].lower())
    if len(fw) < 2:
        return False
    if fw in _BAD_HEADER_FIRST:
        return False
    # "3. Loan:", "1. Age:", "1. Project Cost:" — subsection lines, not "47. Scheme Title".
    if fw in _SUBLIST_FIRST_WORDS:
        return False
    # One line containing multiple "1. ... 2. ..." enumerators is body text, not a heading row.
    if len(line) > 100 and len(re.findall(r"\b\d{1,2}\.\s+[A-Za-z]", line)) >= 2:
        return False
    return True


def _split_into_numbered_sections(full_text: str) -> List[Tuple[str, str]]:
    """
    Split full PDF text into (header_line, body_text) sections.
    Body includes everything after the header line until the next scheme header line.
    """
    lines = full_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: List[Tuple[str, str]] = []
    current_header: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal current_header, buf
        if current_header is None:
            return
        body = "\n".join(buf).strip()
        sections.append((current_header, body))
        buf = []

    for line in lines:
        if _is_scheme_header_line(line):
            flush()
            current_header = line.strip()
        else:
            if current_header is not None:
                buf.append(line)
    flush()
    return sections


def _title_from_header(header_line: str) -> str:
    return re.sub(r"^\s*\d{1,3}\.\s+", "", header_line).strip()


def _score_section(user: str, header_line: str, body: str) -> float:
    """
    Title-first: match the numbered PDF heading, not incidental words in the body.
    (A long query string can appear in another section's body via cross-references — do not treat that as a 0.95 hit.)
    """
    u = _norm_key(user)
    title = _norm_key(_title_from_header(header_line))
    body_n = _norm_key(body[:8000])

    if not u:
        return 0.0

    # Strong: query aligns with this section's title line only.
    if len(u) >= 8 and u in title:
        return 0.95
    if len(u) >= 8 and len(title) >= 12 and title in u:
        return 0.93

    base = SequenceMatcher(None, u, title).ratio()
    u_words = set(re.findall(r"[a-z0-9]{3,}", u))
    t_words = set(re.findall(r"[a-z0-9]{3,}", title))
    if u_words:
        overlap_ratio = len(u_words & t_words) / len(u_words)
    else:
        overlap_ratio = 0.0
    title_score = max(base, overlap_ratio * 0.85)

    # Body boost only as a tie-breaker when the title already shares tokens with the query
    # (e.g. "KMDC …" in query and in body for the same scheme), not when only the body mentions "kmdc".
    overlap_tokens = u_words & t_words
    body_boost = 0.0
    if overlap_tokens and title_score >= 0.22:
        for w in u_words:
            if len(w) >= 4 and w in body_n:
                body_boost = max(body_boost, 0.08)

    return title_score + body_boost


def extract_scheme_text_from_source_pdf(
    scheme_name: str,
    *,
    pdf_path: Optional[str] = None,
    alternate_names: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Return (text, note). Text is only the chosen scheme section from the source PDF.
    If empty, note explains; caller may fall back to Chroma.

    alternate_names: extra strings to score against (e.g. Gemini display label vs Chroma scheme_name).
    The best (section, score) wins across all non-empty queries.
    """
    path = pdf_path or os.environ.get("KARNATAKA_SCHEMES_PDF") or DEFAULT_PDF_IN_DATA
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return "", f"PDF not found: {path}"

    try:
        pages = _extract_pages_quiet(path)
    except Exception as e:
        return "", f"failed to read PDF: {e}"

    if not pages:
        return "", "no text extracted from PDF"

    full_text = "\n".join(t for _, t in pages)
    sections = _split_into_numbered_sections(full_text)
    if not sections:
        return "", "could not split PDF into numbered scheme sections"

    queries: List[str] = []
    for q in [scheme_name, *(alternate_names or [])]:
        q = (q or "").strip()
        if not q:
            continue
        if q.lower() not in {x.lower() for x in queries}:
            queries.append(q)
    if not queries:
        return "", "empty scheme name"

    best_i = -1
    best_score = 0.0
    for i, (hdr, body) in enumerate(sections):
        sc = max(_score_section(q, hdr, body) for q in queries)
        if sc > best_score:
            best_score = sc
            best_i = i

    if best_i < 0 or best_score < 0.28:
        return "", f"no scheme section matched (best score {best_score:.2f})"

    hdr, body = sections[best_i]
    # One clean block: numbered title line + body (no other schemes).
    block = f"{hdr}\n\n{body}".strip()
    note = "" if best_score >= 0.45 else f"low-confidence match (score {best_score:.2f})"
    return block, note
