"""
Human-readable one-line labels for scheme rows. Handles fallback metadata
'Karnataka Schemes (page N)' by scanning chunk text for a real scheme title,
not eligibility bullets like '1. Age: 18–45'.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_FALLBACK_RE = re.compile(r"^Karnataka Schemes \(page (\d+)\)\s*$", re.IGNORECASE)


def is_fallback_scheme_metadata_name(name: str) -> bool:
    """True when Chroma metadata used the PDF page fallback label instead of a real scheme name."""
    return bool(name and _FALLBACK_RE.match(name.strip()))

# Lines that look like eligibility / process, not scheme titles
_ELIGIBILITY_START = re.compile(
    r"^\d+\.\s*(Age|Educational|Income|Annual|Must|Should|The\s+applicant|Family|"
    r"Below|Above|Resident|Domicile|BPL|Aadhaar|Bank|Caste|Certificate|Documents?)\b",
    re.IGNORECASE,
)
_BULLET_LINE = re.compile(r"^[•\-\*●○◦]\s*")
_FULL_PROCESS = re.compile(r"full\s+(application\s+)?process", re.I)
_APPLY_ONLINE = re.compile(r"apply\s+online", re.I)

# Strong signals for a Karnataka scheme *name*
_SCHEME_WORD = re.compile(
    r"(Yojana|Scheme|Nidhi|Pathakam|Pension|Subsidy|Scholarship|Allowance|"
    r"Suraksha|Griha|Lakshmi|Bhagya|Bandhu|Awards?|Stipend|Loan|Empowerment)",
    re.I,
)
_NUMBERED_SCHEME = re.compile(r"^\d+\.\s+.{8,120}$")  # "16. Dr. B.R. Ambedkar Nivasa..."


def _score_line_for_title(line: str) -> int:
    s = line.strip()
    if not s or len(s) < 6:
        return -100
    if s.lower().startswith("category:"):
        return -100
    if _BULLET_LINE.match(s):
        return -20
    if _ELIGIBILITY_START.match(s):
        return -50
    if _FULL_PROCESS.search(s) or (_APPLY_ONLINE.search(s) and len(s) > 80):
        return -30
    if len(s) > 160:
        return -10

    score = 0
    if _SCHEME_WORD.search(s):
        score += 25
    if _NUMBERED_SCHEME.match(s):
        score += 20
    # Prefer moderate length title-like lines
    if 15 <= len(s) <= 95:
        score += 5
    if s[0].isdigit() and "." in s[:4] and _SCHEME_WORD.search(s):
        score += 15
    return score


def _best_title_lines(lines: List[str]) -> List[Tuple[int, str]]:
    scored: List[Tuple[int, str]] = []
    for ln in lines:
        t = ln.strip()
        if not t:
            continue
        sc = _score_line_for_title(t)
        if sc > -50:
            scored.append((sc, t))
    scored.sort(key=lambda x: -x[0])
    return scored


def extract_scheme_title_line(text: str, *, max_len: int = 78) -> str:
    """
    Pick one line that looks like a scheme name/title, not eligibility text.
    """
    if not (text or "").strip():
        return ""

    lines = text.splitlines()
    scored = _best_title_lines(lines)
    if scored and scored[0][0] >= 10:
        pick = scored[0][1]
    elif scored and scored[0][0] >= 0:
        pick = scored[0][1]
    else:
        # Last resort: first non-junk line
        pick = ""
        for ln in lines:
            t = ln.strip()
            if not t or t.lower().startswith("category:"):
                continue
            if _ELIGIBILITY_START.match(t) or _BULLET_LINE.match(t):
                continue
            pick = t
            break
        if not pick and lines:
            pick = lines[0].strip()

    pick = re.sub(r"\s+", " ", pick).strip()
    if len(pick) > max_len:
        pick = pick[: max_len - 1].rstrip() + "…"
    return pick


def label_for_rank_row(row: Dict[str, Any], *, index_1based: Optional[int] = None) -> str:
    """
    Single compact line: optional '#. ' added by caller, or pass index_1based for '1. ' prefix.
    Format: '1. Scheme Name Here (page 8)' when index given, else 'Scheme Name Here (page 8)'.
    """
    name = str(row.get("scheme_name") or "").strip()
    # Prefer longer sample for title mining (see ranked_unique_schemes_by_retrieval)
    blob = str(row.get("text_for_label") or row.get("snippet") or "")
    page = row.get("page")

    m = _FALLBACK_RE.match(name)
    if m:
        pg = m.group(1)
        title = extract_scheme_title_line(blob)
        if not title:
            title = f"Page {pg} chunk"
        core = f"{title} (page {pg})"
    else:
        core = name
        if page is not None and str(page) not in core:
            core = f"{core} (page {page})"

    if len(core) > 100:
        core = core[:97] + "…"

    if index_1based is not None:
        return f"{index_1based}. {core}"
    return core


def scheme_heading_title(scheme_name: str, snippet: str = "") -> str:
    """Title line for PDF when metadata is generic."""
    name = (scheme_name or "").strip()
    m = _FALLBACK_RE.match(name)
    if m:
        pg = m.group(1)
        title = extract_scheme_title_line(snippet or "")
        if title:
            return f"{title} (PDF page {pg})"
        return f"Scheme text from PDF page {pg}"
    return name
