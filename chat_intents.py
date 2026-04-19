from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentResult:
    intent: str
    payload: str = ""
    scheme_a: str = ""
    scheme_b: str = ""
    index: int = -1


_COMPARE_RE = re.compile(
    r"(?:compare|difference|advantage).{0,40}?(?:scheme\s+)?(.+?)\s+(?:vs|versus)\s+(?:scheme\s+)?(.+)",
    re.IGNORECASE,
)
_SELECT_INDEX_RE = re.compile(r"(?:select|choose)\s+(?:scheme\s+)?(\d+)", re.IGNORECASE)
_PDF_RE = re.compile(r"(?:generate|create|download).{0,20}(?:pdf|kit)", re.IGNORECASE)
_UNLOCK_RE = re.compile(r"\b(?:unlock\s+scheme|clear\s+selection|reset\s+selection)\b", re.IGNORECASE)


def classify_intent(message: str) -> IntentResult:
    text = (message or "").strip()
    if not text:
        return IntentResult(intent="smalltalk")

    m_compare = _COMPARE_RE.search(text)
    if m_compare:
        return IntentResult(
            intent="compare_schemes",
            scheme_a=m_compare.group(1).strip(" .?,"),
            scheme_b=m_compare.group(2).strip(" .?,"),
        )

    m_select = _SELECT_INDEX_RE.search(text)
    if m_select:
        return IntentResult(intent="select_scheme", index=max(0, int(m_select.group(1)) - 1))

    if _PDF_RE.search(text):
        return IntentResult(intent="generate_pdf")

    if _UNLOCK_RE.search(text):
        return IntentResult(intent="unlock_scheme")

    if any(k in text.lower() for k in ("hi", "hello", "hey")) and len(text.split()) <= 3:
        return IntentResult(intent="smalltalk")

    return IntentResult(intent="discover_schemes", payload=text)
