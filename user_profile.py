"""
Shared user profile formatting for Flask and Streamlit (matches templates/index.html fields).
"""

from __future__ import annotations

from typing import Any, Dict

PROFILE_FIELD_ORDER = [
    "name",
    "gender",
    "age",
    "caste",
    "residence",
    "marital_status",
    "disability_percentage",
    "employment_status",
    "occupation",
    "minority",
    "below_poverty_line",
    "economic_distress",
]


def profile_to_text(profile: Dict[str, Any]) -> str:
    rows = []
    for key in PROFILE_FIELD_ORDER:
        raw = profile.get(key)
        value = (str(raw) if raw is not None else "").strip()
        if value:
            rows.append(f"- {key.replace('_', ' ').title()}: {value}")
    return "\n".join(rows)


def compose_rag_question(
    situation: str,
    profile: Dict[str, Any],
    *,
    location_line: str = "",
) -> str:
    """
    Same structure as Flask answer_question: profile + question + constraints,
    optional location line for centre / local context.
    """
    situation = situation.strip()
    profile_text = profile_to_text(profile)
    if profile_text:
        body = (
            f"User profile:\n{profile_text}\n\n"
            f"Question:\n{situation}\n\n"
            "Please consider profile constraints while selecting schemes."
        )
    else:
        body = situation
    loc = (location_line or "").strip()
    if loc:
        body = f"{body}\n\nLocation context for centre suggestion and local schemes (if relevant): {loc}"
    return body
