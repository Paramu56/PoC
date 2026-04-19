from __future__ import annotations

import re
from typing import Any, Dict, List

from user_profile import PROFILE_FIELD_ORDER


def empty_profile() -> Dict[str, str]:
    return {k: "" for k in PROFILE_FIELD_ORDER}


def update_profile_from_message(profile: Dict[str, Any], message: str) -> Dict[str, str]:
    out = {k: str(v or "").strip() for k, v in (profile or {}).items()}
    text = (message or "").strip()
    low = text.lower()

    if "female" in low:
        out["gender"] = "Female"
    elif "male" in low:
        out["gender"] = "Male"
    elif "transgender" in low:
        out["gender"] = "Transgender"

    age_match = re.search(
        r"\b(?:age\s*(?:is|:)?\s*(\d{1,3})|i\s*am\s*(\d{1,3})\s*(?:years?\s*old|yo)?|(\d{1,3})\s*years?\s*old)\b",
        low,
    )
    if age_match and not out.get("age"):
        raw_age = age_match.group(1) or age_match.group(2) or age_match.group(3)
        if raw_age:
            age_val = int(raw_age)
            if 0 < age_val < 120:
                out["age"] = str(age_val)

    for caste in ("sc", "st", "obc", "general", "pvtg", "dnt"):
        if re.search(rf"\b{re.escape(caste)}\b", low):
            out["caste"] = caste.upper() if len(caste) <= 3 else caste.title()

    if "rural" in low:
        out["residence"] = "Rural"
    elif "urban" in low:
        out["residence"] = "Urban"

    if "student" in low and not out.get("occupation"):
        out["occupation"] = "Student"
    if "farmer" in low and not out.get("occupation"):
        out["occupation"] = "Farmer"

    return out


def next_profile_question(profile: Dict[str, Any]) -> str:
    p = {k: str(v or "").strip() for k, v in (profile or {}).items()}
    prompts: List[tuple[str, str]] = [
        ("gender", "If you share your gender, I can refine scheme eligibility."),
        ("age", "If you share your age, I can filter age-specific schemes."),
        ("caste", "If relevant, sharing caste category can improve matching (optional)."),
        ("residence", "Is your residence rural or urban? That can affect eligibility."),
    ]
    for key, question in prompts:
        if not p.get(key):
            return question
    return ""
