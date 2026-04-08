"""
Map scheme display names to required documents using DEPENDS_ON edges in schemes.cypher.

PoC: regex-based parse (no Neo4j). Handles inline Document nodes and references like (BPLCard).
"""

from __future__ import annotations

import re
from typing import Dict, List

from graph_knowledge import DEFAULT_CYPHER_PATH


_SCHEME_LABEL_TO_NAME_RE = re.compile(
    r'CREATE\s+\(\s*(\w+)\s*:\s*Scheme\s*\{[^}]*\bname\s*:\s*"([^"]+)"',
    re.IGNORECASE | re.DOTALL,
)

# Any Document node pattern in the file (inline on relationships or standalone CREATE).
_DOC_NAME_RE = re.compile(
    r'(\w+)\s*:\s*Document\s*\{\s*name\s*:\s*"([^"]+)"\s*\}',
    re.IGNORECASE,
)

def _parse_depends_lines(text: str) -> List[tuple]:
    """
    Line-based parse so parentheses inside quoted document names do not break the pattern.
    Returns list of (scheme_label, target_inner) where target_inner is inside outer (...).
    """
    rows: List[tuple] = []
    for raw in text.splitlines():
        line = raw.strip()
        if "DEPENDS_ON" not in line or not line.upper().startswith("CREATE"):
            continue
        m = re.match(
            r"CREATE\s+\(\s*(\w+)\s*\)\s*-\s*\[\s*:DEPENDS_ON\s*\]\s*->\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if not m:
            continue
        scheme_label, rhs = m.group(1), m.group(2).strip()
        if rhs.endswith(";"):
            rhs = rhs[:-1].strip()
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        rows.append((scheme_label, rhs))
    return rows


def load_scheme_documents_from_cypher(cypher_path: str = DEFAULT_CYPHER_PATH) -> Dict[str, List[str]]:
    """
    Returns mapping: scheme display name -> ordered unique document names from DEPENDS_ON.
    """
    try:
        with open(cypher_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return {}

    label_to_scheme_name: Dict[str, str] = {}
    for m in _SCHEME_LABEL_TO_NAME_RE.finditer(text):
        label, display = m.group(1), m.group(2)
        label_to_scheme_name[label] = display

    doc_label_to_name: Dict[str, str] = {}
    for m in _DOC_NAME_RE.finditer(text):
        doc_label_to_name[m.group(1)] = m.group(2)

    by_scheme_label: Dict[str, List[str]] = {}
    for scheme_label, target in _parse_depends_lines(text):
        doc_names: List[str] = []
        inline = _DOC_NAME_RE.search(target)
        if inline:
            doc_names.append(inline.group(2))
        else:
            ref = re.match(r"^(\w+)$", target)
            if ref:
                dlabel = ref.group(1)
                if dlabel in doc_label_to_name:
                    doc_names.append(doc_label_to_name[dlabel])
        if not doc_names:
            continue
        by_scheme_label.setdefault(scheme_label, []).extend(doc_names)

    by_display: Dict[str, List[str]] = {}
    for scheme_label, docs in by_scheme_label.items():
        display = label_to_scheme_name.get(scheme_label)
        if not display:
            continue
        seen = set()
        uniq: List[str] = []
        for d in docs:
            if d in seen:
                continue
            seen.add(d)
            uniq.append(d)
        by_display[display] = uniq

    return by_display
