"""
Load Cypher-style scheme rules (exclusions, relationships) without Neo4j.

Place your file at: data/schemes.cypher
After ingest, statements are compiled to: data/graph_compiled.json
At query time we filter statements relevant to retrieved scheme names.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_CYPHER_PATH = os.path.join(DATA_DIR, "schemes.cypher")
GRAPH_COMPILED_PATH = os.path.join(DATA_DIR, "graph_compiled.json")


def _strip_line_comments(line: str) -> str:
    if "//" in line:
        in_string = False
        quote = ""
        out: List[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if not in_string and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            if ch in ('"', "'"):
                if not in_string:
                    in_string = True
                    quote = ch
                elif ch == quote:
                    in_string = False
                    quote = ""
            out.append(ch)
            i += 1
        return "".join(out).rstrip()
    return line


def parse_cypher_statements(text: str) -> List[str]:
    """Split file into rough statements on ';' and strip // comments."""
    lines = [_strip_line_comments(ln) for ln in text.splitlines()]
    joined = "\n".join(lines)
    parts = re.split(r";\s*\n", joined)
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if s and not s.startswith("//"):
            out.append(s)
    return out


def compile_cypher_file(cypher_path: str) -> Dict[str, Any]:
    if not os.path.isfile(cypher_path):
        return {"statements": [], "source": None}
    with open(cypher_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    statements = parse_cypher_statements(text)
    return {
        "statements": statements,
        "source": os.path.abspath(cypher_path),
        "count": len(statements),
    }


def save_compiled_graph(data: Dict[str, Any], out_path: str = GRAPH_COMPILED_PATH) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_compiled_graph(path: str = GRAPH_COMPILED_PATH) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {"statements": [], "source": None}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_global_rule(statement: str) -> bool:
    sl = statement.lower()
    return any(
        k in sl
        for k in (
            "excl",
            "exclusion",
            "not eligible",
            "ineligible",
            "constraint",
            "global",
        )
    )


def filter_statements_for_schemes(statements: List[str], scheme_names: List[str]) -> List[str]:
    names = [n.strip().lower() for n in scheme_names if n and str(n).strip()]
    if not statements:
        return []
    picked: List[str] = []
    seen = set()
    for st in statements:
        sl = st.lower()
        if any(n in sl for n in names if len(n) >= 3):
            if st not in seen:
                seen.add(st)
                picked.append(st)
    for st in statements:
        if _is_global_rule(st) and st not in seen:
            seen.add(st)
            picked.append(st)
    return picked[:80]


def format_graph_context_for_llm(statements: List[str], max_chars_per_stmt: int = 600) -> str:
    if not statements:
        return ""
    lines = ["KNOWLEDGE GRAPH (from schemes.cypher; rules & relationships — apply when relevant):"]
    for st in statements:
        one = " ".join(st.split())
        if len(one) > max_chars_per_stmt:
            one = one[: max_chars_per_stmt - 3] + "..."
        lines.append(f"- {one}")
    return "\n".join(lines)


def graph_addon_for_metadatas(metas: List[Dict[str, Any]]) -> str:
    names: List[str] = []
    for m in metas:
        n = m.get("scheme_name")
        if n:
            names.append(str(n))
    data = load_compiled_graph()
    stmts = data.get("statements") or []
    filtered = filter_statements_for_schemes(stmts, names)
    return format_graph_context_for_llm(filtered)
