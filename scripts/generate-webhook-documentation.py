#!/usr/bin/env python3
"""
Generate a Markdown reference of SimpleTuner webhook lifecycle events.

This script scans the SimpleTuner codebase for calls to the helpers defined in
``simpletuner.helpers.webhooks.events`` and emits a table of discovered events.

Usage:
    python scripts/generate-webhook-documentation.py

The output is written to ``documentation/api/WEBHOOKS.md``.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Directory layout
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = REPO_ROOT / "simpletuner"
OUTPUT_PATH = REPO_ROOT / "documentation" / "api" / "WEBHOOKS.md"

# Targets to inspect: function name -> field extractors
TARGETS: Mapping[str, Mapping[str, Sequence[Tuple[str, Any]]]] = OrderedDict(
    {
        "lifecycle_stage_event": {
            "type": [("literal", "lifecycle.stage")],
            "stage_key": [("keyword", "key"), ("positional", 0)],
            "status": [("keyword", "status")],
            "label": [("keyword", "label")],
            "message": [("keyword", "message")],
            "severity": [("keyword", "severity")],
        },
        "training_status_event": {
            "type": [("literal", "training.status")],
            "status": [("keyword", "status"), ("positional", 0)],
            "message": [("keyword", "message")],
            "severity": [("keyword", "severity")],
        },
        "notification_event": {
            "type": [("literal", "notification")],
            "message": [("positional", 0), ("keyword", "message")],
            "title": [("keyword", "title")],
            "severity": [("keyword", "severity")],
        },
        "error_event": {
            "type": [("literal", "error")],
            "message": [("positional", 0), ("keyword", "message")],
            "title": [("keyword", "title")],
        },
        "checkpoint_event": {
            "type": [("literal", "training.checkpoint")],
            "label": [("keyword", "label")],
            "path": [("keyword", "path")],
            "is_final": [("keyword", "is_final")],
            "severity": [("keyword", "severity")],
        },
    }
)


@dataclass
class EventRecord:
    """Single event emission site."""

    func: str
    fields: MutableMapping[str, Optional[str]]
    file: Path
    line: int

    @property
    def rel_path(self) -> Path:
        try:
            return self.file.relative_to(REPO_ROOT)
        except ValueError:
            return self.file


def discover_python_files() -> List[Path]:
    """Return Python files that should be scanned for events."""
    if not SOURCE_ROOT.exists():
        print(f"[!] Expected source directory {SOURCE_ROOT} not found", file=sys.stderr)
        sys.exit(1)
    return sorted(path for path in SOURCE_ROOT.rglob("*.py") if path.is_file())


def get_function_name(node: ast.AST) -> Optional[str]:
    """Extract the function name called by *node* if available."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def resolve_attribute(node: ast.Attribute) -> str:
    parts: List[str] = []
    current: Any = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        snippet = ast.get_source_segment("", node) or ""
        parts.append(snippet)
    return ".".join(reversed(parts))


def literal_value(node: ast.AST, source: str) -> Optional[Any]:
    """Attempt to resolve *node* to a literal value."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return resolve_attribute(node)
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                segment = ast.get_source_segment(source, value.value)
                if segment:
                    parts.append("{" + segment.strip() + "}")
                else:
                    parts.append("{...}")
        return "".join(parts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = literal_value(node.operand, source)
        if isinstance(operand, (int, float)):
            return -operand
    return None


def format_value(node: Optional[ast.AST], source: str) -> Optional[str]:
    if node is None:
        return None
    value = literal_value(node, source)
    if value is not None:
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)
    snippet = ast.get_source_segment(source, node)
    if snippet:
        return snippet.strip()
    return None


def extract_arg(node: ast.Call, source: str, spec: Sequence[Tuple[str, Any]]) -> Optional[str]:
    for kind, value in spec:
        if kind == "literal":
            return str(value)
        if kind == "keyword":
            for keyword in node.keywords:
                if keyword.arg == value:
                    return format_value(keyword.value, source)
        elif kind == "positional":
            index = int(value)
            if index < len(node.args):
                return format_value(node.args[index], source)
    return None


class EventCollector(ast.NodeVisitor):
    """AST visitor that records webhook helper invocations."""

    def __init__(self, file_path: Path, source: str, results: MutableMapping[str, List[EventRecord]]):
        self.file_path = file_path
        self.source = source
        self.results = results

    def visit_Call(self, node: ast.Call) -> Any:
        func_name = get_function_name(node.func)
        if func_name in TARGETS:
            fields_spec = TARGETS[func_name]
            fields: MutableMapping[str, Optional[str]] = {}
            for field_name, spec in fields_spec.items():
                fields[field_name] = extract_arg(node, self.source, spec)
            record = EventRecord(func=func_name, fields=fields, file=self.file_path, line=node.lineno)
            self.results[func_name].append(record)
        self.generic_visit(node)


def collect_events(files: Iterable[Path]) -> Mapping[str, List[EventRecord]]:
    results: MutableMapping[str, List[EventRecord]] = defaultdict(list)
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # Skip non-UTF8 files
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        EventCollector(path, source, results).visit(tree)
    return results


def make_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    if not rows:
        return ["_No occurrences found._", ""]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    table_lines = [header_line, separator_line]
    table_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    table_lines.append("")
    return table_lines


def build_markdown(events: Mapping[str, List[EventRecord]]) -> str:
    lines: List[str] = []
    lines.append("# Webhook Event Reference")
    lines.append("")
    lines.append("_This file is auto-generated by `scripts/generate-webhook-documentation.py`. Do not edit manually._")
    lines.append("")

    for func_name, field_spec in TARGETS.items():
        records = events.get(func_name, [])
        display_name = field_spec.get("type", [("literal", func_name)])[0][1]
        lines.append(f"## {display_name}")
        lines.append("")
        if func_name == "lifecycle_stage_event":
            headers = ["Stage Key", "Status", "Label", "Message", "Severity", "Source"]
            rows = [
                [
                    record.fields.get("stage_key") or "—",
                    record.fields.get("status") or "—",
                    record.fields.get("label") or "—",
                    record.fields.get("message") or "—",
                    record.fields.get("severity") or "—",
                    f"{record.rel_path}:{record.line}",
                ]
                for record in sorted(
                    records,
                    key=lambda r: (r.fields.get("stage_key") or "", str(r.rel_path), r.line),
                )
            ]
        elif func_name == "training_status_event":
            headers = ["Status", "Message", "Severity", "Source"]
            rows = [
                [
                    record.fields.get("status") or "—",
                    record.fields.get("message") or "—",
                    record.fields.get("severity") or "—",
                    f"{record.rel_path}:{record.line}",
                ]
                for record in sorted(
                    records,
                    key=lambda r: (r.fields.get("status") or "", str(r.rel_path), r.line),
                )
            ]
        elif func_name in {"notification_event", "error_event"}:
            headers = ["Message", "Title", "Severity", "Source"]
            rows = [
                [
                    record.fields.get("message") or "—",
                    record.fields.get("title") or "—",
                    record.fields.get("severity") or ("error" if func_name == "error_event" else "—"),
                    f"{record.rel_path}:{record.line}",
                ]
                for record in sorted(
                    records,
                    key=lambda r: (r.fields.get("message") or "", str(r.rel_path), r.line),
                )
            ]
        elif func_name == "checkpoint_event":
            headers = ["Label", "Path", "Final", "Severity", "Source"]
            rows = [
                [
                    record.fields.get("label") or "—",
                    record.fields.get("path") or "—",
                    record.fields.get("is_final") or "—",
                    record.fields.get("severity") or "—",
                    f"{record.rel_path}:{record.line}",
                ]
                for record in sorted(
                    records,
                    key=lambda r: (r.fields.get("label") or "", str(r.rel_path), r.line),
                )
            ]
        else:
            headers = ["Field", "Value", "Source"]
            rows = [
                [field, value or "—", f"{record.rel_path}:{record.line}"]
                for record in records
                for field, value in record.fields.items()
            ]

        lines.extend(make_table(headers, rows))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    files = discover_python_files()
    events = collect_events(files)
    markdown = build_markdown(events)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
