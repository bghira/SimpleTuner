from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_FILENAME = "schema.json"
TARGETS_FILENAME = "targets.safetensors"


def load_schema(prepared_dir: str | Path) -> dict[str, Any]:
    schema_path = Path(prepared_dir).expanduser() / SCHEMA_FILENAME
    if not schema_path.is_file():
        raise FileNotFoundError(f"Prompt2Effect schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError(f"Prompt2Effect schema must be a JSON object: {schema_path}")
    return schema


def save_schema(prepared_dir: str | Path, schema: dict[str, Any]) -> Path:
    prepared_dir = Path(prepared_dir).expanduser()
    prepared_dir.mkdir(parents=True, exist_ok=True)
    schema_path = prepared_dir / SCHEMA_FILENAME
    with schema_path.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return schema_path
