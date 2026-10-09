"""Load persisted parsed documents."""

from __future__ import annotations

from pathlib import Path

from contractguard.ingestion.pdf import ParsedDocument


def load_parsed_document(json_path: str | Path) -> ParsedDocument:
    """Load a persisted ``ParsedDocument`` JSON file."""
    path = Path(json_path)
    return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
