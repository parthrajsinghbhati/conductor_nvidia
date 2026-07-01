"""Config diff and persistence helpers."""
from __future__ import annotations

from pathlib import Path

from target_workflow import WorkflowConfig


def config_changes(before: WorkflowConfig, after: WorkflowConfig) -> list[dict[str, str]]:
    """Return human-readable before/after rows for each changed field."""
    rows: list[dict[str, str]] = []

    def _add(field: str, old: str, new: str):
        if old != new:
            rows.append({"field": field, "before": old, "after": new})

    _add("steps.decompose.model", before.decompose.model, after.decompose.model)
    _add("steps.retrieve.model", before.retrieve.model, after.retrieve.model)
    _add("steps.synthesize.model", before.synthesize.model, after.synthesize.model)
    _add("execution.retrieve_mode", before.retrieve_mode, after.retrieve_mode)
    return rows


def save_config(config: WorkflowConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_yaml())
    return path
