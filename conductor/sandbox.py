"""Sandboxed execution — stages config in an isolated directory before apply."""
from __future__ import annotations

import shutil
from pathlib import Path

from target_workflow import WorkflowConfig

SANDBOX_DIR = Path("traces/sandbox")


def stage_config(config: WorkflowConfig, run_id: str) -> Path:
    """Write candidate config to sandbox dir (OpenShell-style staging). Returns path."""
    run_dir = SANDBOX_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "candidate.yaml"
    out.write_text(config.to_yaml())
    return out


def load_staged_config(path: Path, name: str) -> WorkflowConfig:
    cfg = WorkflowConfig.from_yaml(str(path))
    cfg.name = name
    return cfg
