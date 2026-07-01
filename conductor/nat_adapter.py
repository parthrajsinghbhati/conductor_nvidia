"""NAT-compatible trace export and optional NeMo Agent Toolkit bridge."""
from __future__ import annotations

import json
from pathlib import Path

from observability import WorkflowTrace

NAT_AVAILABLE = False
NAT_VERSION = None

try:
    import nvidia_nat  # noqa: F401
    NAT_AVAILABLE = True
    NAT_VERSION = getattr(nvidia_nat, "__version__", "unknown")
except ImportError:
    pass


def nat_status() -> dict:
    return {
        "available": NAT_AVAILABLE,
        "version": NAT_VERSION,
        "mode": "integrated" if NAT_AVAILABLE else "compatible-export",
    }


def export_nat_traces(traces: list[WorkflowTrace], out_dir: Path, run_name: str) -> Path:
    """Export traces in NAT profiler JSON shape for downstream `nat eval` analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_profiler_traces.json"

    requests = []
    for trace in traces:
        events = []
        for i, step in enumerate(trace.steps):
            events.append({
                "event_type": "llm_call",
                "step": step.step,
                "step_index": i,
                "model": step.model,
                "prompt_tokens": step.prompt_tokens,
                "completion_tokens": step.completion_tokens,
                "latency_ms": step.latency_ms,
                "cost_usd": step.cost_usd,
                "cached": step.cached,
            })
        requests.append({
            "request_id": trace.question_id,
            "question": trace.question,
            "answer": trace.answer,
            "events": events,
            "total_latency_ms": trace.total_latency_ms,
            "total_cost_usd": trace.total_cost_usd,
        })

    payload = {
        "source": "conductor",
        "nat_integration": nat_status(),
        "requests": requests,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def export_standardized_csv(traces: list[WorkflowTrace], out_dir: Path, run_name: str) -> Path:
    """Export NAT-style standardized_data CSV for pandas / slides."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_standardized_data.csv"
    lines = [
        "request_id,step,model,prompt_tokens,completion_tokens,latency_ms,cost_usd,cached"
    ]
    for trace in traces:
        for step in trace.steps:
            lines.append(
                f"{trace.question_id},{step.step},{step.model},"
                f"{step.prompt_tokens},{step.completion_tokens},"
                f"{step.latency_ms:.2f},{step.cost_usd:.8f},{step.cached}"
            )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
