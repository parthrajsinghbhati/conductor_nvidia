"""Profiler agent: analyses WorkflowTraces and localises bottlenecks."""
from __future__ import annotations

from dataclasses import dataclass

from observability import WorkflowTrace


@dataclass
class StepProfile:
    step: str
    avg_latency_ms: float
    avg_tokens: float
    pct_latency: float    # fraction of total latency
    pct_cost: float       # fraction of total cost
    complexity: str       # "low" | "medium" | "high"


@dataclass
class ProfileReport:
    steps: list[StepProfile]
    total_avg_latency_ms: float
    total_avg_cost_usd: float
    retrieve_mode: str
    bottleneck: str        # step name that dominates latency
    notes: list[str]


# Heuristic thresholds
_LOW_TOKEN_CEILING    = 200   # avg completion tokens
_MEDIUM_TOKEN_CEILING = 350


def analyze(traces: list[WorkflowTrace], retrieve_mode: str = "serial") -> ProfileReport:
    """Aggregate traces into a ProfileReport."""
    if not traces:
        raise ValueError("No traces to analyse.")

    step_names = list(dict.fromkeys(s.step for t in traces for s in t.steps))

    # Aggregate per step
    def step_data(name: str):
        rows = [s for t in traces for s in t.steps if s.step == name]
        return rows

    total_latency = sum(sum(s.latency_ms for s in t.steps) for t in traces) / len(traces)
    total_cost    = sum(t.total_cost_usd for t in traces) / len(traces)

    step_profiles: list[StepProfile] = []
    for name in step_names:
        rows = step_data(name)
        if not rows:
            continue
        # For parallel retrieve, wall-clock per question = max(retrieve latencies)
        if name == "retrieve" and retrieve_mode == "parallel":
            # group by question_id
            by_q: dict[int, list] = {}
            for t in traces:
                for s in t.steps:
                    if s.step == name:
                        by_q.setdefault(t.question_id, []).append(s.latency_ms)
            avg_lat = sum(max(v) for v in by_q.values()) / max(len(by_q), 1)
        else:
            avg_lat = sum(r.latency_ms for r in rows) / len(rows)

        avg_tok  = sum(r.completion_tokens for r in rows) / len(rows)
        avg_cost = sum(r.cost_usd for r in rows) / len(rows)

        complexity = (
            "low"    if avg_tok < _LOW_TOKEN_CEILING    else
            "medium" if avg_tok < _MEDIUM_TOKEN_CEILING else
            "high"
        )

        step_profiles.append(StepProfile(
            step=name,
            avg_latency_ms=avg_lat,
            avg_tokens=avg_tok,
            pct_latency=avg_lat / max(total_latency, 1),
            pct_cost=avg_cost / max(total_cost, 1),
            complexity=complexity,
        ))

    bottleneck = max(step_profiles, key=lambda p: p.avg_latency_ms).step

    notes: list[str] = []
    decompose_p = next((p for p in step_profiles if p.step == "decompose"), None)
    retrieve_p  = next((p for p in step_profiles if p.step == "retrieve"),  None)
    synthesize_p= next((p for p in step_profiles if p.step == "synthesize"),None)

    if decompose_p and decompose_p.complexity == "low":
        notes.append("decompose: low complexity — candidate for small-model routing")
    if retrieve_p and retrieve_mode == "serial":
        notes.append("retrieve: independent calls run serially — candidate for parallelisation")
    if synthesize_p and synthesize_p.complexity == "high":
        notes.append("synthesize: high complexity — keep on big model")

    return ProfileReport(
        steps=step_profiles,
        total_avg_latency_ms=total_latency,
        total_avg_cost_usd=total_cost,
        retrieve_mode=retrieve_mode,
        bottleneck=bottleneck,
        notes=notes,
    )
