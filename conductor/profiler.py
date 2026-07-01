"""Profiler agent: analyses WorkflowTraces and localises bottlenecks."""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_client import NvidiaClient, SMALL_MODEL
from observability import WorkflowTrace

_LOW_TOKEN_CEILING = 200
_MEDIUM_TOKEN_CEILING = 350


@dataclass
class StepProfile:
    step: str
    avg_latency_ms: float
    avg_tokens: float
    pct_latency: float
    pct_cost: float
    complexity: str
    cache_hits: int = 0


@dataclass
class ProfileReport:
    steps: list[StepProfile]
    total_avg_latency_ms: float
    total_avg_cost_usd: float
    retrieve_mode: str
    cache_enabled: bool
    bottleneck: str
    notes: list[str] = field(default_factory=list)
    llm_summary: str = ""


def analyze(
    traces: list[WorkflowTrace],
    retrieve_mode: str = "serial",
    cache_enabled: bool = False,
    client: NvidiaClient | None = None,
) -> ProfileReport:
    if not traces:
        raise ValueError("No traces to analyse.")

    step_names = list(dict.fromkeys(s.step for t in traces for s in t.steps))
    total_latency = sum(sum(s.latency_ms for s in t.steps) for t in traces) / len(traces)
    total_cost = sum(t.total_cost_usd for t in traces) / len(traces)

    step_profiles: list[StepProfile] = []
    for name in step_names:
        rows = [s for t in traces for s in t.steps if s.step == name]
        if not rows:
            continue

        if name == "retrieve" and retrieve_mode == "parallel":
            by_q: dict[int, list] = {}
            for t in traces:
                for s in t.steps:
                    if s.step == name:
                        by_q.setdefault(t.question_id, []).append(s.latency_ms)
            avg_lat = sum(max(v) for v in by_q.values()) / max(len(by_q), 1)
        else:
            avg_lat = sum(r.latency_ms for r in rows) / len(rows)

        avg_tok = sum(r.completion_tokens for r in rows) / len(rows)
        avg_cost = sum(r.cost_usd for r in rows) / len(rows)
        hits = sum(1 for r in rows if r.cached)

        complexity = (
            "low" if avg_tok < _LOW_TOKEN_CEILING else
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
            cache_hits=hits,
        ))

    bottleneck = max(step_profiles, key=lambda p: p.avg_latency_ms).step
    notes: list[str] = []

    decompose_p = next((p for p in step_profiles if p.step == "decompose"), None)
    retrieve_p = next((p for p in step_profiles if p.step == "retrieve"), None)
    synthesize_p = next((p for p in step_profiles if p.step == "synthesize"), None)

    if decompose_p and decompose_p.complexity == "low":
        notes.append("decompose: low complexity — candidate for small-model routing")
    if retrieve_p and retrieve_mode == "serial":
        notes.append("retrieve: independent calls run serially — candidate for parallelisation")
    if retrieve_p and not cache_enabled:
        notes.append("retrieve: repeated sub-questions possible — candidate for caching")
    if retrieve_p and retrieve_p.cache_hits:
        notes.append(f"retrieve: {retrieve_p.cache_hits} cache hits in this run")
    if synthesize_p and synthesize_p.complexity == "high":
        notes.append("synthesize: high complexity — keep on big model")

    report = ProfileReport(
        steps=step_profiles,
        total_avg_latency_ms=total_latency,
        total_avg_cost_usd=total_cost,
        retrieve_mode=retrieve_mode,
        cache_enabled=cache_enabled,
        bottleneck=bottleneck,
        notes=notes,
    )

    if client and not client.mock:
        report.llm_summary = _llm_summarize(report, client)
    else:
        report.llm_summary = _mock_summarize(report)

    return report


def _mock_summarize(report: ProfileReport) -> str:
    top = report.bottleneck
    return (
        f"Bottleneck is '{top}'. "
        + " ".join(report.notes[:2])
        if report.notes else f"Bottleneck is '{top}'."
    )


def _llm_summarize(report: ProfileReport, client: NvidiaClient) -> str:
    lines = [f"- {p.step}: {p.avg_latency_ms:.0f}ms, complexity={p.complexity}" for p in report.steps]
    messages = [
        {
            "role": "system",
            "content": "Summarize this agent workflow profile in 2 sentences for an infra engineer.",
        },
        {"role": "user", "content": "Profile:\n" + "\n".join(lines) + "\n\nSummary:"},
    ]
    resp = client.chat(SMALL_MODEL, messages, max_tokens=120)
    return resp.content.strip()
