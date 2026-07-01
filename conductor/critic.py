"""Critic / Evaluator agent: accepts or rejects an optimization based on quality gate."""
from __future__ import annotations

from dataclasses import dataclass

from observability import RunMetrics

QUALITY_TOLERANCE   = 0.10   # accept if quality drops by at most 10 percentage points
LATENCY_MUST_IMPROVE = False  # if True, reject if latency doesn't improve
COST_MUST_IMPROVE   = False


@dataclass
class Verdict:
    accepted: bool
    reason: str
    latency_delta_pct: float   # negative = improvement
    cost_delta_pct: float
    quality_delta: float       # negative = regression


def evaluate(baseline: RunMetrics, candidate: RunMetrics) -> Verdict:
    """Compare candidate run against baseline; return accept/reject verdict."""
    latency_delta = (
        (candidate.avg_latency_ms - baseline.avg_latency_ms)
        / max(baseline.avg_latency_ms, 1)
    )
    cost_delta = (
        (candidate.total_cost_usd - baseline.total_cost_usd)
        / max(baseline.total_cost_usd, 1e-9)
    )
    quality_delta = candidate.quality_score - baseline.quality_score

    # Quality gate
    if quality_delta < -QUALITY_TOLERANCE:
        return Verdict(
            accepted=False,
            reason=(
                f"Quality regression: {baseline.quality_score:.2%} → "
                f"{candidate.quality_score:.2%} "
                f"(delta {quality_delta:+.2%}, threshold ±{QUALITY_TOLERANCE:.0%})"
            ),
            latency_delta_pct=latency_delta * 100,
            cost_delta_pct=cost_delta * 100,
            quality_delta=quality_delta,
        )

    reason_parts = []
    if latency_delta < 0:
        reason_parts.append(f"latency {latency_delta*100:+.1f}%")
    if cost_delta < 0:
        reason_parts.append(f"cost {cost_delta*100:+.1f}%")
    if not reason_parts:
        reason_parts.append("no latency or cost improvement — marginal win")

    return Verdict(
        accepted=True,
        reason="Quality preserved. " + ", ".join(reason_parts) + ".",
        latency_delta_pct=latency_delta * 100,
        cost_delta_pct=cost_delta * 100,
        quality_delta=quality_delta,
    )
