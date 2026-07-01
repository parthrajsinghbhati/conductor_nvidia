"""Critic / Evaluator agent: accepts or rejects an optimization based on quality gate."""
from __future__ import annotations

from dataclasses import dataclass

from llm_client import NvidiaClient, SMALL_MODEL
from observability import RunMetrics

QUALITY_TOLERANCE = 0.10
# When True, reject a quality-preserving change that makes the given metric worse.
# Default False: the demo accepts any change that holds quality within tolerance.
LATENCY_MUST_IMPROVE = False
COST_MUST_IMPROVE = False


@dataclass
class Verdict:
    accepted: bool
    reason: str
    latency_delta_pct: float
    cost_delta_pct: float
    quality_delta: float
    llm_explanation: str = ""


def evaluate(
    baseline: RunMetrics,
    candidate: RunMetrics,
    client: NvidiaClient | None = None,
) -> Verdict:
    latency_delta = (
        (candidate.avg_latency_ms - baseline.avg_latency_ms)
        / max(baseline.avg_latency_ms, 1)
    )
    cost_delta = (
        (candidate.total_cost_usd - baseline.total_cost_usd)
        / max(baseline.total_cost_usd, 1e-9)
    )
    quality_delta = candidate.quality_score - baseline.quality_score

    if quality_delta < -QUALITY_TOLERANCE:
        reason = (
            f"Quality regression: {baseline.quality_score:.2%} → "
            f"{candidate.quality_score:.2%} "
            f"(delta {quality_delta:+.2%}, threshold ±{QUALITY_TOLERANCE:.0%})"
        )
        verdict = Verdict(
            accepted=False,
            reason=reason,
            latency_delta_pct=latency_delta * 100,
            cost_delta_pct=cost_delta * 100,
            quality_delta=quality_delta,
        )
    else:
        # Quality preserved. Optionally require that latency/cost did not regress.
        required_regressions = []
        if LATENCY_MUST_IMPROVE and latency_delta > 0:
            required_regressions.append(f"latency {latency_delta * 100:+.1f}%")
        if COST_MUST_IMPROVE and cost_delta > 0:
            required_regressions.append(f"cost {cost_delta * 100:+.1f}%")

        if required_regressions:
            verdict = Verdict(
                accepted=False,
                reason=(
                    "Quality preserved but a required metric regressed: "
                    + ", ".join(required_regressions) + "."
                ),
                latency_delta_pct=latency_delta * 100,
                cost_delta_pct=cost_delta * 100,
                quality_delta=quality_delta,
            )
        else:
            wins = []
            if latency_delta < 0:
                wins.append(f"latency {latency_delta * 100:+.1f}%")
            if cost_delta < 0:
                wins.append(f"cost {cost_delta * 100:+.1f}%")
            if candidate.total_cache_hits:
                wins.append(f"{candidate.total_cache_hits} cache hits")
            if wins:
                summary = ", ".join(wins)
            else:
                # No net improvement — report the deltas honestly, don't call it a win.
                summary = (
                    f"no net gain (latency {latency_delta * 100:+.1f}%, "
                    f"cost {cost_delta * 100:+.1f}%)"
                )
            verdict = Verdict(
                accepted=True,
                reason="Quality preserved. " + summary + ".",
                latency_delta_pct=latency_delta * 100,
                cost_delta_pct=cost_delta * 100,
                quality_delta=quality_delta,
            )

    if client and not client.mock:
        verdict.llm_explanation = _llm_explain(verdict, client)
    else:
        verdict.llm_explanation = verdict.reason

    return verdict


def _llm_explain(verdict: Verdict, client: NvidiaClient) -> str:
    status = "ACCEPT" if verdict.accepted else "REJECT"
    messages = [
        {
            "role": "system",
            "content": "Explain this workflow optimization verdict in one sentence for a demo audience.",
        },
        {
            "role": "user",
            "content": (
                f"Verdict: {status}\n"
                f"Latency delta: {verdict.latency_delta_pct:+.1f}%\n"
                f"Cost delta: {verdict.cost_delta_pct:+.1f}%\n"
                f"Quality delta: {verdict.quality_delta:+.2%}\n"
                f"Reason: {verdict.reason}\n\nExplanation:"
            ),
        },
    ]
    resp = client.chat(SMALL_MODEL, messages, max_tokens=100)
    return resp.content.strip()
