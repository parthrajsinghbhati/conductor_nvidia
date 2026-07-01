"""Executor agent: applies a proposed config and runs the pipeline on the eval set."""
from __future__ import annotations

import time

from llm_client import NvidiaClient
from observability import RunMetrics
from target_workflow import WorkflowConfig, ResearchPipeline


def run_eval(
    config: WorkflowConfig,
    eval_set: list[dict],
    client: NvidiaClient | None = None,
    quality_fn=None,
) -> RunMetrics:
    """Run the pipeline over the eval set and return aggregated metrics."""
    pipeline = ResearchPipeline(config, client=client or NvidiaClient())
    traces = []
    wall_start = time.monotonic()

    for item in eval_set:
        trace = pipeline.run(
            item["id"],
            item["question"],
            mock_key_terms=item.get("key_terms"),
        )
        traces.append(trace)

    wall_ms = (time.monotonic() - wall_start) * 1000

    # Quality scoring
    if quality_fn is None:
        quality_fn = keyword_quality

    scores = [
        quality_fn(trace.answer, item.get("key_terms", []))
        for trace, item in zip(traces, eval_set)
    ]
    quality = sum(scores) / max(len(scores), 1)

    return RunMetrics(
        config_name=config.name,
        traces=traces,
        wall_latency_ms=wall_ms,
        quality_score=quality,
    )


def keyword_quality(answer: str, key_terms: list[str]) -> float:
    """Simple quality metric: fraction of key_terms present in the answer."""
    if not key_terms:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for term in key_terms if term.lower() in answer_lower)
    return hits / len(key_terms)
