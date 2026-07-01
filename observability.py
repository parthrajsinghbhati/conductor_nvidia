"""Trace dataclasses and cost/latency aggregation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepTrace:
    step: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class WorkflowTrace:
    question_id: int
    question: str
    answer: str
    steps: list[StepTrace] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def cache_hits(self) -> int:
        return sum(1 for s in self.steps if s.cached)


@dataclass
class RunMetrics:
    """Aggregated metrics over a full eval run."""
    config_name: str
    traces: list[WorkflowTrace]
    wall_latency_ms: float
    quality_score: float

    @property
    def avg_latency_ms(self) -> float:
        return self.wall_latency_ms / max(len(self.traces), 1)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.total_cost_usd for t in self.traces)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.traces)

    @property
    def total_cache_hits(self) -> int:
        return sum(t.cache_hits for t in self.traces)

    def summary(self) -> dict:
        return {
            "config": self.config_name,
            "questions": len(self.traces),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "quality_score": round(self.quality_score, 3),
            "cache_hits": self.total_cache_hits,
        }
