"""Strategist agent: reasons over the profile and proposes one optimization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_client import NvidiaClient, BIG_MODEL, SMALL_MODEL
from target_workflow import WorkflowConfig, StepConfig
from conductor.profiler import ProfileReport


@dataclass
class Optimization:
    id: int
    name: str
    description: str
    rationale: str
    config_delta: dict[str, Any]   # human-readable summary of what changes


# The fixed sequence of optimizations Conductor will attempt (in order).
# Conductor advances only after the previous is accepted.
_OPTIMIZATION_SEQUENCE = [
    {
        "id": 1,
        "name": "route_decompose_to_small",
        "description": "Route the decompose step to Nemotron Nano",
        "rationale": (
            "Decompose only reformats the question into sub-questions — a simple "
            "structural task that a small model handles just as well at 10× lower cost."
        ),
        "config_delta": {"steps.decompose.model": f"big → small ({SMALL_MODEL})"},
    },
    {
        "id": 2,
        "name": "parallelize_retrieve",
        "description": "Run the retrieve sub-calls in parallel (ThreadPoolExecutor)",
        "rationale": (
            "The three retrieve calls are fully independent. Running them concurrently "
            "reduces wall-clock latency to max(retrieve) instead of sum(retrieve)."
        ),
        "config_delta": {"execution.retrieve_mode": "serial → parallel"},
    },
    {
        "id": 3,
        "name": "route_synthesize_to_small",
        "description": "Route the synthesize step to Nemotron Nano",
        "rationale": (
            "Synthesize is the most reasoning-heavy step. Routing it to the small model "
            "would cut cost, but risks losing accuracy on complex multi-hop answers."
        ),
        "config_delta": {"steps.synthesize.model": f"big → small ({SMALL_MODEL})"},
    },
]


class Strategist:
    """Proposes the next optimization to try given the current profile and iteration."""

    def __init__(self, client: NvidiaClient | None = None):
        self.client = client or NvidiaClient()

    def propose(self, profile: ProfileReport, iteration: int) -> Optimization | None:
        """Return the next Optimization, or None when the sequence is exhausted."""
        if iteration >= len(_OPTIMIZATION_SEQUENCE):
            return None

        raw = _OPTIMIZATION_SEQUENCE[iteration]
        opt = Optimization(**raw)

        # In mock mode (or when strategist reasoning would be expensive) we trust the
        # pre-defined sequence.  With a real key, we can ask the model to confirm the
        # rationale — this is where NAT traces would feed into a real Nemotron call.
        if not self.client.mock:
            opt.rationale = self._llm_rationale(profile, opt)

        return opt

    # ── LLM-backed rationale (real mode only) ─────────────────────────────

    def _llm_rationale(self, profile: ProfileReport, opt: Optimization) -> str:
        profile_txt = json.dumps(
            [
                {
                    "step": p.step,
                    "avg_latency_ms": round(p.avg_latency_ms, 1),
                    "avg_tokens": round(p.avg_tokens, 1),
                    "complexity": p.complexity,
                    "pct_latency": round(p.pct_latency * 100, 1),
                }
                for p in profile.steps
            ],
            indent=2,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert ML infrastructure engineer optimising an agentic "
                    "workflow. Given the profiling data and a proposed optimization, "
                    "write a 2–3 sentence rationale explaining WHY this optimization is "
                    "safe and likely to improve cost/latency without harming quality."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Profile:\n{profile_txt}\n\n"
                    f"Proposed optimization: {opt.name}\n"
                    f"Description: {opt.description}\n\n"
                    "Rationale:"
                ),
            },
        ]
        resp = self.client.chat(BIG_MODEL, messages, max_tokens=200)
        return resp.content.strip()


def apply_optimization(config: WorkflowConfig, opt: Optimization) -> WorkflowConfig:
    """Return a new WorkflowConfig with the optimization applied."""
    import copy
    cfg = copy.deepcopy(config)

    if opt.name == "route_decompose_to_small":
        cfg.decompose = StepConfig(model=SMALL_MODEL, max_tokens=cfg.decompose.max_tokens)
        cfg.name = f"opt{opt.id}_decompose_small"

    elif opt.name == "parallelize_retrieve":
        cfg.retrieve_mode = "parallel"
        cfg.name = f"opt{opt.id}_parallel_retrieve"

    elif opt.name == "route_synthesize_to_small":
        cfg.synthesize = StepConfig(model=SMALL_MODEL, max_tokens=cfg.synthesize.max_tokens)
        cfg.name = f"opt{opt.id}_synthesize_small"

    return cfg
