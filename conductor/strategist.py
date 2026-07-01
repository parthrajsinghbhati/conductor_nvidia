"""Strategist agent: profile-driven optimization proposals."""
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
    config_delta: dict[str, Any]


_OPTIMIZATION_CATALOG = [
    {
        "id": 1,
        "name": "route_decompose_to_small",
        "description": "Route the decompose step to the small model",
        "rationale": (
            "Decompose only reformats the question into sub-questions — a simple "
            "structural task that a small model handles just as well at lower cost."
        ),
        "config_delta": {"steps.decompose.model": f"big → small ({SMALL_MODEL})"},
    },
    {
        "id": 2,
        "name": "parallelize_retrieve",
        "description": "Run retrieve sub-calls in parallel (ThreadPoolExecutor)",
        "rationale": (
            "Retrieve calls are independent. Running them concurrently reduces "
            "wall-clock latency to max(retrieve) instead of sum(retrieve)."
        ),
        "config_delta": {"execution.retrieve_mode": "serial → parallel"},
    },
    {
        "id": 3,
        "name": "enable_retrieve_cache",
        "description": "Enable LRU cache for retrieve sub-calls",
        "rationale": (
            "Repeated sub-questions across the eval set can be served from cache, "
            "eliminating redundant LLM calls and token spend."
        ),
        "config_delta": {"execution.cache_retrieve": "false → true"},
    },
    {
        "id": 4,
        "name": "route_synthesize_to_small",
        "description": "Route the synthesize step to the small model",
        "rationale": (
            "Synthesize is reasoning-heavy. Routing to Nano cuts cost but may "
            "lose accuracy on multi-hop answers — must pass quality gate."
        ),
        "config_delta": {"steps.synthesize.model": f"big → small ({SMALL_MODEL})"},
    },
]


class Strategist:
    """Picks the next applicable optimization from the profile (not blind iteration)."""

    def __init__(self, client: NvidiaClient | None = None, skip_llm_extras: bool = False):
        self.client = client or NvidiaClient()
        self.skip_llm_extras = skip_llm_extras

    def propose(
        self,
        profile: ProfileReport,
        config: WorkflowConfig,
        tried_names: set[str],
    ) -> Optimization | None:
        for raw in _OPTIMIZATION_CATALOG:
            if raw["name"] in tried_names:
                continue
            if not self._is_applicable(raw["name"], profile, config):
                continue
            opt = Optimization(**raw)
            if not self.client.mock and not self.skip_llm_extras:
                opt.rationale = self._llm_rationale(profile, opt)
            return opt
        return None

    def _is_applicable(self, name: str, profile: ProfileReport, config: WorkflowConfig) -> bool:
        decompose_p = next((p for p in profile.steps if p.step == "decompose"), None)
        synthesize_p = next((p for p in profile.steps if p.step == "synthesize"), None)

        if name == "route_decompose_to_small":
            return (
                config.decompose.model != SMALL_MODEL
                and decompose_p is not None
                and decompose_p.complexity == "low"
            )
        if name == "parallelize_retrieve":
            return config.retrieve_mode == "serial"
        if name == "enable_retrieve_cache":
            return not config.cache_retrieve
        if name == "route_synthesize_to_small":
            return config.synthesize.model != SMALL_MODEL
        return False

    def _llm_rationale(self, profile: ProfileReport, opt: Optimization) -> str:
        profile_txt = json.dumps(
            [{"step": p.step, "complexity": p.complexity, "pct_latency": round(p.pct_latency * 100, 1)}
             for p in profile.steps],
            indent=2,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an ML infrastructure engineer. Given profiling data and a "
                    "proposed workflow optimization, write 2 sentences on why it is safe."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Profile:\n{profile_txt}\n\n"
                    f"Optimization: {opt.name}\n{opt.description}\n\nRationale:"
                ),
            },
        ]
        resp = self.client.chat(BIG_MODEL, messages, max_tokens=200)
        return resp.content.strip()


def apply_optimization(config: WorkflowConfig, opt: Optimization) -> WorkflowConfig:
    import copy
    cfg = copy.deepcopy(config)

    if opt.name == "route_decompose_to_small":
        cfg.decompose = StepConfig(model=SMALL_MODEL, max_tokens=cfg.decompose.max_tokens)
        cfg.name = f"opt{opt.id}_decompose_small"

    elif opt.name == "parallelize_retrieve":
        cfg.retrieve_mode = "parallel"
        cfg.name = f"opt{opt.id}_parallel_retrieve"

    elif opt.name == "enable_retrieve_cache":
        cfg.cache_retrieve = True
        cfg.name = f"opt{opt.id}_retrieve_cache"

    elif opt.name == "route_synthesize_to_small":
        cfg.synthesize = StepConfig(model=SMALL_MODEL, max_tokens=cfg.synthesize.max_tokens)
        cfg.name = f"opt{opt.id}_synthesize_small"

    return cfg
