"""Target workflow: a deliberately naive 3-step research pipeline.

Baseline behaviour (configs/baseline.yaml):
  - Every step uses the big model.
  - Retrieve calls run serially.

Conductor optimizes this by swapping models, enabling parallel retrieve, and caching.
"""
from __future__ import annotations

import concurrent.futures
import copy
import time
from dataclasses import dataclass
from typing import Optional

import yaml

from llm_client import NvidiaClient, LLMResponse
from observability import StepTrace, WorkflowTrace

# Shared retrieve cache persists across eval runs in the same process (demo realism).
_RETRIEVE_CACHE: dict[tuple[str, str], LLMResponse] = {}


def clear_retrieve_cache() -> None:
    _RETRIEVE_CACHE.clear()


@dataclass
class StepConfig:
    model: str
    max_tokens: int


@dataclass
class WorkflowConfig:
    decompose: StepConfig
    retrieve: StepConfig
    synthesize: StepConfig
    retrieve_mode: str   # "serial" | "parallel"
    cache_retrieve: bool = False
    name: str = "unnamed"

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        def sc(key: str) -> StepConfig:
            return StepConfig(**raw["steps"][key])

        execution = raw.get("execution", {})
        return cls(
            decompose=sc("decompose"),
            retrieve=sc("retrieve"),
            synthesize=sc("synthesize"),
            retrieve_mode=execution.get("retrieve_mode", "serial"),
            cache_retrieve=execution.get("cache_retrieve", False),
            name=path.split("/")[-1].replace(".yaml", ""),
        )

    def to_yaml(self) -> str:
        return yaml.dump(
            {
                "steps": {
                    "decompose": {"model": self.decompose.model, "max_tokens": self.decompose.max_tokens},
                    "retrieve": {"model": self.retrieve.model, "max_tokens": self.retrieve.max_tokens},
                    "synthesize": {"model": self.synthesize.model, "max_tokens": self.synthesize.max_tokens},
                },
                "execution": {
                    "retrieve_mode": self.retrieve_mode,
                    "cache_retrieve": self.cache_retrieve,
                },
            },
            default_flow_style=False,
        )


NUM_SUBQUESTIONS = 2


class ResearchPipeline:
    """Three-step research pipeline: Decompose → Retrieve × N → Synthesize."""

    def __init__(self, config: WorkflowConfig, client: Optional[NvidiaClient] = None):
        self.config = config
        self.client = client or NvidiaClient()

    def run(
        self,
        question_id: int,
        question: str,
        mock_key_terms: Optional[list[str]] = None,
    ) -> WorkflowTrace:
        trace = WorkflowTrace(question_id=question_id, question=question, answer="")

        decomp_resp = self._decompose(question, mock_key_terms)
        trace.steps.append(_to_step_trace("decompose", decomp_resp))
        sub_questions = _parse_sub_questions(decomp_resp.content)

        if self.config.retrieve_mode == "parallel":
            retrieve_resps, _ = self._retrieve_parallel(sub_questions, mock_key_terms)
        else:
            retrieve_resps, _ = self._retrieve_serial(sub_questions, mock_key_terms)

        for r, cached in retrieve_resps:
            trace.steps.append(_to_step_trace("retrieve", r, cached=cached))
        summaries = [r.content for r, _ in retrieve_resps]

        synth_resp = self._synthesize(question, summaries, mock_key_terms)
        trace.steps.append(_to_step_trace("synthesize", synth_resp))

        trace.answer = synth_resp.content
        return trace

    def _decompose(self, question: str, mock_key_terms=None) -> LLMResponse:
        cfg = self.config.decompose
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at breaking complex questions into exactly "
                    f"{NUM_SUBQUESTIONS} focused sub-questions. "
                    "Respond ONLY with a numbered list, one sub-question per line."
                ),
            },
            {"role": "user", "content": f"Break this into sub-questions: {question}"},
        ]
        return self.client.chat(
            cfg.model, messages, cfg.max_tokens,
            step_name="decompose", mock_key_terms=mock_key_terms,
        )

    def _retrieve_serial(
        self, sub_questions: list[str], mock_key_terms=None
    ) -> tuple[list[tuple[LLMResponse, bool]], float]:
        t0 = time.monotonic()
        resps = [self._retrieve_one(q, mock_key_terms) for q in sub_questions]
        return resps, (time.monotonic() - t0) * 1000

    def _retrieve_parallel(
        self, sub_questions: list[str], mock_key_terms=None
    ) -> tuple[list[tuple[LLMResponse, bool]], float]:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_SUBQUESTIONS) as ex:
            futures = [ex.submit(self._retrieve_one, q, mock_key_terms) for q in sub_questions]
            resps = [f.result() for f in futures]
        return resps, (time.monotonic() - t0) * 1000

    def _retrieve_one(self, sub_question: str, mock_key_terms=None) -> tuple[LLMResponse, bool]:
        cfg = self.config.retrieve
        cache_key = (cfg.model, sub_question.strip().lower())

        if self.config.cache_retrieve and cache_key in _RETRIEVE_CACHE:
            cached = copy.copy(_RETRIEVE_CACHE[cache_key])
            cached.latency_ms = 0.5
            cached.cost_usd = 0.0
            cached.prompt_tokens = 0
            cached.completion_tokens = 0
            return cached, True

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Find and summarize key facts "
                    "that answer the question. Be specific and concise."
                ),
            },
            {"role": "user", "content": f"Find information about: {sub_question}"},
        ]
        resp = self.client.chat(
            cfg.model, messages, cfg.max_tokens,
            step_name="retrieve", mock_key_terms=mock_key_terms,
        )
        if self.config.cache_retrieve:
            _RETRIEVE_CACHE[cache_key] = resp
        return resp, False

    def _synthesize(self, question: str, summaries: list[str], mock_key_terms=None) -> LLMResponse:
        cfg = self.config.synthesize
        combined = "\n\n".join(f"Source {i+1}: {s}" for i, s in enumerate(summaries))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a synthesis expert. Combine the provided research summaries "
                    "into a concise, accurate final answer. Be specific — name exact values, "
                    "names, or symbols when they are the answer."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nResearch:\n{combined}\n\nFinal answer:",
            },
        ]
        return self.client.chat(
            cfg.model, messages, cfg.max_tokens,
            step_name="synthesize", mock_key_terms=mock_key_terms,
        )


def _to_step_trace(step: str, resp: LLMResponse, cached: bool = False) -> StepTrace:
    return StepTrace(
        step=step,
        model=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
        cost_usd=resp.cost_usd,
        cached=cached,
    )


def _parse_sub_questions(text: str) -> list[str]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    questions = []
    for line in lines:
        for sep in (". ", ") ", "- "):
            if sep in line[:4]:
                line = line.split(sep, 1)[-1].strip()
                break
        if line:
            questions.append(line)
    while len(questions) < NUM_SUBQUESTIONS:
        questions.append(questions[0] if questions else "Provide more detail.")
    return questions[:NUM_SUBQUESTIONS]
