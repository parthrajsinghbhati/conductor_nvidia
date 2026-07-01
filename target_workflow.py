"""Target workflow: a deliberately naive 3-step research pipeline.

Baseline behaviour (configs/baseline.yaml):
  - Every step uses the big model.
  - Retrieve calls run serially.

Conductor optimizes this by swapping models and enabling parallel retrieve.
"""
from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Optional

import yaml

from llm_client import NvidiaClient, LLMResponse
from observability import StepTrace, WorkflowTrace


# ── Config ────────────────────────────────────────────────────────────────────

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
    name: str = "unnamed"

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        def sc(key: str) -> StepConfig:
            return StepConfig(**raw["steps"][key])

        return cls(
            decompose=sc("decompose"),
            retrieve=sc("retrieve"),
            synthesize=sc("synthesize"),
            retrieve_mode=raw.get("execution", {}).get("retrieve_mode", "serial"),
            name=path.split("/")[-1].replace(".yaml", ""),
        )

    def to_yaml(self) -> str:
        return yaml.dump(
            {
                "steps": {
                    "decompose": {"model": self.decompose.model, "max_tokens": self.decompose.max_tokens},
                    "retrieve":  {"model": self.retrieve.model,  "max_tokens": self.retrieve.max_tokens},
                    "synthesize":{"model": self.synthesize.model,"max_tokens": self.synthesize.max_tokens},
                },
                "execution": {"retrieve_mode": self.retrieve_mode},
            },
            default_flow_style=False,
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

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

        # ── Step 1: Decompose ────────────────────────────────────────────
        decomp_resp = self._decompose(question, mock_key_terms)
        trace.steps.append(_to_step_trace("decompose", decomp_resp))
        sub_questions = _parse_sub_questions(decomp_resp.content)

        # ── Step 2: Retrieve (serial or parallel) ────────────────────────
        if self.config.retrieve_mode == "parallel":
            retrieve_resps, _ = self._retrieve_parallel(sub_questions, mock_key_terms)
        else:
            retrieve_resps, _ = self._retrieve_serial(sub_questions, mock_key_terms)

        for r in retrieve_resps:
            trace.steps.append(_to_step_trace("retrieve", r))
        summaries = [r.content for r in retrieve_resps]

        # ── Step 3: Synthesize ───────────────────────────────────────────
        synth_resp = self._synthesize(question, summaries, mock_key_terms)
        trace.steps.append(_to_step_trace("synthesize", synth_resp))

        trace.answer = synth_resp.content
        return trace

    # ── private steps ─────────────────────────────────────────────────────

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
    ) -> tuple[list[LLMResponse], float]:
        t0 = time.monotonic()
        resps = [self._retrieve_one(q, mock_key_terms) for q in sub_questions]
        return resps, (time.monotonic() - t0) * 1000

    def _retrieve_parallel(
        self, sub_questions: list[str], mock_key_terms=None
    ) -> tuple[list[LLMResponse], float]:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_SUBQUESTIONS) as ex:
            futures = [ex.submit(self._retrieve_one, q, mock_key_terms) for q in sub_questions]
            resps = [f.result() for f in futures]
        return resps, (time.monotonic() - t0) * 1000

    def _retrieve_one(self, sub_question: str, mock_key_terms=None) -> LLMResponse:
        cfg = self.config.retrieve
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
        return self.client.chat(
            cfg.model, messages, cfg.max_tokens,
            step_name="retrieve", mock_key_terms=mock_key_terms,
        )

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_step_trace(step: str, resp: LLMResponse) -> StepTrace:
    return StepTrace(
        step=step,
        model=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
        cost_usd=resp.cost_usd,
    )


def _parse_sub_questions(text: str) -> list[str]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    questions = []
    for line in lines:
        # Strip leading "1." / "1)" / "- " etc.
        for sep in (". ", ") ", "- "):
            if sep in line[:4]:
                line = line.split(sep, 1)[-1].strip()
                break
        if line:
            questions.append(line)
    # Always return exactly NUM_SUBQUESTIONS entries
    while len(questions) < NUM_SUBQUESTIONS:
        questions.append(questions[0] if questions else "Provide more detail.")
    return questions[:NUM_SUBQUESTIONS]
