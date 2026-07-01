"""NVIDIA NIM API client with transparent mock fallback.

Real mode  : set NVIDIA_API_KEY in .env  → hits integrate.api.nvidia.com
Mock mode  : no key or MOCK_MODE=true    → simulated latencies + plausible text
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

BIG_MODEL   = "nvidia/nemotron-3-ultra-550b-a55b"
SMALL_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"

# Approximate cost per token (input ≈ output averaged for simplicity)
COST_PER_TOKEN: dict[str, float] = {
    BIG_MODEL:   7.99 / 1_000_000,
    SMALL_MODEL: 0.10 / 1_000_000,
}

# Mock: ms to generate one output token
MS_PER_TOKEN: dict[str, float] = {
    BIG_MODEL:   0.60,   # ~600 ms / 1000 tokens
    SMALL_MODEL: 0.18,   # ~180 ms / 1000 tokens
}


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float


class NvidiaClient:
    """Drop-in client for NVIDIA NIM endpoints."""

    def __init__(self, mock: bool | None = None):
        api_key = os.getenv("NVIDIA_API_KEY", "")
        env_mock = os.getenv("MOCK_MODE", "false").lower() == "true"
        self.mock = mock if mock is not None else (env_mock or not api_key)
        self._oai = None

        if not self.mock:
            try:
                from openai import OpenAI
                self._oai = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=api_key,
                )
            except ImportError:
                self.mock = True

    # ── public ────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        step_name: str = "",
        mock_key_terms: list[str] | None = None,
    ) -> LLMResponse:
        if self.mock:
            return self._mock(model, messages, max_tokens, step_name, mock_key_terms or [])
        return self._real(model, messages, max_tokens)

    # ── real API ──────────────────────────────────────────────────────────

    def _real(self, model: str, messages: list[dict], max_tokens: int) -> LLMResponse:
        t0 = time.monotonic()
        resp = self._oai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        u = resp.usage
        total = u.prompt_tokens + u.completion_tokens
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=model,
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=total * COST_PER_TOKEN.get(model, 0.5 / 1_000_000),
        )

    # ── mock ──────────────────────────────────────────────────────────────

    def _mock(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        step_name: str,
        key_terms: list[str],
    ) -> LLMResponse:
        prompt_words = sum(len(m.get("content", "").split()) for m in messages)
        prompt_tokens     = int(prompt_words * 1.35)
        completion_tokens = min(max_tokens, random.randint(90, 160))
        base_ms = completion_tokens * MS_PER_TOKEN.get(model, 0.40)
        latency_ms = base_ms + random.uniform(-10, 20)
        time.sleep(latency_ms / 1000)

        total = prompt_tokens + completion_tokens
        content = self._mock_content(model, messages, step_name, key_terms)
        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=total * COST_PER_TOKEN.get(model, 0.5 / 1_000_000),
        )

    def _mock_content(
        self, model: str, messages: list[dict], step_name: str, key_terms: list[str]
    ) -> str:
        user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        subject = self._extract_subject(user_msg)

        if step_name == "decompose":
            return (
                f"1. What is the definition and origin of {subject}?\n"
                f"2. What are the key properties or characteristics of {subject}?\n"
                f"3. What is the modern significance or application of {subject}?"
            )

        if step_name == "retrieve":
            answer_hint = key_terms[0] if key_terms else subject
            return (
                f"Research summary: {answer_hint.capitalize()} is a well-documented subject. "
                f"Primary sources confirm that {answer_hint} exhibits the expected properties. "
                f"The evidence strongly supports its established role and classification."
            )

        if step_name == "synthesize":
            if model == SMALL_MODEL:
                # Deliberately vague — triggers the quality regression the demo needs to catch
                return (
                    "The topic involves several interrelated factors. "
                    "Based on the gathered information, the situation is complex and "
                    "context-dependent, requiring further analysis to draw firm conclusions."
                )
            # Big model — explicitly includes all key terms so quality scoring works
            if key_terms:
                terms_str = ", ".join(key_terms)
                return (
                    f"Based on comprehensive research, the answer is: {terms_str}. "
                    f"Multiple independent sources confirm these values. "
                    f"The subject ({key_terms[0]}) is precisely identified by these "
                    f"characteristics and has significant relevance in this domain."
                )
            return (
                f"Based on comprehensive research, {subject} is the well-supported answer. "
                f"Multiple sources confirm the established properties of {subject}."
            )

        # Strategist / critic internal calls
        return (
            "Analysis complete. The evidence strongly supports the proposed approach. "
            "Confidence is high based on the available data."
        )

    @staticmethod
    def _extract_subject(text: str) -> str:
        """Pull a 1-2 word subject from the question for mock responses."""
        skip = {
            "what", "which", "who", "where", "when", "how", "why", "is", "are",
            "the", "a", "an", "of", "in", "on", "and", "or", "to", "for",
            "its", "their", "this", "that", "with", "has", "have", "was", "were",
        }
        words = [w.strip(".,?!\"'").lower() for w in text.split()]
        candidates = [w for w in words if w and w not in skip and len(w) > 3]
        return " ".join(candidates[:2]) if candidates else "this subject"
