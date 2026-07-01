"""Main Conductor optimization loop.

Profile → Propose → Approve → Execute → Evaluate → Accept / Revert → repeat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llm_client import NvidiaClient
from observability import RunMetrics
from target_workflow import WorkflowConfig
from conductor.profiler import ProfileReport, analyze
from conductor.strategist import Strategist, Optimization, apply_optimization
from conductor.executor import run_eval
from conductor.critic import Verdict, evaluate
from conductor.config_io import config_changes, save_config


@dataclass
class IterationResult:
    iteration: int
    optimization: Optimization
    candidate_metrics: RunMetrics | None
    verdict: Verdict | None
    accepted_config: WorkflowConfig
    config_changes: list[dict[str, str]] = field(default_factory=list)
    skipped_by_user: bool = False


@dataclass
class ConductorResult:
    baseline: RunMetrics
    baseline_config: WorkflowConfig
    profile: ProfileReport | None = None
    iterations: list[IterationResult] = field(default_factory=list)
    saved_configs: list[Path] = field(default_factory=list)

    @property
    def final_config(self) -> WorkflowConfig | None:
        for it in reversed(self.iterations):
            if it.verdict and it.verdict.accepted:
                return it.accepted_config
        return None

    @property
    def accepted(self) -> list[IterationResult]:
        return [it for it in self.iterations if it.verdict and it.verdict.accepted]

    @property
    def rejected(self) -> list[IterationResult]:
        return [it for it in self.iterations if it.verdict and not it.verdict.accepted]

    @property
    def skipped(self) -> list[IterationResult]:
        return [it for it in self.iterations if it.skipped_by_user]

    def total_latency_improvement_pct(self) -> float:
        if not self.accepted:
            return 0.0
        last = self.accepted[-1].candidate_metrics
        assert last is not None
        return (
            (self.baseline.avg_latency_ms - last.avg_latency_ms)
            / max(self.baseline.avg_latency_ms, 1)
        ) * 100

    def total_cost_improvement_pct(self) -> float:
        if not self.accepted:
            return 0.0
        last = self.accepted[-1].candidate_metrics
        assert last is not None
        return (
            (self.baseline.total_cost_usd - last.total_cost_usd)
            / max(self.baseline.total_cost_usd, 1e-9)
        ) * 100


ApprovalFn = Callable[[Optimization, WorkflowConfig, WorkflowConfig, list[dict[str, str]]], bool]


class ConductorLoop:
    def __init__(
        self,
        baseline_config: WorkflowConfig,
        eval_set: list[dict],
        client: NvidiaClient | None = None,
        progress_cb: Callable[[str], None] | None = None,
        approval_fn: ApprovalFn | None = None,
        auto_approve: bool = False,
        save_dir: Path | str | None = Path("configs"),
        run_baseline: bool = True,
        baseline_metrics: RunMetrics | None = None,
    ):
        self.baseline_config = baseline_config
        self.eval_set = eval_set
        self.client = client or NvidiaClient()
        self.strategist = Strategist(client=self.client)
        self.progress_cb = progress_cb or (lambda msg: None)
        self.approval_fn = approval_fn
        self.auto_approve = auto_approve
        self.save_dir = Path(save_dir) if save_dir else None

        self._baseline_metrics = baseline_metrics
        self._run_baseline = run_baseline

    def run(self) -> ConductorResult:
        if self._run_baseline or self._baseline_metrics is None:
            self.progress_cb("Running BASELINE…")
            baseline_metrics = run_eval(self.baseline_config, self.eval_set, self.client)
        else:
            baseline_metrics = self._baseline_metrics

        profile = analyze(baseline_metrics.traces, self.baseline_config.retrieve_mode)
        result = ConductorResult(
            baseline=baseline_metrics,
            baseline_config=self.baseline_config,
            profile=profile,
        )

        current_config = self.baseline_config
        current_metrics = baseline_metrics
        iteration = 0

        while True:
            self.progress_cb(f"[iter {iteration + 1}] Strategist proposing…")
            opt = self.strategist.propose(profile, iteration)
            if opt is None:
                self.progress_cb("No more optimizations to try. Done.")
                break

            candidate_config = apply_optimization(current_config, opt)
            changes = config_changes(current_config, candidate_config)

            approved = self.auto_approve
            if not approved:
                if self.approval_fn is not None:
                    approved = self.approval_fn(opt, current_config, candidate_config, changes)
                else:
                    approved = True

            if not approved:
                self.progress_cb(f"[iter {iteration + 1}] Skipped by user — not applied.")
                result.iterations.append(
                    IterationResult(
                        iteration=iteration + 1,
                        optimization=opt,
                        candidate_metrics=None,
                        verdict=None,
                        accepted_config=current_config,
                        config_changes=changes,
                        skipped_by_user=True,
                    )
                )
                iteration += 1
                continue

            self.progress_cb(f"[iter {iteration + 1}] Applying: {opt.description}")
            candidate_metrics = run_eval(candidate_config, self.eval_set, self.client)

            self.progress_cb(f"[iter {iteration + 1}] Evaluating…")
            verdict = evaluate(current_metrics, candidate_metrics)

            iter_result = IterationResult(
                iteration=iteration + 1,
                optimization=opt,
                candidate_metrics=candidate_metrics,
                verdict=verdict,
                accepted_config=candidate_config if verdict.accepted else current_config,
                config_changes=changes,
            )
            result.iterations.append(iter_result)

            if verdict.accepted:
                self.progress_cb(f"  ✓ ACCEPTED — {verdict.reason}")
                current_config = candidate_config
                current_metrics = candidate_metrics
                profile = analyze(current_metrics.traces, current_config.retrieve_mode)

                if self.save_dir:
                    out = self.save_dir / f"{candidate_config.name}.yaml"
                    save_config(candidate_config, out)
                    result.saved_configs.append(out)
                    self.progress_cb(f"  Saved config → {out}")
            else:
                self.progress_cb(f"  ✗ REJECTED — {verdict.reason} — reverting.")

            iteration += 1

        if result.final_config and self.save_dir:
            final_path = self.save_dir / "final.yaml"
            save_config(result.final_config, final_path)
            if final_path not in result.saved_configs:
                result.saved_configs.append(final_path)

        return result
