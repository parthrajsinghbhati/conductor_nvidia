"""Main Conductor optimization loop.

Profile → Propose → Approve → Sandbox → Execute → Evaluate → Accept / Revert → repeat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llm_client import NvidiaClient
from observability import RunMetrics
from target_workflow import WorkflowConfig, clear_retrieve_cache
from conductor.profiler import ProfileReport, analyze
from conductor.strategist import Strategist, Optimization, apply_optimization
from conductor.executor import run_eval
from conductor.critic import Verdict, evaluate
from conductor.config_io import config_changes, save_config
from conductor.sandbox import stage_config, load_staged_config
from conductor.nat_adapter import nat_status


@dataclass
class IterationResult:
    iteration: int
    optimization: Optimization
    candidate_metrics: RunMetrics | None
    verdict: Verdict | None
    accepted_config: WorkflowConfig
    config_changes: list[dict[str, str]] = field(default_factory=list)
    skipped_by_user: bool = False
    sandbox_path: Path | None = None


@dataclass
class ConductorResult:
    baseline: RunMetrics
    baseline_config: WorkflowConfig
    profile: ProfileReport | None = None
    iterations: list[IterationResult] = field(default_factory=list)
    saved_configs: list[Path] = field(default_factory=list)
    nat: dict = field(default_factory=nat_status)

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
        use_sandbox: bool = True,
        export_traces_dir: Path | str | None = Path("traces"),
        skip_llm_extras: bool = False,
    ):
        self.baseline_config = baseline_config
        self.eval_set = eval_set
        self.client = client or NvidiaClient()
        self.skip_llm_extras = skip_llm_extras
        self.strategist = Strategist(client=self.client, skip_llm_extras=skip_llm_extras)
        self.progress_cb = progress_cb or (lambda msg: None)
        self.approval_fn = approval_fn
        self.auto_approve = auto_approve
        self.save_dir = Path(save_dir) if save_dir else None
        self.export_traces_dir = Path(export_traces_dir) if export_traces_dir else None
        self.use_sandbox = use_sandbox
        self._baseline_metrics = baseline_metrics
        self._run_baseline = run_baseline

    def run(self) -> ConductorResult:
        clear_retrieve_cache()

        if self._run_baseline or self._baseline_metrics is None:
            self.progress_cb("Running BASELINE…")
            baseline_metrics = run_eval(
                self.baseline_config, self.eval_set, self.client,
                export_traces_dir=self.export_traces_dir,
            )
        else:
            baseline_metrics = self._baseline_metrics

        llm = None if self.skip_llm_extras else self.client
        profile = analyze(
            baseline_metrics.traces,
            self.baseline_config.retrieve_mode,
            self.baseline_config.cache_retrieve,
            client=llm,
        )
        result = ConductorResult(
            baseline=baseline_metrics,
            baseline_config=self.baseline_config,
            profile=profile,
        )

        current_config = self.baseline_config
        current_metrics = baseline_metrics
        tried_names: set[str] = set()
        iteration = 0

        while True:
            opt = self.strategist.propose(profile, current_config, tried_names)
            if opt is None:
                self.progress_cb("No more optimizations to try. Done.")
                break

            tried_names.add(opt.name)
            candidate_config = apply_optimization(current_config, opt)
            changes = config_changes(current_config, candidate_config)
            sandbox_path = None

            approved = self.auto_approve
            if not approved and self.approval_fn is not None:
                approved = self.approval_fn(opt, current_config, candidate_config, changes)

            if not approved:
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

            if self.use_sandbox:
                sandbox_path = stage_config(candidate_config, f"iter{iteration + 1}_{opt.name}")
                candidate_config = load_staged_config(sandbox_path, candidate_config.name)
                self.progress_cb(f"  Sandbox staged → {sandbox_path}")

            candidate_metrics = run_eval(
                candidate_config, self.eval_set, self.client,
                export_traces_dir=self.export_traces_dir,
            )
            verdict = evaluate(current_metrics, candidate_metrics, client=llm)

            iter_result = IterationResult(
                iteration=iteration + 1,
                optimization=opt,
                candidate_metrics=candidate_metrics,
                verdict=verdict,
                accepted_config=candidate_config if verdict.accepted else current_config,
                config_changes=changes,
                sandbox_path=sandbox_path,
            )
            result.iterations.append(iter_result)

            if verdict.accepted:
                current_config = candidate_config
                current_metrics = candidate_metrics
                profile = analyze(
                    current_metrics.traces,
                    current_config.retrieve_mode,
                    current_config.cache_retrieve,
                    client=llm,
                )
                if self.save_dir:
                    out = self.save_dir / f"{candidate_config.name}.yaml"
                    save_config(candidate_config, out)
                    result.saved_configs.append(out)
            iteration += 1

        if result.final_config and self.save_dir:
            final_path = self.save_dir / "final.yaml"
            save_config(result.final_config, final_path)
            if final_path not in result.saved_configs:
                result.saved_configs.append(final_path)

        return result
