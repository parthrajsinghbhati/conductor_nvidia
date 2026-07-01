#!/usr/bin/env python3
"""
Conductor Demo — NVIDIA Hackathon Track A: Agentic Workflows
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from llm_client import NvidiaClient
from target_workflow import WorkflowConfig
from conductor.loop import ConductorLoop, ConductorResult, IterationResult
from conductor.profiler import ProfileReport, analyze
from conductor.executor import run_eval
from conductor.strategist import Optimization
from conductor.nat_adapter import nat_status

CONSOLE = Console(width=120)
EVAL_PATH = Path("eval_set.json")
BASELINE_CFG = Path("configs/baseline.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="Conductor demo")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="Force mock mode")
    g.add_argument("--real", action="store_true", help="Force real NVIDIA API")
    p.add_argument("--yes", "-y", action="store_true", help="Auto-approve all proposals")
    p.add_argument("--no-sandbox", action="store_true", help="Skip sandbox staging")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Faster live run: 3 questions, lower max_tokens, skip extra LLM summaries",
    )
    p.add_argument(
        "--questions",
        type=int,
        default=None,
        metavar="N",
        help="Use first N eval questions (overridden by --quick)",
    )
    return p.parse_args()


def _shrink_tokens(cfg: WorkflowConfig) -> WorkflowConfig:
    """Lower max_tokens for faster live API calls (quality gate still applies)."""
    c = copy.deepcopy(cfg)
    c.decompose.max_tokens = 64
    c.retrieve.max_tokens = 96
    c.synthesize.max_tokens = 128
    return c


def _header():
    CONSOLE.print()
    CONSOLE.print(
        Panel.fit(
            "[bold green]CONDUCTOR[/bold green]  ·  [dim]SRE for Agentic Workflows[/dim]\n"
            "[dim]NVIDIA Hackathon · Track A: Agentic Workflows[/dim]",
            border_style="green",
            padding=(0, 4),
        )
    )


def _section(title: str, style: str = "bold cyan"):
    CONSOLE.print()
    CONSOLE.print(Rule(f"[{style}]{title}[/{style}]", style=style))


def _metrics_table(title: str, metrics, n_questions: int, style: str = "white") -> Table:
    t = Table(title=title, box=box.ROUNDED, border_style=style, title_style=f"bold {style}")
    t.add_column("Metric", style="dim", width=28)
    t.add_column("Value", justify="right", style=style)
    t.add_row("Avg latency / question", f"{metrics.avg_latency_ms:,.0f} ms")
    t.add_row(f"Total tokens ({n_questions} Qs)", f"{metrics.total_tokens:,}")
    t.add_row(f"Total cost ({n_questions} Qs)", f"${metrics.total_cost_usd:.4f}")
    t.add_row("Quality score", f"{metrics.quality_score:.0%}")
    if metrics.total_cache_hits:
        t.add_row("Retrieve cache hits", str(metrics.total_cache_hits))
    return t


def _profile_table(profile: ProfileReport) -> Table:
    t = Table(title="Step Profile", box=box.SIMPLE_HEAVY, border_style="cyan")
    t.add_column("Step", style="bold", width=12)
    t.add_column("Avg Latency", justify="right", width=12)
    t.add_column("Avg Tokens", justify="right", width=10)
    t.add_column("% Latency", justify="right", width=10)
    t.add_column("Complexity", justify="center", width=10)
    t.add_column("Cache hits", justify="right", width=10)

    complexity_color = {"low": "green", "medium": "yellow", "high": "red"}
    for p in profile.steps:
        color = complexity_color.get(p.complexity, "white")
        t.add_row(
            p.step,
            f"{p.avg_latency_ms:,.0f} ms",
            f"{p.avg_tokens:,.0f}",
            f"{p.pct_latency:.0%}",
            f"[{color}]{p.complexity}[/{color}]",
            str(p.cache_hits),
        )
    return t


def _config_diff_table(changes: list[dict[str, str]]) -> Table:
    t = Table(title="Config changes", box=box.SIMPLE, border_style="yellow", expand=True)
    t.add_column("Field", style="bold", width=28, no_wrap=True)
    t.add_column("Before", style="red", overflow="fold")
    t.add_column("After", style="green", overflow="fold")
    for row in changes:
        t.add_row(row["field"], row["before"], row["after"])
    return t


def _iteration_panel(it: IterationResult) -> Panel:
    if it.skipped_by_user:
        return Panel(
            f"[bold]Optimization {it.iteration}:[/bold] {it.optimization.description}\n"
            "[dim yellow]⊘ SKIPPED — human gate declined.[/dim yellow]",
            border_style="yellow",
        )

    assert it.verdict is not None
    color = "green" if it.verdict.accepted else "red"
    icon = "✓ ACCEPTED" if it.verdict.accepted else "✗ REJECTED"
    sandbox = f"\n[dim]Sandbox: {it.sandbox_path}[/dim]" if it.sandbox_path else ""

    lines = [
        f"[bold]Optimization {it.iteration}:[/bold] {it.optimization.description}",
        f"[dim]{it.optimization.rationale}[/dim]",
        "",
        f"  Latency delta : [bold]{it.verdict.latency_delta_pct:+.1f}%[/bold]",
        f"  Cost delta    : [bold]{it.verdict.cost_delta_pct:+.1f}%[/bold]",
        f"  Quality delta : [bold]{it.verdict.quality_delta:+.2%}[/bold]",
        "",
        f"[bold {color}]{icon}[/bold {color}]  {it.verdict.reason}",
        f"[dim italic]{it.verdict.llm_explanation}[/dim italic]",
    ]
    if not it.verdict.accepted:
        lines.append("[dim]↩ Reverting to previous config.[/dim]")
    lines.append(sandbox)
    return Panel("\n".join(l for l in lines if l), border_style=color, padding=(0, 2))


def _final_summary(result: ConductorResult):
    baseline_q = result.baseline.quality_score
    final_q = (
        result.accepted[-1].candidate_metrics.quality_score
        if result.accepted and result.accepted[-1].candidate_metrics
        else baseline_q
    )
    nat = result.nat
    nat_line = (
        f"NAT: [green]integrated v{nat['version']}[/green]"
        if nat["available"] else
        "NAT: [yellow]compatible trace export[/yellow] → traces/"
    )

    saved = ""
    if result.saved_configs:
        saved = "\n  Saved: " + ", ".join(str(p) for p in result.saved_configs)

    CONSOLE.print()
    CONSOLE.print(
        Panel(
            f"[bold green]CONDUCTOR COMPLETE[/bold green]\n\n"
            f"  Latency  : {result.baseline.avg_latency_ms:,.0f} ms → "
            f"{result.accepted[-1].candidate_metrics.avg_latency_ms if result.accepted and result.accepted[-1].candidate_metrics else result.baseline.avg_latency_ms:,.0f} ms  "
            f"({result.total_latency_improvement_pct():.0f}% faster)\n"
            f"  Cost     : ${result.baseline.total_cost_usd:.4f} → "
            f"${result.accepted[-1].candidate_metrics.total_cost_usd if result.accepted and result.accepted[-1].candidate_metrics else result.baseline.total_cost_usd:.4f}  "
            f"({result.total_cost_improvement_pct():.0f}% cheaper)\n"
            f"  Quality  : {baseline_q:.0%} → {final_q:.0%}  PRESERVED\n\n"
            f"  Accepted: {len(result.accepted)}  Rejected: {len(result.rejected)}  "
            f"Skipped: {len(result.skipped)}\n"
            f"  {nat_line}{saved}",
            border_style="green",
            title="Results",
            padding=(1, 2),
        )
    )


def _prompt_approval(opt, _before, _after, changes) -> bool:
    CONSOLE.print(f"\n[bold]Proposal:[/bold] {opt.description}")
    CONSOLE.print(f"[dim]{opt.rationale}[/dim]")
    if changes:
        CONSOLE.print(_config_diff_table(changes))
    while True:
        answer = CONSOLE.input("[cyan]Apply this optimization?[/cyan] [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def main():
    args = parse_args()
    mock = True if args.mock else False if args.real else None
    client = NvidiaClient(mock=mock)

    _header()
    CONSOLE.print(f"Mode: {'[yellow]MOCK[/yellow]' if client.mock else '[green]REAL[/green]'}")
    CONSOLE.print(f"Gate: {'auto-approve' if args.yes else 'human approval'}")
    CONSOLE.print(f"Sandbox: {'off' if args.no_sandbox else 'on'}")

    eval_set = json.loads(EVAL_PATH.read_text())
    if args.quick:
        eval_set = eval_set[:3]
    elif args.questions is not None:
        eval_set = eval_set[: max(args.questions, 1)]

    baseline_config = WorkflowConfig.from_yaml(str(BASELINE_CFG))
    baseline_config.name = "baseline"

    skip_llm_extras = args.quick and not client.mock
    if skip_llm_extras:
        baseline_config = _shrink_tokens(baseline_config)
        CONSOLE.print(
            "\n[yellow]Quick live mode[/yellow]: 3 questions, reduced max_tokens, "
            "no Strategist/Profiler/Critic LLM calls"
        )
        CONSOLE.print("[dim]Estimated time: ~8–15 min (full --real is ~30–60 min)[/dim]")

    CONSOLE.print(f"\nEval set: {len(eval_set)} questions")

    _section("Step 1 — Baseline")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=CONSOLE) as prog:
        task = prog.add_task(f"Running baseline on {len(eval_set)} questions…", total=None)
        baseline_metrics = run_eval(baseline_config, eval_set, client)
        prog.update(task, completed=True)
    CONSOLE.print(_metrics_table("Baseline", baseline_metrics, len(eval_set)))

    _section("Step 2 — Profile")
    profile = analyze(
        baseline_metrics.traces, baseline_config.retrieve_mode,
        baseline_config.cache_retrieve,
        client=None if skip_llm_extras else client,
    )
    CONSOLE.print(_profile_table(profile))
    CONSOLE.print(f"\n[bold]Profiler summary:[/bold] {profile.llm_summary}")
    for note in profile.notes:
        CONSOLE.print(f"  → {note}")

    _section("Step 3 — Optimize Loop")
    loop = ConductorLoop(
        baseline_config, eval_set, client=client,
        approval_fn=None if args.yes else _prompt_approval,
        auto_approve=args.yes,
        run_baseline=False,
        baseline_metrics=baseline_metrics,
        use_sandbox=not args.no_sandbox,
        skip_llm_extras=skip_llm_extras,
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=CONSOLE) as prog:
        task = prog.add_task("Optimization loop…", total=None)
        result = loop.run()
        prog.update(task, completed=True)

    for it in result.iterations:
        if it.config_changes:
            CONSOLE.print(_config_diff_table(it.config_changes))
        CONSOLE.print(_iteration_panel(it))

    _final_summary(result)


if __name__ == "__main__":
    main()
