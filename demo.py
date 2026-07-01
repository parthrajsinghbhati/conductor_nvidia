#!/usr/bin/env python3
"""
Conductor Demo — NVIDIA Hackathon Track A: Agentic Workflows
============================================================
Runs the full optimize loop with rich terminal output.

Usage:
    python demo.py                 # auto-detects NVIDIA_API_KEY; falls back to mock
    python demo.py --mock          # force mock mode (no API key needed)
    python demo.py --real          # force real NVIDIA API (must set NVIDIA_API_KEY)
    python demo.py --yes           # skip human approval prompts (auto-apply proposals)
"""
from __future__ import annotations

import argparse
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

CONSOLE = Console()
EVAL_PATH = Path("eval_set.json")
BASELINE_CFG = Path("configs/baseline.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="Conductor demo")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="Force mock mode")
    g.add_argument("--real", action="store_true", help="Force real NVIDIA API")
    p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-approve all proposals (skip human gate)",
    )
    return p.parse_args()


def _header():
    CONSOLE.print()
    CONSOLE.print(
        Panel.fit(
            "[bold green]CONDUCTOR[/bold green]  ·  "
            "[dim]SRE for Agentic Workflows[/dim]\n"
            "[dim]NVIDIA Hackathon · Track A: Agentic Workflows[/dim]",
            border_style="green",
            padding=(0, 4),
        )
    )
    CONSOLE.print()


def _section(title: str, style: str = "bold cyan"):
    CONSOLE.print()
    CONSOLE.print(Rule(f"[{style}]{title}[/{style}]", style=style))
    CONSOLE.print()


def _metrics_table(title: str, metrics, n_questions: int, style: str = "white") -> Table:
    t = Table(title=title, box=box.ROUNDED, border_style=style, title_style=f"bold {style}")
    t.add_column("Metric", style="dim", width=28)
    t.add_column("Value", justify="right", style=style)
    t.add_row("Avg latency / question", f"{metrics.avg_latency_ms:,.0f} ms")
    t.add_row(f"Total tokens ({n_questions} Qs)", f"{metrics.total_tokens:,}")
    t.add_row(f"Total cost ({n_questions} Qs)", f"${metrics.total_cost_usd:.4f}")
    t.add_row("Quality score", f"{metrics.quality_score:.0%}")
    return t


def _profile_table(profile: ProfileReport) -> Table:
    t = Table(title="Step Profile", box=box.SIMPLE_HEAVY, border_style="cyan")
    t.add_column("Step", style="bold")
    t.add_column("Avg Latency", justify="right")
    t.add_column("Avg Tokens", justify="right")
    t.add_column("% Latency", justify="right")
    t.add_column("% Cost", justify="right")
    t.add_column("Complexity", justify="center")

    complexity_color = {"low": "green", "medium": "yellow", "high": "red"}
    for p in profile.steps:
        color = complexity_color.get(p.complexity, "white")
        t.add_row(
            p.step,
            f"{p.avg_latency_ms:,.0f} ms",
            f"{p.avg_tokens:,.0f}",
            f"{p.pct_latency:.0%}",
            f"{p.pct_cost:.0%}",
            f"[{color}]{p.complexity}[/{color}]",
        )
    return t


def _config_diff_table(changes: list[dict[str, str]]) -> Table:
    t = Table(title="Config changes", box=box.SIMPLE, border_style="yellow")
    t.add_column("Field", style="bold")
    t.add_column("Before", style="red")
    t.add_column("After", style="green")
    for row in changes:
        t.add_row(row["field"], row["before"], row["after"])
    return t


def _iteration_panel(it: IterationResult) -> Panel:
    if it.skipped_by_user:
        return Panel(
            f"[bold]Optimization {it.iteration}:[/bold] {it.optimization.description}\n"
            f"[dim yellow]⊘ SKIPPED — human gate declined to apply this change.[/dim yellow]",
            border_style="yellow",
            padding=(0, 2),
        )

    assert it.verdict is not None
    accepted = it.verdict.accepted
    color = "green" if accepted else "red"
    icon = "✓ ACCEPTED" if accepted else "✗ REJECTED"

    lines = [
        f"[bold]Optimization {it.iteration}:[/bold] {it.optimization.description}",
        f"[dim]{it.optimization.rationale}[/dim]",
        "",
        f"  Latency delta : [bold]{it.verdict.latency_delta_pct:+.1f}%[/bold]",
        f"  Cost delta    : [bold]{it.verdict.cost_delta_pct:+.1f}%[/bold]",
        f"  Quality delta : [bold]{it.verdict.quality_delta:+.2%}[/bold]",
        "",
        f"[bold {color}]{icon}[/bold {color}]  {it.verdict.reason}",
    ]
    if not accepted:
        lines.append("[dim]↩ Reverting to previous config.[/dim]")

    return Panel("\n".join(lines), border_style=color, padding=(0, 2))


def _final_summary(result: ConductorResult):
    lat_imp = result.total_latency_improvement_pct()
    cost_imp = result.total_cost_improvement_pct()
    baseline_q = result.baseline.quality_score
    final_q = (
        result.accepted[-1].candidate_metrics.quality_score
        if result.accepted and result.accepted[-1].candidate_metrics
        else baseline_q
    )

    saved = ""
    if result.saved_configs:
        saved = "\n\n  Saved configs:\n" + "\n".join(
            f"    • {p}" for p in result.saved_configs
        )

    CONSOLE.print()
    CONSOLE.print(
        Panel(
            f"[bold green]CONDUCTOR COMPLETE[/bold green]\n\n"
            f"  Latency  : {result.baseline.avg_latency_ms:,.0f} ms → "
            f"{result.accepted[-1].candidate_metrics.avg_latency_ms if result.accepted and result.accepted[-1].candidate_metrics else result.baseline.avg_latency_ms:,.0f} ms  "
            f"[bold green]({lat_imp:.0f}% faster)[/bold green]\n"
            f"  Cost     : ${result.baseline.total_cost_usd:.4f} → "
            f"${result.accepted[-1].candidate_metrics.total_cost_usd if result.accepted and result.accepted[-1].candidate_metrics else result.baseline.total_cost_usd:.4f}  "
            f"[bold green]({cost_imp:.0f}% cheaper)[/bold green]\n"
            f"  Quality  : {baseline_q:.0%} → {final_q:.0%}  "
            f"[bold green]PRESERVED[/bold green]\n\n"
            f"  Optimizations accepted : [bold]{len(result.accepted)}[/bold]\n"
            f"  Optimizations rejected : [bold red]{len(result.rejected)}[/bold red]\n"
            f"  Skipped by human gate  : [bold yellow]{len(result.skipped)}[/bold yellow]"
            f"{saved}",
            border_style="green",
            padding=(1, 4),
            title="[bold green]Results[/bold green]",
        )
    )


def _prompt_approval(
    opt: Optimization,
    _before: WorkflowConfig,
    _after: WorkflowConfig,
    changes: list[dict[str, str]],
) -> bool:
    CONSOLE.print(f"\n[bold]Proposal {opt.id}:[/bold] {opt.description}")
    CONSOLE.print(f"[dim]Rationale: {opt.rationale}[/dim]")
    for key, val in opt.config_delta.items():
        CONSOLE.print(f"  [yellow]Δ[/yellow] {key}: {val}")
    if changes:
        CONSOLE.print(_config_diff_table(changes))
    while True:
        answer = CONSOLE.input("[bold cyan]Apply this optimization?[/bold cyan] [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        CONSOLE.print("[dim]Please enter Y or n.[/dim]")


def main():
    args = parse_args()

    if args.mock:
        mock = True
    elif args.real:
        mock = False
    else:
        mock = None

    client = NvidiaClient(mock=mock)
    _header()

    mode_str = "[yellow]MOCK[/yellow]" if client.mock else "[green]REAL — NVIDIA NIM[/green]"
    CONSOLE.print(f"Mode: {mode_str}")
    if client.mock:
        CONSOLE.print("[dim]Set NVIDIA_API_KEY in .env or pass --real to use the live API.[/dim]")
    gate_str = "auto-approve" if args.yes else "human approval required"
    CONSOLE.print(f"Gate: [bold]{gate_str}[/bold]")

    eval_set = json.loads(EVAL_PATH.read_text())
    baseline_config = WorkflowConfig.from_yaml(str(BASELINE_CFG))
    baseline_config.name = "baseline"

    CONSOLE.print(f"\nEval set : {len(eval_set)} questions")
    CONSOLE.print(f"Baseline : {baseline_config.decompose.model} on all steps, serial retrieve")

    _section("Step 1 — Baseline Run", "cyan")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as prog:
        task = prog.add_task(f"Running baseline pipeline on {len(eval_set)} questions…", total=None)
        baseline_metrics = run_eval(baseline_config, eval_set, client)
        prog.update(task, completed=True)

    CONSOLE.print(_metrics_table("Baseline Metrics", baseline_metrics, len(eval_set), style="white"))

    _section("Step 2 — Profile", "cyan")
    profile = analyze(baseline_metrics.traces, baseline_config.retrieve_mode)
    CONSOLE.print(_profile_table(profile))
    CONSOLE.print()
    for note in profile.notes:
        CONSOLE.print(f"  [dim]→[/dim] {note}")

    _section("Step 3 — Optimize Loop", "cyan")

    pending_iterations: list[IterationResult] = []

    def on_progress(msg: str):
        if msg.startswith("  ✓") or msg.startswith("  ✗") or "Skipped" in msg:
            CONSOLE.print(f"[dim]{msg}[/dim]")

    loop = ConductorLoop(
        baseline_config,
        eval_set,
        client=client,
        progress_cb=on_progress,
        approval_fn=None if args.yes else _prompt_approval,
        auto_approve=args.yes,
        save_dir=Path("configs"),
        run_baseline=False,
        baseline_metrics=baseline_metrics,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as prog:
        task = prog.add_task("Running optimization loop…", total=None)

        # Run loop without baseline (already done) — display iterations after
        result = loop.run()
        prog.update(task, completed=True)

    for it in result.iterations:
        if not it.skipped_by_user and it.config_changes:
            CONSOLE.print(_config_diff_table(it.config_changes))
        CONSOLE.print(_iteration_panel(it))

    _final_summary(result)


if __name__ == "__main__":
    main()
