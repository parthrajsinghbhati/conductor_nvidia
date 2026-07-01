"""
Conductor Web UI — Streamlit dashboard for judges.

Usage:
    pip install streamlit
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from llm_client import NvidiaClient, BIG_MODEL, SMALL_MODEL
from target_workflow import WorkflowConfig
from conductor.loop import ConductorLoop
from conductor.executor import run_eval
from conductor.profiler import analyze
from conductor.config_io import config_changes

EVAL_PATH = Path("eval_set.json")
BASELINE_CFG = Path("configs/baseline.yaml")


st.set_page_config(page_title="Conductor", page_icon="🎼", layout="wide")
st.title("Conductor — SRE for Agentic Workflows")
st.caption("NVIDIA Hackathon · Track A: Agentic Workflows")

mock_mode = st.sidebar.checkbox("Mock mode (no API key)", value=True)
auto_approve = st.sidebar.checkbox("Auto-approve proposals", value=True)

if st.sidebar.button("Run Conductor", type="primary"):
    client = NvidiaClient(mock=mock_mode)
    eval_set = json.loads(EVAL_PATH.read_text())
    baseline_config = WorkflowConfig.from_yaml(str(BASELINE_CFG))
    baseline_config.name = "baseline"

    st.subheader("Baseline")
    with st.spinner("Running baseline…"):
        baseline_metrics = run_eval(baseline_config, eval_set, client)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg latency", f"{baseline_metrics.avg_latency_ms:,.0f} ms")
    c2.metric("Total cost", f"${baseline_metrics.total_cost_usd:.4f}")
    c3.metric("Total tokens", f"{baseline_metrics.total_tokens:,}")
    c4.metric("Quality", f"{baseline_metrics.quality_score:.0%}")

    profile = analyze(baseline_metrics.traces, baseline_config.retrieve_mode)
    st.subheader("Profile")
    st.dataframe(
        [
            {
                "step": p.step,
                "avg_latency_ms": round(p.avg_latency_ms, 1),
                "avg_tokens": round(p.avg_tokens, 1),
                "pct_latency": f"{p.pct_latency:.0%}",
                "complexity": p.complexity,
            }
            for p in profile.steps
        ],
        use_container_width=True,
    )
    for note in profile.notes:
        st.info(note)

    st.subheader("Optimization loop")
    loop = ConductorLoop(
        baseline_config,
        eval_set,
        client=client,
        auto_approve=auto_approve,
        save_dir=Path("configs"),
        run_baseline=False,
        baseline_metrics=baseline_metrics,
    )
    with st.spinner("Optimizing…"):
        result = loop.run()

    for it in result.iterations:
        with st.expander(
            f"{'✅' if it.verdict and it.verdict.accepted else '❌' if it.verdict else '⊘'} "
            f"Opt {it.iteration}: {it.optimization.description}",
            expanded=True,
        ):
            st.write(it.optimization.rationale)
            if it.config_changes:
                st.table(it.config_changes)
            if it.skipped_by_user:
                st.warning("Skipped — human gate declined.")
            elif it.verdict:
                st.write(it.verdict.reason)
                if it.candidate_metrics:
                    st.metric("Quality after", f"{it.candidate_metrics.quality_score:.0%}")

    st.subheader("Results")
    st.success(
        f"Latency: {result.total_latency_improvement_pct():.0f}% faster · "
        f"Cost: {result.total_cost_improvement_pct():.0f}% cheaper · "
        f"Accepted: {len(result.accepted)} · Rejected: {len(result.rejected)}"
    )
    if result.saved_configs:
        st.write("Saved configs:", [str(p) for p in result.saved_configs])

else:
    st.markdown("""
    ### How to use
    1. Enable **Mock mode** for a fast demo (~10 sec), or disable it and set `NVIDIA_API_KEY` in `.env`.
    2. Click **Run Conductor** in the sidebar.
    3. Expand each optimization to see **config changes**, verdict, and quality delta.

    ### Models
    - **Big:** `{big}`
    - **Small:** `{small}`
    """.format(big=BIG_MODEL, small=SMALL_MODEL))
