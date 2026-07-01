# Conductor

**NVIDIA Hackathon · Track A: Agentic Workflows**

> Agents that make your *other* agents faster and cheaper — and prove it with a before/after number, without breaking correctness.

Conductor is a supervisory multi-agent system — an **SRE for agentic workflows**. It profiles a target pipeline, proposes optimizations (model routing, parallelization), validates each change against a quality gate, and reports measured latency and cost savings while preserving output quality.

---

## Problem

Teams ship multi-agent systems but fly blind on cost and latency. A single request can fan out into dozens of LLM calls. The default is "use the big model everywhere and run everything serially" — which quietly burns money and adds latency.

The hard part is not detecting slowness. It is **fixing it safely**. An optimizer that makes a workflow faster but subtly wrong is worse than useless.

**Conductor** profiles bottlenecks, proposes config changes, validates them against a quality gate, and reverts anything that degrades output — then reports **% latency down, % cost down, quality preserved**.

---

## Solution

Four agents in a closed loop:

| Agent | Job | Model |
|---|---|---|
| **Profiler** | Ingests workflow traces; localizes hot spots (latency, token spend) | Nemotron Ultra (hosted) |
| **Strategist** | Reasons over the profile and proposes optimizations | Nemotron Ultra (hosted) |
| **Executor** | Applies config changes and re-runs the workflow on the eval set | Tool-driven |
| **Critic** | Compares quality + cost + latency vs. baseline; accepts or reverts | Rule-based quality gate |

```
profile → propose → approve → apply → evaluate → accept or revert → repeat
```

**Target workflow:** a deliberately naive 3-step research pipeline — Decompose → Retrieve × N → Synthesize — where every step uses the big model and retrieve runs serially.

**Demo optimizations:**
1. Route **decompose** to a small model → accepted
2. **Parallelize** retrieve calls → accepted
3. Route **synthesize** to a small model → **rejected** (quality regression caught by the gate)

---

## NVIDIA stack

| Component | Role in Conductor |
|---|---|
| **NVIDIA NIM (hosted)** | All LLM inference via `integrate.api.nvidia.com` |
| **Nemotron 3 Ultra** | Baseline pipeline + Strategist reasoning |
| **Nemotron Nano 8B** | Small-model routing target after optimization |
| **NeMo Agent Toolkit (NAT)** | Stretch — custom trace aggregation in MVP |
| **OpenShell** | Stretch — config edits applied in-process for MVP |

Get a free API key at [build.nvidia.com](https://build.nvidia.com).

---

## Quick start (local)

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd nvidia-main

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add NVIDIA_API_KEY from https://build.nvidia.com (optional for mock demo)

python demo.py --mock --yes         # fast demo, auto-approve (~10s, 10 questions)
python demo.py --mock               # interactive — prompts [Y/n] before each change
python demo.py --real --yes         # live Nemotron (~20–40 min)
streamlit run app.py                # web dashboard for judges
```

**Step-by-step checklist of what you need to provide and when:** see [`SETUP_CHECKLIST.md`](SETUP_CHECKLIST.md).

---

## Hosting & deployment

This project ships as a **CLI demo** (Rich terminal) plus an optional **Streamlit web UI**. "Hosting" means making it runnable for judges, teammates, and CI.

### Web UI (judges)

```bash
streamlit run app.py
```

Toggle mock mode in the sidebar, click **Run Conductor**, expand each optimization to see config diffs and verdicts.

### 1. Local (recommended for live pitch)

Best for hackathon presentations. Run on your laptop with the terminal visible to judges.

```bash
source .venv/bin/activate
python demo.py --mock     # reliable, no network — use if API key fails
python demo.py --real     # live NVIDIA endpoints — use if key works
```

Record the terminal with [asciinema](https://asciinema.org/) or QuickTime as a backup demo video.

### 2. GitHub (hackathon submission)

Push the repo and add these in **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `NVIDIA_API_KEY` | Your key from build.nvidia.com (optional — CI uses mock mode) |

Judges clone and follow [Quick start](#quick-start-local). Include a **demo video link** in the repo description or submission form.

### 3. GitHub Codespaces (zero-setup for judges)

Open the repo in a cloud dev environment — no local Python install needed.

1. Push this repo to GitHub.
2. Click **Code → Codespaces → Create codespace on main**.
3. In the terminal:
   ```bash
   pip install -r requirements.txt
   python demo.py --mock
   ```

A `.devcontainer/devcontainer.json` is included for automatic setup.

### 4. Docker (any cloud VM or container platform)

Build and run anywhere Docker is supported (AWS EC2, GCP Compute, Azure VM, Railway, Fly.io, etc.).

```bash
docker build -t conductor .
docker run --rm conductor                          # mock demo (default)
docker run --rm -e NVIDIA_API_KEY=nvapi-xxx conductor python demo.py --real
```

On a cloud VM (e.g. NVIDIA Brev, a $5 Linux droplet):

```bash
ssh user@your-vm
git clone <your-repo-url> && cd nvidia-main
docker build -t conductor .
docker run --rm -e NVIDIA_API_KEY=$NVIDIA_API_KEY conductor python demo.py --real
```

### 5. GitHub Actions (CI smoke test)

Every push/PR runs the full demo in mock mode automatically (see `.github/workflows/ci.yml`). This proves the loop works without spending API credits.

### 6. NVIDIA cloud (real inference)

LLM calls always go to **NVIDIA hosted NIM** — you do not self-host models for the MVP.

| Option | Use case |
|---|---|
| [build.nvidia.com](https://build.nvidia.com) API key | Default — Mac/laptop/cloud VM all call hosted Nemotron |
| NVIDIA Brev / Launchpad | GPU VM if you add stretch features (self-hosted NIM, Dynamo) |
| Google Colab | Alternative if local install is fiddly — copy repo + `pip install` |

**No GPU required** for the current demo. Profiling, routing, parallelization, and the eval gate all run on CPU and call hosted endpoints.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | For `--real` mode | API key from [build.nvidia.com](https://build.nvidia.com) |
| `MOCK_MODE` | No | Set `true` to force simulated responses (default: auto-detect) |

Copy `.env.example` to `.env` for local development. **Never commit `.env`** — it is gitignored.

---

## Why the `openai` Python package?

You are **not calling OpenAI**. NVIDIA NIM exposes an OpenAI-compatible HTTP API. The `openai` SDK is used only as an HTTP client pointed at NVIDIA:

```python
OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)
```

All models are NVIDIA-hosted. No OpenAI API key is used.

---

## Models

| Role | Model ID |
|---|---|
| Big / reasoning | `nvidia/nemotron-3-ultra-550b-a55b` |
| Small / cheap | `nvidia/llama-3.1-nemotron-nano-8b-v1` |

Configure per-step models in `configs/baseline.yaml`. Accepted optimizations are saved to `configs/opt*.yaml` and `configs/final.yaml`. Quality is scored by keyword overlap against `eval_set.json` (10 questions); the Critic rejects changes that drop quality by more than 10 percentage points.

---

## Project structure

```
.
├── demo.py                 # Entry point — Rich terminal demo
├── app.py                  # Streamlit web UI for judges
├── SETUP_CHECKLIST.md      # What to provide and when (phased)
├── llm_client.py           # NVIDIA NIM client + mock fallback
├── target_workflow.py      # 3-step research pipeline
├── observability.py        # Trace dataclasses and metrics
├── eval_set.json           # Scoreable questions with key_terms
├── configs/baseline.yaml   # Naive baseline config
├── Dockerfile              # Container image for cloud/CI
├── .devcontainer/          # GitHub Codespaces config
└── conductor/
    ├── loop.py             # Main optimization loop
    ├── profiler.py         # Bottleneck analysis
    ├── strategist.py       # Optimization proposals
    ├── executor.py         # Eval runner + quality scoring
    ├── critic.py           # Quality gate
    └── config_io.py        # Config diff + save helpers
```

---

## Demo script (for judges)

1. Show the naive baseline — big model on every step, serial retrieve.
2. Run `python demo.py --mock` (or `--real` with a valid key).
3. Watch Conductor profile bottlenecks and propose optimizations.
4. See a **bad** proposal (synthesize → small model) get **rejected** by the quality gate.
5. Final slide: e.g. **−34% latency, −24% cost, quality preserved**.

---

## Success metrics

- % latency reduction on the sample workflow
- % cost / token reduction
- Quality preserved (eval score within tolerance of baseline)
- Number of bad optimizations correctly rejected by the gate

---

## Safety model

- Conductor only operates on a **sample workflow we own**, not production systems.
- Every change is a **config edit** — the original is always kept.
- **Propose → validate → revert** — quality regressions are auto-rejected.
- Stretch: OpenShell sandbox for untrusted config changes.

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | HTTP client for NVIDIA NIM (OpenAI-compatible API) |
| `pyyaml` | Workflow config files |
| `rich` | Terminal demo UI |
| `python-dotenv` | Load env vars from `.env` |

---

## License

MIT
