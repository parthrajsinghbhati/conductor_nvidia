# Conductor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NVIDIA Hackathon · Track A: Agentic Workflows**

> Agents that make your *other* agents faster and cheaper — and prove it with before/after numbers, without breaking correctness.

Conductor is a supervisory multi-agent system — an **SRE for agentic workflows**. It profiles a 3-step research pipeline, proposes optimizations (model routing, parallelization, caching), validates each change against a quality gate, and reports measured latency and cost savings while preserving output quality.

---

## For judges — run the demo

Pick **one** option below. Mock mode needs no API key and finishes in ~15 seconds. Real mode calls live Nemotron 3 Ultra on NVIDIA NIM (~10–20 min quick, ~30–60 min full).

### Option A — Mock demo in Codespaces (recommended, no API key)

1. Open this repo on GitHub.
2. Click **Code** → **Codespaces** → **Create codespace on main**.
3. Wait for the environment to build (dependencies install automatically).
4. In the terminal:

```bash
python demo.py --mock --yes
# or: bash scripts/run_demo.sh
```

**No NVIDIA API key required.** Codespaces sets `MOCK_MODE=true` automatically.

### Option B — Mock demo on your machine (no API key)

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd nvidia-main

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python demo.py --mock --yes          # ~15 seconds
```

### Option C — Real run with live NVIDIA API (~10–20 min)

Use this if you want to see **actual Nemotron inference** (not simulated). Requires a free API key from [build.nvidia.com](https://build.nvidia.com).

**In Codespaces or locally:**

```bash
cp .env.example .env
# Edit .env — set:  NVIDIA_API_KEY=nvapi-your-key-here

python scripts/validate_key.py       # must show ✅ for both models (~5 sec)
bash scripts/run_real.sh             # quick live run: 3 questions, ~10–20 min
```

**Or run manually:**

```bash
python demo.py --real --quick --yes  # quick: 3 questions (~10–20 min)
python demo.py --real --yes          # full: 10 questions (~30–60 min)
```

| Command | API key | Time | Questions |
|---|---|---|---|
| `python demo.py --mock --yes` | Not needed | ~15 sec | 10 |
| `python demo.py --real --quick --yes` | Required | ~10–20 min | 3 |
| `python demo.py --real --yes` | Required | ~30–60 min | 10 |
| `python demo.py --mock --yes --no-sandbox` | Not needed | ~15 sec | 10 (no sandbox staging) |

**Note:** `--real` overrides Codespaces' default mock mode. You do not need to change `MOCK_MODE` in `.env` if you pass `--real`.

**What `--quick` changes:** uses 3 eval questions and lower `max_tokens` on pipeline steps. Profiler, Strategist, and Critic still run with live LLM calls — only the heavy pipeline eval is shortened.

---

## What you should see

The terminal demo runs four optimization rounds on the 10-question eval set:

| Step | What happens |
|---|---|
| **Baseline** | Naive config — Nemotron 3 Ultra on every step, serial retrieve, no cache |
| **Profile** | Bottleneck table + notes (e.g. decompose is low-complexity) |
| **Optimize loop** | Four proposals; you auto-approve with `--yes` |

**Expected outcome:**

| # | Optimization | Verdict |
|---|---|---|
| 1 | Route **decompose** → small model | ✅ Accepted |
| 2 | **Parallelize** retrieve | ✅ Accepted |
| 3 | **Enable retrieve cache** | ✅ Accepted |
| 4 | Route **synthesize** → small model | ❌ **Rejected** (quality regression) |

**Final summary (approximate):** ~60–70% faster, ~60–70% cheaper, **quality preserved**. Accepted configs are written at runtime to `configs/opt*.yaml` and `configs/final.yaml` (generated artifacts — git-ignored, regenerated on every run).

This rejection is intentional — it shows the **quality gate** working: Conductor will not ship a faster-but-wrong config.

During the run you will also see **sandbox paths** (`traces/sandbox/iter*/candidate.yaml`) — each optimization is staged in an isolated directory before it is applied.

---

## Problem & solution

**Problem:** Multi-agent workflows are expensive and slow. The default is "big model everywhere, run serially" — and optimizers that trade quality for speed are dangerous.

**Solution:** Four agents in a closed loop:

```
profile → propose → approve → sandbox → apply → evaluate → accept or revert → repeat
```

| Agent | Role |
|---|---|
| **Profiler** | Aggregates traces; identifies bottlenecks and complexity per step |
| **Strategist** | Profile-driven optimization proposals (not blind iteration) |
| **Executor** | Runs eval set with sandbox-staged config |
| **Critic** | Quality gate — rejects changes that drop eval score by >10% |

---

## NVIDIA stack

| Component | Role |
|---|---|
| **NVIDIA NIM (hosted)** | All LLM inference via `integrate.api.nvidia.com` |
| **Nemotron 3 Ultra** | Baseline pipeline + Strategist reasoning (real mode) |
| **Llama 3.1 8B Instruct** | Default small model for routing optimizations |
| **NeMo Agent Toolkit (NAT)** | Trace export to `traces/*.json` and `*.csv` (NAT profiler shape); optional full NAT via `pip install -r requirements-nat.txt` |
| **OpenShell-style sandbox** | Each candidate config is staged in `traces/sandbox/` before apply (see below) |

### OpenShell sandbox & NAT — what is integrated

| Feature | Status | Where |
|---|---|---|
| **Sandbox staging** | ✅ Working | `conductor/sandbox.py` — writes `candidate.yaml` to `traces/sandbox/iter{N}_{opt}/` before each optimization is applied |
| **NAT trace export** | ✅ Working | `conductor/nat_adapter.py` — exports `traces/{run}_profiler_traces.json` and `*_standardized_data.csv` after every eval pass |
| **Full OpenShell runtime** | Not integrated | Would require OpenShell SDK access; we use the same *staging pattern* (isolate → review → apply) |
| **Full NAT library** | Optional | `pip install -r requirements-nat.txt` — demo works without it |

To verify sandbox after a run:

```bash
ls traces/sandbox/*/candidate.yaml
ls traces/baseline_profiler_traces.json
```

Use `--no-sandbox` to skip staging (not recommended for demo).

---

## Project structure

```
.
├── demo.py                 # Entry point — Rich terminal demo
├── scripts/
│   ├── run_demo.sh         # One-command mock demo (judges)
│   ├── run_real.sh         # One-command live API demo (needs key)
│   └── validate_key.py     # Test NVIDIA API key
├── llm_client.py           # NVIDIA NIM client + mock fallback
├── target_workflow.py      # 3-step research pipeline (decompose → retrieve → synthesize)
├── observability.py        # Trace dataclasses and metrics
├── eval_set.json           # 10 scoreable questions with key_terms
├── configs/baseline.yaml   # Naive baseline config
├── .devcontainer/          # GitHub Codespaces auto-setup
└── conductor/
    ├── __init__.py          # Package marker
    ├── loop.py              # Main optimization loop
    ├── profiler.py          # Bottleneck analysis
    ├── strategist.py        # Optimization proposals
    ├── executor.py          # Eval runner + quality scoring
    ├── critic.py            # Quality gate
    ├── config_io.py         # Config diff + persistence helpers
    ├── sandbox.py           # Config staging (OpenShell-style)
    └── nat_adapter.py       # NAT-compatible trace export
```

---

## How quality is measured

Each question in `eval_set.json` has `key_terms` (e.g. `"au"`, `"79"` for gold). The synthesize step is scored by keyword overlap. The Critic rejects any optimization that drops the aggregate score by more than 10 percentage points vs baseline.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | For `--real` only | API key from [build.nvidia.com](https://build.nvidia.com) |
| `MOCK_MODE` | No | Set `true` to force simulated responses (Codespaces default) |
| `SMALL_MODEL` | No | Override small model (default: `meta/llama-3.1-8b-instruct`) |

Copy `.env.example` to `.env` for local development. **Never commit `.env`.**

---

## Why the `openai` Python package?

You are **not calling OpenAI**. NVIDIA NIM exposes an OpenAI-compatible HTTP API. The `openai` SDK is used only as an HTTP client:

```python
OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)
```

All models are NVIDIA-hosted. No OpenAI API key is used.

---

## CI

Every push runs `python demo.py --mock --yes` in GitHub Actions (`.github/workflows/ci.yml`) — no API key or GPU required.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Demo runs but you want live API | Set `NVIDIA_API_KEY` in `.env`, run `python scripts/validate_key.py` |
| `403` on real mode | Regenerate key at build.nvidia.com |
| Transient API error mid-run (timeout / rate limit / 5xx) | Retried automatically up to 3× with backoff; a missing/invalid key still fails fast so it surfaces immediately |
| Real mode very slow | Use `--quick` flag or stick to `--mock` for evaluation |
| Nemotron Nano hangs | Do not set `SMALL_MODEL` to Nano in `.env`; use default Llama 8B |

---

## License

MIT License — see [LICENSE](LICENSE).
