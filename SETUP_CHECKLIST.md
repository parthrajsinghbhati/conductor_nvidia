# Conductor — What to give the team, and when

Use this checklist during the hackathon. Only send items when you reach that phase — nothing is needed upfront except Phase 0.

---

## Phase 0 — Right now (before anything else)

**Goal:** Repo runs locally in mock mode.

| # | You provide | How |
|---|---|---|
| 0.1 | Nothing | Clone repo, run `python demo.py --mock --yes` |

**Done when:** Terminal shows baseline → profile → 3 proposals → 2 accepted, 1 rejected → final summary.

---

## Phase 1 — Live NVIDIA API (when you want `--real`)

**Goal:** Demo hits hosted Nemotron on build.nvidia.com.

| # | You provide | When | How |
|---|---|---|---|
| 1.1 | **Working `NVIDIA_API_KEY`** | Before first `--real` run | [build.nvidia.com](https://build.nvidia.com) → Get API Key → paste in `.env` |
| 1.2 | **Confirm model access** | Same time | On build.nvidia.com, open these models and click "Get API Key" / verify access: |
| | | | • `nvidia/nemotron-3-ultra-550b-a55b` (big) |
| | | | • `nvidia/llama-3.1-nemotron-nano-8b-v1` (small) |
| 1.3 | **Screenshot or error text** | If you get 403 | Send the full terminal error — usually means key invalid or model not enabled |

**Test command:**
```bash
python demo.py --real --yes
```
Expect **20–40 minutes** for full run on 10 questions.

**Decision needed from you:**

| Question | Options | Default if no answer |
|---|---|---|
| Live demo for judges? | `--mock` (~10 sec) or `--real` (~30 min) | Use `--mock` for pitch |

---

## Phase 2 — Human approval gate (live demo with interaction)

**Goal:** Show "propose → human approves → apply" story from the hackathon spec.

| # | You provide | When | How |
|---|---|---|---|
| 2.1 | **Nothing** | When practicing demo | Run without `--yes`: `python demo.py --mock` |
| 2.2 | **Demo script preference** | Before pitch | Tell us: press `Y` on opts 1 & 2, `n` on opt 3? (or accept all with `--yes`) |

**Commands:**
```bash
python demo.py --mock          # prompts [Y/n] before each optimization
python demo.py --mock --yes    # skip prompts (CI / fast run)
```

---

## Phase 3 — Web UI for judges (optional)

**Goal:** Browser dashboard instead of terminal only.

| # | You provide | When | How |
|---|---|---|---|
| 3.1 | **Hosting choice** | Before deploy | Pick one: local laptop / Streamlit Cloud / Hugging Face Spaces |
| 3.2 | **`NVIDIA_API_KEY` as secret** | Only if deploying web UI with `--real` | Add secret in Streamlit Cloud or HF Spaces settings |
| 3.3 | **GitHub repo URL** | If using Streamlit Cloud | Push repo to GitHub, connect at share.streamlit.io |

**Run locally:**
```bash
pip install streamlit
streamlit run app.py
```

**Decision needed:**

| Question | Options |
|---|---|
| Deploy web UI? | Yes (send repo URL) / No (terminal demo only) |

---

## Phase 4 — Custom eval set (optional)

**Goal:** Replace trivia with your domain (support tickets, internal docs, etc.).

| # | You provide | When | Format |
|---|---|---|---|
| 4.1 | **6–10 Q&A pairs** | When you want custom domain | Each item: `question` + `key_terms` (words that must appear in a correct answer) |
| 4.2 | **Or approval of draft questions** | If you don't have domain Qs | Reply: "use default trivia" |

**Example entry:**
```json
{
  "id": 11,
  "question": "What API key env var does Conductor use?",
  "key_terms": ["nvidia_api_key", "NVIDIA_API_KEY"]
}
```

File to edit: `eval_set.json`

---

## Phase 5 — GitHub submission

**Goal:** Judges can clone and run.

| # | You provide | When | How |
|---|---|---|---|
| 5.1 | **GitHub repo URL** | Before submission deadline | Push code, add README link |
| 5.2 | **Demo video** (recommended) | Before judging | Record `python demo.py --mock --yes` (~30 sec screen recording) |
| 5.3 | **One-line pitch** | For README / submission form | e.g. "Conductor cuts agent workflow cost 24% with a quality gate that rejects bad optimizations" |

**Do NOT commit:** `.env` (contains API key)

---

## Phase 6 — NVIDIA stack stretch (NAT / NemoClaw / OpenShell)

**Goal:** Closer to full hackathon spec. **Only start if organizers confirm access.**

| # | You provide | When | How |
|---|---|---|---|
| 6.1 | **Organizer email / Slack** | Before Phase 6 | Any message about Brev, NAT, NemoClaw, OpenShell access |
| 6.2 | **GPU credits?** | Same time | Yes/No — needed for Dynamo stretch |
| 6.3 | **Linux VM or Colab** | If Mac NAT install fails | SSH access or shared Colab link |

**Do not start Phase 6 until Phase 0–1 work and you have organizer confirmation.**

---

## Quick reference — commands

| Task | Command | Time |
|---|---|---|
| Fast demo (judges) | `python demo.py --mock --yes` | ~10 sec |
| Interactive demo | `python demo.py --mock` | ~10 sec + prompts |
| Live Nemotron | `python demo.py --real --yes` | ~20–40 min |
| Web UI | `streamlit run app.py` | ~10 sec (mock) |
| Docker | `docker build -t conductor . && docker run --rm conductor` | ~10 sec (mock) |
| CI | Push to GitHub — runs automatically | ~30 sec |

---

## What changed in Package A (already built)

- [x] 10-question eval set
- [x] Nemotron Nano as small model (`nvidia/llama-3.1-nemotron-nano-8b-v1`)
- [x] Config diff table (before → after) in terminal
- [x] Human approval gate (`[Y/n]`, skip with `--yes`)
- [x] Accepted configs saved to `configs/opt*.yaml` and `configs/final.yaml`
- [x] Streamlit web UI (`app.py`)
- [x] This checklist

---

## Send to the team in this order

1. **Now:** "Phase 0 done" or error output if mock fails
2. **When ready for real API:** Phase 1 items (key + model access confirmation)
3. **Before pitch:** Phase 2 decision (mock vs real, auto vs interactive)
4. **If you want web deploy:** Phase 3 (hosting choice + repo URL)
5. **If you want custom domain:** Phase 4 (questions or "use default")
6. **Before deadline:** Phase 5 (repo URL + optional video)
7. **Only if organizers reply:** Phase 6 materials
