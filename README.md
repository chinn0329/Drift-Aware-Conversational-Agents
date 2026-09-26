# Drift-Aware Conversational Agents

## Project Description

Large language models perform reliably on single-turn instructions but degrade sharply in
realistic multi-turn conversations where requirements are revealed progressively — the way
people actually talk to AI assistants. Research (Laban et al., ICLR 2026 Best Paper) shows an
average 39% performance drop in these settings: once a model locks onto an early, incorrect
assumption, it rarely revisits it as later turns supply the missing information.

This project builds a lightweight, real-time **drift detector** that sits around any base LLM
(without retraining or fine-tuning it) to catch the exact turn where a model commits to a wrong
assumption, and to intervene before the conversation is unrecoverable. It has three layers:

- **Signal layer** — four lightweight per-turn measurements of conversation health (self-consistency
  divergence, assumption-contradiction tracking, verbalized-confidence trend, commitment-to-
  underspecification ratio).
- **Decision layer** — a small trained classifier (logistic regression / XGBoost / MLP, compared
  side by side) that fuses the four signals into one calibrated `drift_score ∈ [0, 1]`.
- **Action layer** — one of three interventions (re-grounding, rollback, targeted clarification),
  triggered only when the drift score crosses a calibrated threshold.

The system is evaluated against a vanilla multi-turn baseline and MT-OSC (a fixed-schedule
context-condensation method) using the open-source "Lost in Conversation" benchmark, with the
training data generated automatically from the benchmark's own pass/fail verdicts — scaled across
multiple model backbones and repeated runs to produce a defensible, non-trivial evidence base.


## Layout

```
drift-aware-agents/
├── configs/                # YAML configs: model choices, thresholds, paths (no secrets committed)
├── data/
│   ├── raw/                 # Cloned "Lost in Conversation" shards + any extended benchmarks (Sec 8.3)
│   ├── processed/           # Simulated conversation transcripts + per-turn feature logs (Sec 8.2)
│   └── splits/              # train/val/held-out-task-family splits, split by conversation ID (Sec 8.2)
├── src/
│   ├── signals/             # The four drift signals (Section 5)
│   │   ├── self_consistency.py      # 5.1 resampling + embedding variance
│   │   ├── contradiction.py         # 5.2 NLI-based assumption-contradiction tracking
│   │   ├── confidence.py            # 5.3 verbalized-confidence trend
│   │   ├── commitment_ratio.py      # 5.4 silent-commitment-to-underspecification ratio
│   │   └── feature_fusion.py        # 5.5 assembles features_t vector
│   ├── classifier/           # The Decision Layer (Section 6)
│   │   ├── models.py         # logistic regression / XGBoost / MLP definitions
│   │   ├── train.py          # training entrypoint, model comparison
│   │   └── calibration.py    # drift_score calibration + threshold sweep (Sec 7)
│   ├── intervention/         # The Action Layer (Section 7)
│   │   ├── policy.py         # threshold gate + dominant-signal routing
│   │   └── strategies.py     # re-grounding, rollback, targeted clarification
│   ├── pipeline/             # Offline dataset-generation workflow (Section 8, 11.1)
│   │   ├── backends.py       # LLM backend wrappers (Claude/GPT API, Llama/Qwen local)
│   │   ├── simulate.py       # runs the benchmark harness turn-by-turn, logs features
│   │   └── dataset_builder.py# back-labels turns, flattens into training table
│   ├── orchestrator/         # Live/runtime workflow + backend API (Section 10.1, 11.2)
│   │   ├── session.py        # runs baseline + drift-aware agent in parallel
│   │   └── api.py            # FastAPI endpoints for the demo frontend
│   └── utils/                 # logging, IO, shared helpers
├── frontend/                 # Split-screen demo UI (Section 10.2)
│   └── src/components/       # ChatPanel, DriftSparkline, FeaturePanel, AmbiguityInjector
├── notebooks/                # exploratory work — baseline repro, signal exploration, ablations
├── tests/                    # unit tests for signals, classifier, intervention logic
├── scripts/                  # shell entrypoints (run_baseline.sh, generate_dataset.sh, train_classifier.sh)
├── docs/                     # execution plan + any design notes
└── reports/                  # figures and write-up material for the final paper (Section 15)
```


## Section → code map

| Plan section | Code location |
|---|---|
| 5. Drift Detection signals | `src/signals/` |
| 6. Drift Classifier | `src/classifier/` |
| 7. Intervention Logic | `src/intervention/` |
| 8. Dataset Generation | `src/pipeline/`, `data/` |
| 10.1 Backend | `src/orchestrator/` |
| 10.2 Frontend | `frontend/` |
| 11.1 Offline workflow | `scripts/generate_dataset.sh` → `src/pipeline/` |
| 11.2 Live/runtime workflow | `src/orchestrator/session.py` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

## Week 1–2 goal (per timeline)

Clone the "Lost in Conversation" framework into `data/raw/`, reproduce the paper's baseline
degradation number unmodified, and confirm the harness is wired correctly before any drift
signal is added.

## Work Division (3 people)

Two shared contracts are agreed on **before** splitting up, so each person can build and test
independently without waiting on the others:

1. **Feature schema** — the 5-element vector from Section 5.5 (`self_consistency_variance`,
   `contradiction_score`, `confidence`, `confidence_delta`, `silent_commitment_ratio`), plus a
   fixed row format (`conversation_id`, `turn_index`, `eventual_success`).
2. **Drift-score interface** — a single function signature: `predict_drift_score(features) -> float in [0,1]`.

### Person A — Signals & Dataset Generation
**Owns:** `src/signals/`, `src/pipeline/`, `data/`
- Clone the benchmark, reproduce the baseline (Weeks 1–2)
- Implement the four drift signals independently (Weeks 3–4)
- Build the simulation + back-labeling pipeline, scale across models/runs (Weeks 5–6)
- **Delivers:** the labeled turn-level feature table, in the agreed schema
- Fully independent — needs only the Week 1 schema agreement, nothing from B or C

### Person B — Classifier & Evaluation
**Owns:** `src/classifier/`, evaluation/ablation work (Section 9)
- Starts immediately on synthetic data matching the schema — no need to wait for A
- Build and compare logistic regression / XGBoost / MLP (Weeks 5–7), swap in A's real
  dataset once it lands (~Week 6)
- Calibrate threshold, sweep precision/recall (Week 8)
- Run ablations and generalization tests (Week 11)
- **Delivers:** a `predict_drift_score()` implementation matching the interface
- Only touchpoint with A: a data swap, not a code dependency

### Person C — Intervention, Orchestrator & Demo
**Owns:** `src/intervention/`, `src/orchestrator/`, `frontend/`
- Starts immediately using a stub classifier that returns rule-based scores matching the
  interface — no need to wait for B
- Build the three intervention strategies + policy routing (Week 8, against the stub)
- Build the FastAPI orchestrator running baseline vs. drift-aware sessions in parallel (Week 9)
- Build the split-screen React frontend, sparkline, feature panel, ambiguity-injection control (Week 12)
- **Delivers:** the working demo
- Only touchpoint with B: swapping the stub for the real classifier (~Week 9) — a drop-in
  swap since both coded to the same function signature

### Integration checkpoints (the only unavoidable joint work)
- **Week 1:** agree on the feature schema and drift-score interface
- **Week 9–10:** wire A's real dataset, B's real classifier, and C's real orchestrator/demo
  together for the first time
- **Weeks 10–13:** final eval, ablations, and demo polish — splittable further (B owns eval
  numbers, C owns demo polish, A can help extend data if generalization tests need more)
