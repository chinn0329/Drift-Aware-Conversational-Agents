# Drift classifier (Person B)

Fuses the per-turn signals into one calibrated `drift_score` = estimated P(conversation ends in failure),
and a threshold that decides when Person C's intervention fires.

## Run

```bash
pip install -r requirements-classifier.txt
scripts/train_classifier.sh          # Windows: scripts\train_classifier.bat
pytest tests/test_classifier.py -q
```

Equivalent pieces: `python -m src.classifier.data` (data report) -> `python -m src.classifier.train` ->
`python -m src.classifier.evaluate` (ablations).

## When Person A delivers more data

Nothing changes except the files in `data/`. Either replace the split CSVs (`data.source: splits`) or drop
new `*.jsonl` files into `data/processed/` and run with `--source raw_logs` (existing conversations keep their
split because the split is a hash of `conversation_id`). Then rerun the script. The pipeline adapts on its own:

| Setting | Rule |
|---|---|
| Evaluation mode | `auto`: grouped repeated CV while val/test are tiny; switches to train/val/test once each holds >=100 conversations and >=20 of the rarer outcome |
| Hyperparameter search | `auto`: off below 150 conversations, grid search above |
| Model choice | simplest model within 0.01 AUROC of the best (within-group AUROC) |
| History | every run appended to `reports/classifier/runs_history.csv` |

## How the data is interpreted (multi-file, multi-model logs)

| Term | Meaning |
|---|---|
| **task** | the benchmark item (`conversation_id` in Person A's files). Cross-validation and bootstrap GROUP by task, so every attempt of a task stays on one side of a split. |
| **run** | one attempt = task + `assistant_model` + `run_index`. This is the "conversation" for labels, conversation-level metrics and the per-run consistency check. Rows without model metadata are the model `legacy`, run 0. |

The same task run by two models with different outcomes is **two runs, not a label conflict**.

Loading steps (all reported at the top of every run, all switchable in `data.*` of the config):
1. every `*.jsonl` in `data/processed/` is read; mistyped column names (whitespace) are folded into the real column;
2. the same (run, turn) present in several files is kept once (rows with metadata win, then the newest file);
3. runs flagged `label_valid == 0`, truncated, or in `exclude_label_reasons` / not in `extractor_versions` are dropped whole;
4. a run whose rows disagree on the outcome stops the pipeline (that would be a real data error).

## Extra evaluation outputs (reports/classifier/<run>/ablations/)

| File | Question it answers |
|---|---|
| `incremental_over_source.csv`, `incremental_gain.csv` | Do the signals add anything beyond task source? Paired CI on the gain. |
| `signal_by_source.csv` | Raw AUROC of each signal inside each source, with CI and a check of the expected direction (audit table for Person A). |
| `within_task_auroc.csv`, `within_task_paired.csv` | Same task, different outcome: does the score (or each signal) separate them? |
| `leave_one_model_out.csv` | Does it transfer to an unseen model? Strict (unseen tasks) and lenient (seen tasks) protocols. |
| `leave_one_source_out.csv` | Does it transfer to an unseen task source? |

## Trajectory features (optional, off by default)

`data.trajectory_features: true` adds, for every signal, its running max, running mean and change since turn 1
(`<signal>__cummax`, `__cummean`, `__dfirst`). They use only the current and past turns, so they can be built live.
Why: a single turn's value is noisy (self-consistency rests on 2 resamples) and drift is a trajectory.

A model trained this way needs per-conversation history at runtime. Person C keeps one tracker per conversation:

```python
from src.classifier.predict import TrajectoryTracker, predict_drift_score, explain
tracker = TrajectoryTracker()                    # pass-through if the model has no trajectory features
feats = tracker.update(turn_features)            # adds the derived columns
score = predict_drift_score(feats)
tracker.truncate(k)                              # after a ROLLBACK to turn k: forget the discarded turns
```
A raw feature dict is refused by a trajectory model (it raises) rather than silently scored worse.

## Threshold strategies

| strategy | picks the threshold that... |
|---|---|
| `youden` (default) | maximises recall - false-positive rate (turn level) |
| `max_false_alarm` | catches the most failing runs EARLY, while interrupting at most `max_false_alarm` of succeeding runs |
| `target_recall`, `max_flag_rate`, `fbeta` | see config |

"Early" = flagged before the run's last turn; an intervention needs lead time. The sweep CSV and the training table
report `conv_recall_early` and `conv_false_alarm_early` next to the any-turn versions.

## Contract for the rest of the team (unchanged by retraining)

`predict_drift_score(dict) -> float in [0,1]`, `should_intervene`, `explain` (dominant signal in
`self_consistency | contradiction | confidence | silent_commitment`, or None), `get_threshold()`, and
`models/latest.json`. Retraining changes the scores, the threshold and which signal dominates, so callers must read
the threshold from `get_threshold()` and never hard-code it. Features at runtime must be computed exactly as in
training (same extractor version, same number of resamples), otherwise the model is off-distribution.

## Output layout

```
models/latest.json                    pointer to the current run + threshold  (what Person C loads)
models/<run_id>/primary.joblib        model + calibrator + threshold + metadata
models/<run_id>/{logreg,xgboost,mlp}.joblib
reports/classifier/<run_id>/          model_comparison.csv, threshold_sweep*.csv/png, reliability.png, roc.png,
                                      auroc_by_turn.csv, logreg_coefficients.csv, official_split_check.csv,
                                      data_report.json, ablations/
```

## Interface for Person C

```python
from src.classifier.predict import predict_drift_score, should_intervene, explain, get_threshold

predict_drift_score(features)   # float in [0,1]   (features: dict | list in contract order | pandas row)
should_intervene(features)      # score >= calibrated threshold
explain(features)               # {"drift_score", "threshold", "intervene", "dominant_signal", "contributions"}
```

`dominant_signal` is one of `self_consistency | contradiction | confidence | silent_commitment` (groups in the
config). It is a model-agnostic occlusion: each group is replaced by its median on successful conversations and the
score drop is measured, so it works for every model class.

Commit `models/latest.json` and the current run's `primary.joblib` so teammates can use the model without retraining
(add `models/*/` other runs and `reports/` to `.gitignore`).

## Key design choices

* **Label**: `y = 1` means the conversation *failed*. Every turn inherits its conversation's outcome (plan Sec. 8.2).
* **Splits by conversation**, never by row; leakage across splits raises an error.
* **Probabilities are calibrated** (Platt by default) on out-of-fold scores, then a threshold is swept: turn-level
  precision/recall/flag-rate and conversation-level catch-rate vs. false-alarm rate.
* **Threshold strategy is Youden's J**, not F1: with ~80% of conversations failing, maximising F1 just flags everything.
* **Selection uses within-group AUROC** so a task source in which everything fails cannot inflate the score.
