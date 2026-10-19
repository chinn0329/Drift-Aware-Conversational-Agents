# Drift classifier (Person B)

Fuses the per-turn signals into one calibrated `drift_score` = estimated P(conversation ends in failure),
and a threshold that decides when Person C's intervention fires.

## Run

```bash
pip install -r requirements-classifier.txt
scripts/train_classifier.sh          # Windows: scripts\train_classifier.bat
python -m pytest tests/test_classifier.py -q
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
