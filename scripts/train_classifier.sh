#!/usr/bin/env bash
# Retrain the drift classifier on whatever is currently in data/ and regenerate all reports.
# Extra args are passed to train.py, e.g.:  scripts/train_classifier.sh --models logreg xgboost
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.classifier.data      # data-quality report first: read the warnings
python -m src.classifier.train "$@"
python -m src.classifier.evaluate  # baselines, signal ablation, learning curve, AUROC by turn
