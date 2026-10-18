#!/usr/bin/env bash
# Section 6 / 11.1: train and compare logistic regression, XGBoost, and MLP,
# then calibrate the intervention threshold against the validation set.
set -e
python -m src.classifier.train "$@"

