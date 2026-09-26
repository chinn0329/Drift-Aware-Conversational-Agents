#!/usr/bin/env bash
# Section 8.2 / 11.1: run the offline simulation + feature logging pipeline
# across all sharded instructions, all chosen backbones, multiple runs each.
set -e
python -m src.pipeline.simulate --config configs/config.yaml
python -m src.pipeline.dataset_builder --config configs/config.yaml
