#!/usr/bin/env bash
# Section 8.2 / 11.1: run the offline simulation + feature logging pipeline
# using Groq (or Ollama) and update train/val/test splits incrementally.
set -e
python -m src.pipeline.simulate "$@"

