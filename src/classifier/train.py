"""
Section 6 / 11.1 (steps 8-9) — Train and compare the drift classifiers.

Loads the flattened turn-level table (features + eventual_success) from
data/splits/, trains logistic regression, XGBoost, and MLP, and reports
accuracy/AUROC/training time/interpretability side by side per Section 9.3
(ablation: classifier comparison).

Selects the best-justified model — not necessarily the highest-scoring one;
interpretability and overhead matter too (Section 11.1, step 8).
"""


def load_splits(splits_dir: str):
    """Load train/val/held-out-task-family splits (split by conversation ID)."""
    raise NotImplementedError


def train_and_compare(train_df, val_df):
    """Train all three model classes and return a comparison table:
    accuracy, AUROC, training_time, interpretability_notes.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("Wire up argparse + call train_and_compare once data/splits/ exists.")
