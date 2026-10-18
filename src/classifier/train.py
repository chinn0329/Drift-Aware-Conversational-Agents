"""Train and evaluate the MLP drift classifier on task-level data splits."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "self_consistency_variance",
    "contradiction_score",
    "confidence",
    "confidence_delta",
    "silent_commitment_ratio",
]
LABEL_COLUMN = "eventual_success"


def load_splits(splits_dir: str | Path):
    """Load the precomputed train, validation, and test CSV files."""
    splits_dir = Path(splits_dir)
    frames = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} split: {path}")
        frames[name] = pd.read_csv(path)
        required = set(FEATURE_COLUMNS + [LABEL_COLUMN])
        missing = required.difference(frames[name].columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frames["train"], frames["val"], frames["test"]


def _prepare(frame: pd.DataFrame):
    labels = frame[LABEL_COLUMN]
    if labels.isna().any():
        raise ValueError(f"{LABEL_COLUMN} contains null values.")
    if labels.dtype == object:
        normalized = labels.astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
        unknown = set(normalized.unique()).difference(mapping)
        if unknown:
            raise ValueError(f"Unrecognized {LABEL_COLUMN} values: {sorted(unknown)}")
        labels = normalized.map(mapping)
    labels = labels.astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN} must contain only non-null booleans.")

    features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any():
        raise ValueError("Feature columns contain null values.")
    return features, labels


def train_mlp(train_df: pd.DataFrame) -> Pipeline:
    """Fit an MLP using only the five numeric drift features."""
    x_train, y_train = _prepare(train_df)
    if y_train.nunique() < 2:
        raise ValueError("The training split needs both success-label classes.")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=1000,
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate(model: Pipeline, frame: pd.DataFrame) -> dict:
    x, y = _prepare(frame)
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None
    )
    return result


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_path: str | Path,
) -> dict:
    model = train_mlp(train_df)
    metrics = {
        "validation": evaluate(model, val_df),
        "test": evaluate(model, test_df),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the MLP drift classifier.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing task-level train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/drift_mlp.joblib"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("data/models/drift_mlp_metrics.json"),
    )
    args = parser.parse_args()

    train_df, val_df, test_df = load_splits(args.splits_dir)
    metrics = train_and_evaluate(
        train_df,
        val_df,
        test_df,
        model_path=args.model_path,
    )
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved MLP pipeline to {args.model_path}")
    print(f"Saved metrics to {args.metrics_path}")
    print(
        "Reminder: results on synthetic proxy labels are for pipeline testing only, "
        "not real benchmark performance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
