import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset(
    input_path: str = "data/processed/raw_logs.jsonl",
    output_dir: str = "data/splits",
    random_state: int = 42,
):
    """Builds task-level disjoint train/val/test splits from raw feature logs."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        print(f"No log file found at {input_path}")
        return

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("No records found to build splits from.")
        return

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "silent_commitment_ratio" not in df.columns and "clarification_ask_rate" in df.columns:
        df["silent_commitment_ratio"] = df["clarification_ask_rate"]
    elif "clarification_ask_rate" not in df.columns and "silent_commitment_ratio" in df.columns:
        df["clarification_ask_rate"] = df["silent_commitment_ratio"]

    required_cols = [
        "conversation_id",
        "turn_index",
        "self_consistency_variance",
        "contradiction_score",
        "confidence",
        "confidence_delta",
        "silent_commitment_ratio",
        "eventual_success",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in records.")

    # Fill any null eventual_success if needed or ensure booleans
    if df["eventual_success"].isna().any():
        print("Warning: Found null eventual_success values; filling with False.")
        df["eventual_success"] = df["eventual_success"].fillna(False)

    # Ensure boolean
    df["eventual_success"] = df["eventual_success"].astype(bool)

    unique_convs = df["conversation_id"].unique()

    if len(unique_convs) < 3:
        df.to_csv(output_dir / "train.csv", index=False)
        print(
            f"Dataset updated: All {len(df)} turns saved to train.csv (insufficient tasks for split)."
        )
        return

    # Task-level split: 70% Train, 15% Val, 15% Test
    train_ids, test_val_ids = train_test_split(
        unique_convs, test_size=0.30, random_state=random_state
    )
    if len(test_val_ids) >= 2:
        val_ids, test_ids = train_test_split(
            test_val_ids, test_size=0.50, random_state=random_state
        )
    else:
        val_ids = test_val_ids
        test_ids = test_val_ids

    train_df = df[df["conversation_id"].isin(train_ids)].sort_values(
        ["conversation_id", "turn_index"]
    )
    val_df = df[df["conversation_id"].isin(val_ids)].sort_values(
        ["conversation_id", "turn_index"]
    )
    test_df = df[df["conversation_id"].isin(test_ids)].sort_values(
        ["conversation_id", "turn_index"]
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"Updated {output_dir}: "
        f"Train={len(train_df)} turns ({len(train_ids)} tasks, {train_df['eventual_success'].sum()} True) | "
        f"Val={len(val_df)} turns ({len(val_ids)} tasks, {val_df['eventual_success'].sum()} True) | "
        f"Test={len(test_df)} turns ({len(test_ids)} tasks, {test_df['eventual_success'].sum()} True)"
    )


if __name__ == "__main__":
    build_dataset()