import json
import os
import pandas as pd
import numpy as np

def build_dataset_splits(
    raw_logs_jsonl: str = "data/processed/raw_logs.jsonl",
    output_dir: str = "data/splits"
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    with open(raw_logs_jsonl, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    unique_convs = df["conversation_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_convs)

    n_total = len(unique_convs)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    train_ids = set(unique_convs[:n_train])
    val_ids = set(unique_convs[n_train: n_train + n_val])
    test_ids = set(unique_convs[n_train + n_val:])

    df[df["conversation_id"].isin(train_ids)].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df[df["conversation_id"].isin(val_ids)].to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df[df["conversation_id"].isin(test_ids)].to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Successfully exported dataset splits to {output_dir}/")

if __name__ == "__main__":
    build_dataset_splits()