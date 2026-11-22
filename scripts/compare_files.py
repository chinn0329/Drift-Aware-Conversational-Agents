import numpy as np
import pandas as pd

a = pd.read_json("data/processed/raw_logs2.jsonl", lines=True)
b = pd.read_json("data/processed/raw_logs3.jsonl", lines=True)
print("odd column in file2:", [repr(c) for c in a.columns if "consi" in c])

for name, d in [("file2", a), ("file3", b)]:
    print(f"\n== {name}")
    for c in ["run_index", "assistant_model", "label_valid", "truncated",
              "extractor_version", "prompt_version", "n_resamples"]:
        print(f"  {c}: {d[c].value_counts(dropna=False).head(8).to_dict()}")
    print("  label_reason:", d["label_reason"].value_counts(dropna=False).head(8).to_dict())

# Same run in both files?
key = ["conversation_id", "turn_index", "run_index", "assistant_model"]
m = a.merge(b, on=key, suffixes=("_2", "_3"))
print(f"\nrows matching on id+turn+run+model: {len(m)}")
for c in ["self_consistency_variance", "contradiction_score", "confidence", "silent_commitment_ratio"]:
    x, y = m[c + "_2"].astype(float), m[c + "_3"].astype(float)
    print(f"  {c}: identical in {np.isclose(x, y, equal_nan=True).mean():.0%} of matched rows")
print("  eventual_success equal in",
      f"{(m['eventual_success_2'] == m['eventual_success_3']).mean():.0%}")

# The 11 conflicts
al = pd.concat([a.assign(_file=2), b.assign(_file=3)], ignore_index=True)
n = al.groupby("conversation_id").eventual_success.nunique()
bad = n[n > 1].index
print(f"\n{len(bad)} conflicting conversation_ids:")
cols = ["conversation_id", "_file", "run_index", "assistant_model", "eventual_success", "label_valid", "label_reason"]
print(al[al.conversation_id.isin(bad)][cols].drop_duplicates().sort_values(cols[:3]).to_string(index=False))