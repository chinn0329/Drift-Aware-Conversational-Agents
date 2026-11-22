import glob, os
import pandas as pd

frames = []
for f in sorted(glob.glob("data/processed/*.jsonl")):
    d = pd.read_json(f, lines=True)
    d["_file"] = os.path.basename(f)
    frames.append(d)
al = pd.concat(frames, ignore_index=True)

print("columns containing 'consi':", [repr(c) for c in al.columns if "consi" in c])
for c in ["source", "assistant_model", "run_index", "label_valid", "label_reason", "truncated",
          "any_turn_truncated", "finish_reason", "extractor_version", "prompt_version",
          "label_version", "n_resamples"]:
    if c in al.columns:
        print(f"\n== {c} (by file)")
        print(al.groupby("_file")[c].value_counts(dropna=False).head(40).to_string())

run_key = ["_file", "conversation_id", "run_index", "assistant_model"]
runs = al.drop_duplicates(run_key)
print("\nruns per file:", runs.groupby("_file").size().to_dict())
print("runs per task (describe):\n", runs.groupby(["_file", "conversation_id"]).size().describe())
print("label conflicts WITHIN a run:",
      int((al.groupby(run_key).eventual_success.nunique() > 1).sum()))
t = runs.groupby("conversation_id").eventual_success.agg(["nunique", "size"])
print("tasks with several runs:", int((t["size"] > 1).sum()),
      "| of those with both outcomes:", int(((t["size"] > 1) & (t["nunique"] > 1)).sum()))