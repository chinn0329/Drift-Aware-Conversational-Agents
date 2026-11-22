import glob
import pandas as pd

frames = []
for f in sorted(glob.glob("data/processed/*.jsonl")):
    d = pd.read_json(f, lines=True)
    print(f, d.shape, "conversations:", d.conversation_id.nunique())
    print("  columns:", list(d.columns))
    frames.append(d.assign(_file=f))

al = pd.concat(frames, ignore_index=True)
print("\ntotal rows:", len(al), "| conversations:", al.conversation_id.nunique())
print("rows sharing (conversation_id, turn_index):",
      int(al.duplicated(["conversation_id", "turn_index"], keep=False).sum()))
print("conversation_ids in more than one file:",
      int((al.groupby("conversation_id")["_file"].nunique() > 1).sum()))
print("conversations with conflicting labels:",
      int((al.groupby("conversation_id")["eventual_success"].nunique() > 1).sum()))