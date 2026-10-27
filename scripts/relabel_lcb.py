import argparse, json, shutil, collections
from src.pipeline.evaluators import evaluate
from src.pipeline.dataset_builder import build_dataset

RAW = "data/processed/raw_logs.jsonl"
TRANS = "data/processed/transcripts.jsonl"
TASKS = "data/raw/lost_in_conversation/data/sharded_instructions_600.json"

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
args = ap.parse_args()

tasks = {t["task_id"]: t for t in json.load(open(TASKS, encoding="utf-8"))}

# last transcript per (conversation, model, run)
latest = {}
for line in open(TRANS, encoding="utf-8"):
    t = json.loads(line)
    if str(t.get("source", "")).startswith("lcb"):
        latest[(t["conversation_id"], t["assistant_model"], t.get("run_index", 0))] = t

new = {}
for key, t in latest.items():
    replies = [m["content"] for m in t["history"] if m["role"] == "assistant"]
    if not replies:
        continue
    ok, valid, why = evaluate(tasks[key[0]], replies[-1], "stop")
    new[key] = (bool(ok), bool(valid), why, t["eventual_success"], t["label_reason"])

print(f"{'task':36s} {'old':>14s}   {'new':>26s}")
for k, (ok, valid, why, old_ok, old_why) in sorted(new.items()):
    flag = "  <-- CHANGED" if (ok != old_ok) else ""
    print(f"{k[0]:36s} {str(old_ok)+'/'+old_why:>14s}   {str(ok)+'/'+why:>26s}{flag}")

if args.apply:
    shutil.copy(RAW, RAW + ".bak")
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    n = 0
    for r in rows:
        k = (r["conversation_id"], r.get("assistant_model"), r.get("run_index", 0))
        if k in new:
            ok, valid, why = new[k][:3]
            r.update(eventual_success=ok, label_valid=valid, label_reason=why,
                     label_version="lcb_private_v1")
            n += 1
    with open(RAW, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"updated {n} rows (backup at {RAW}.bak)")
    build_dataset(input_path=RAW, output_dir="data/splits")