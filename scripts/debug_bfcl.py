import json, sys

tasks = {t["task_id"]: t for t in json.load(open("data/raw/lost_in_conversation/data/sharded_instructions_600.json", encoding="utf-8"))}
T = [json.loads(l) for l in open("data/processed/transcripts.jsonl", encoding="utf-8")]
ids = sys.argv[1:] or [t["conversation_id"] for t in T[-3:]]
for t in T:
    if t["conversation_id"] in ids and t["assistant_model"] == "openai/gpt-oss-20b":
        a = [m["content"] for m in t["history"] if m["role"] == "assistant"]
        print("=====", t["conversation_id"], t["label_reason"])
        print("REFERENCE:", str(tasks[t["conversation_id"]]["reference_answer"])[:600])
        print("LAST REPLY:", repr(a[-1][:700]))