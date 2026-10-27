import sys, json
from src.pipeline import evaluators as E

tid = sys.argv[1]
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
tasks = {t["task_id"]: t for t in json.load(open("data/raw/lost_in_conversation/data/sharded_instructions_600.json", encoding="utf-8"))}
task = tasks[tid]

tr = None
for line in open("data/processed/transcripts.jsonl", encoding="utf-8"):
    t = json.loads(line)
    if t["conversation_id"] == tid and t["assistant_model"] == "openai/gpt-oss-20b":
        tr = t
last = [m["content"] for m in tr["history"] if m["role"] == "assistant"][-1]
code = E.last_code_block(last)
tests = E._load_tests(task)
test = tests[idx]
fn = E._func_name(task)

print("n tests:", len(tests), "| testtype:", test.get("testtype"), "| func_name:", fn)
print("\n--- final code ---\n", code)
print("\n--- test input ---\n", test["input"][:400])
print("\n--- expected ---\n", test["output"][:300])
if test.get("testtype") == "functional":
    res = E._run(E.HEADER + code + "\n" + E.FUNC_TAIL % fn, test["input"])
else:
    res = E._run(E.HEADER + code, test["input"])
print("\n--- result ---")
if res is None:
    print("timeout")
else:
    rc, out, err = res
    print("returncode:", rc)
    print("stdout:", out[:400])
    print("stderr:", err[-600:])