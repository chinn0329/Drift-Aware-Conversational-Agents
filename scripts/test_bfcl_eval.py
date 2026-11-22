import json
from src.pipeline.evaluators import evaluate, _as_obj

d = json.load(open("data/raw/lost_in_conversation/data/sharded_instructions_600.json", encoding="utf-8"))
task = next(x for x in d if x["task_id"] == "sharded-BFCL/parallel_129")
gt = _as_obj(task["reference_answer"])
d1 = gt[0]["create_histogram"]["data"][0]
d2 = gt[1]["create_histogram"]["data"][0]

def block(s): return f"```python\n{s}\n```"
cases = {
    "correct":        block(f"[create_histogram(data={d1}, bins=5), create_histogram(data={d2}, bins=5)]"),
    "swapped order":  block(f"[create_histogram(data={d2}, bins=5), create_histogram(data={d1}, bins=5)]"),
    "positional":     block(f"[create_histogram({d1}, 5), create_histogram({d2}, 5)]"),
    "no brackets":    block(f"create_histogram(data={d1}, bins=5)\ncreate_histogram(data={d2}, bins=5)"),
    "wrong bins":     block(f"[create_histogram(data={d1}, bins=10), create_histogram(data={d2}, bins=5)]"),
    "one call only":  block(f"[create_histogram(data={d1}, bins=5)]"),
    "extra param":    block(f"[create_histogram(data={d1}, bins=5, color='red'), create_histogram(data={d2}, bins=5)]"),
    "prose":          "I will create two histograms with 5 bins each.",
    "cut off":        "```python\n[create_histogram(data=[12, 15,",
    "calls as strings": block(f'["create_histogram(data={d1}, bins=5)", "create_histogram(data={d2}, bins=5)"]'),
    "refusal":          "I’m sorry, but I can’t help with that.",
    "empty list":       block("[]"),
}
for name, text in cases.items():
    print(f"{name:14s} ->", evaluate(task, text, "stop"))