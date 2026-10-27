import json
from src.pipeline.evaluators import evaluate

d = json.load(open("data/raw/lost_in_conversation/data/sharded_instructions_600.json", encoding="utf-8"))
task = next(x for x in d if str(x.get("source", "")).startswith("lcb"))
print(task["task_id"])

cases = {
    "correct":        "```python\nclass Solution:\n    def countSeniors(self, details: List[str]) -> int:\n        return sum(int(d[11:13]) > 60 for d in details)\n```",
    "wrong logic":    "```python\nclass Solution:\n    def countSeniors(self, details: List[str]) -> int:\n        return sum(int(d[11:13]) > 50 for d in details)\n```",
    "renamed func":   "```python\ndef count(details):\n    return sum(int(d[11:13]) > 60 for d in details)\n```",
    "infinite loop":  "```python\nclass Solution:\n    def countSeniors(self, details):\n        while True: pass\n```",
    "cut off":        "```python\nclass Solution:\n    def countSeniors(self, details):\n        return sum(",
    "no code":        "I think you should count ages above sixty.",
}
for name, text in cases.items():
    print(f"{name:14s} ->", evaluate(task, text, "stop"))