import json
from src.pipeline.evaluators import evaluate

d = json.load(open("data/raw/lost_in_conversation/data/sharded_instructions_600.json", encoding="utf-8"))
task = next(x for x in d if x["task_id"] == "sharded-livecodebench/2727")
tmpl = "```python\nclass Solution:\n    def countSeniors(self, details: List[str]) -> int:\n        return sum(1 for d in details if int(d[11:13]) {op} 60)\n```"
print(">=  ->", evaluate(task, tmpl.format(op=">="), "stop"))
print(">   ->", evaluate(task, tmpl.format(op=">"), "stop"))