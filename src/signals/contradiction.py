from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ContradictionTracker:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def extract_assumptions(self, response_text: str, backbone) -> List[str]:
        prompt = [
            {"role": "system", "content": "List assumptions made in the text below as short bullet points. Output ONLY bullet points."},
            {"role": "user", "content": response_text}
        ]
        raw = backbone.generate(messages=prompt, temperature=0.0)
        return [line.strip("- *").strip() for line in raw.split("\n") if line.strip()]

    def check_contradiction(self, historical_assumptions: List[str], new_assumptions: List[str]) -> float:
        if not historical_assumptions or not new_assumptions:
            return 0.0
        max_contradiction = 0.0
        for old_a in historical_assumptions:
            for new_a in new_assumptions:
                inputs = self.tokenizer(old_a, new_a, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)[0]
                    max_contradiction = max(max_contradiction, float(probs[0]))
        return max_contradiction