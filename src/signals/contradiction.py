from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ContradictionTracker:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_assumptions(self, response_text: str, backbone) -> List[str]:
        prompt = [
            {"role": "system", "content": "List assumptions made in the text below as short bullet points. Output ONLY 1-3 bullet points."},
            {"role": "user", "content": response_text[:1200]}
        ]
        raw = backbone.generate(messages=prompt, temperature=0.0, max_tokens=80)
        return [line.strip("- *").strip() for line in raw.split("\n") if line.strip()][:5]

    def check_contradiction(self, historical_assumptions: List[str], new_assumptions: List[str]) -> float:
        if not historical_assumptions or not new_assumptions:
            return 0.0

        old_list = []
        new_list = []
        for old_a in historical_assumptions[-8:]:
            for new_a in new_assumptions[:5]:
                old_list.append(old_a)
                new_list.append(new_a)

        if not old_list:
            return 0.0

        inputs = self.tokenizer(
            old_list, new_list, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            max_contradiction = float(probs[:, 0].max().item())

        return max_contradiction