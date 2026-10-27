import re
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for a in items:
        k = _norm(a)
        if k and k not in seen:
            seen.add(k)
            out.append(a.strip())
    return out


class ContradictionTracker:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Read the label order from the model instead of assuming index 0.
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.contra_idx = next(i for i, l in id2label.items() if l.startswith("contra"))

    def extract_assumptions(self, response_text: str, backbone) -> List[str]:
        prompt = [
            {"role": "system", "content": "List assumptions made in the text below as short bullet points. Output ONLY 1-3 bullet points."},
            {"role": "user", "content": response_text[:1200]},
        ]
        raw = backbone.generate(messages=prompt, temperature=0.0, max_tokens=80)
        out = []
        for line in raw.split("\n"):
            line = re.sub(r"^[\s\-\*\u2022\d\.\)]+", "", line).strip()
            if len(line) < 6 or line.endswith(":"):   # skip headings like "Assumptions:"
                continue
            out.append(line)
        return _dedupe(out)[:3]

    @torch.no_grad()
    def _contra_probs(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            premises, hypotheses, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(self.device)
        probs = torch.softmax(self.model(**inputs).logits, dim=1)
        return probs[:, self.contra_idx].cpu().numpy()

    def check_contradiction(
        self,
        old_assumptions: List[str],
        new_assumptions: List[str],
        embedder=None,
        min_sim: float = 0.45,
    ) -> float:
        """How strongly the CHANGED assumptions in the new turn contradict their closest
        old assumption. Returns 0.0 when nothing changed or nothing is comparable."""
        old = _dedupe(old_assumptions)
        new = _dedupe(new_assumptions)
        if not old or not new:
            return 0.0

        old_norm = {_norm(a) for a in old}
        changed = [a for a in new if _norm(a) not in old_norm]
        if not changed:                      # identical assumptions -> no drift
            return 0.0

        pairs = []                           # (old, new)
        if embedder is not None:
            e_old = np.asarray(embedder.encode(old, normalize_embeddings=True))
            e_new = np.asarray(embedder.encode(changed, normalize_embeddings=True))
            sims = e_new @ e_old.T
            for i, n in enumerate(changed):
                j = int(np.argmax(sims[i]))
                if sims[i, j] >= min_sim:    # only compare statements about the same thing
                    pairs.append((old[j], n))
        else:
            pairs = [(o, n) for n in changed for o in old]

        if not pairs:
            return 0.0

        olds = [p[0] for p in pairs]
        news = [p[1] for p in pairs]
        fwd = self._contra_probs(olds, news)
        bwd = self._contra_probs(news, olds)
        return float(np.max((fwd + bwd) / 2.0))