"""
Contradiction tracker — detects when the assistant makes claims in a new turn
that contradict what it said in the *immediately preceding* turn.

Key design decisions (v2):
- We only compare the current turn's claims against the *previous* turn's claims
  (not the full conversation history). This prevents the score from saturating
  to 1.0 after turn 2, which happened in v1 because every new turn contradicted
  some sentence from the growing historical pool.
- We take the MAX contradiction probability across all cross-pairs. Using the
  mean underestimated individual claim conflicts; using max gives a more sensitive
  and interpretable "did anything flip?" signal.
- Claims are filtered for declarative content: we skip preamble phrases
  ("sure", "let me", "certainly"), short fragments (<20 chars), and code lines.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ContradictionTracker:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Stores claims from the *previous* turn only
        self._prev_turn_claims: List[str] = []

    def extract_assumptions(self, response_text: str, backbone=None) -> List[str]:
        """
        Extract core declarative claims from an assistant response.

        Filters out:
        - Very short fragments (< 20 chars)
        - Preamble / filler phrases
        - Lines that look like code (contain `=`, `(`, `{`)
        Returns at most 6 claims.
        """
        import re

        _SKIP_PREFIXES = (
            "sure", "here is", "certainly", "hello", "hi", "i have",
            "let me", "note:", "okay", "ok,", "of course", "great",
            "happy to", "i'll", "i will", "first", "next", "finally",
            "step", "to ", "in ", "we ",
        )
        _CODE_CHARS = {"=", "(", "{", "def ", "import ", "return ", "->", "//"}

        lines = [ln.strip("- *•›").strip() for ln in response_text.split("\n") if ln.strip()]
        claims: List[str] = []
        for line in lines:
            if len(line) < 20 or len(line) > 280:
                continue
            if line.lower().startswith(_SKIP_PREFIXES):
                continue
            if any(ch in line for ch in _CODE_CHARS):
                continue
            claims.append(line)

        # Fall back: split on sentence boundaries
        if not claims:
            sentences = [s.strip() for s in re.split(r"[.!?\n]+", response_text) if len(s.strip()) >= 20]
            claims = [s for s in sentences if not s.lower().startswith(_SKIP_PREFIXES)]

        return claims[:6]

    def check_contradiction(
        self,
        historical_assumptions: List[str],  # kept for API compat, unused now
        new_assumptions: List[str],
    ) -> float:
        """
        Compare *new_assumptions* against claims from the PREVIOUS turn only.
        Returns the maximum NLI contradiction probability over all cross-pairs.
        """
        prev = self._prev_turn_claims
        # Advance the window: previous turn becomes the new turn after this call
        self._prev_turn_claims = new_assumptions

        if not prev or not new_assumptions:
            return 0.0

        old_list, new_list = [], []
        for old_a in prev[:5]:
            for new_a in new_assumptions[:5]:
                old_list.append(old_a)
                new_list.append(new_a)

        if not old_list:
            return 0.0

        inputs = self.tokenizer(
            old_list, new_list,
            return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            # Index 0 = contradiction in DeBERTa NLI label ordering
            max_contradiction = float(probs[:, 0].max().item())

        return max_contradiction

    def reset(self) -> None:
        """Call between conversations to clear the previous-turn window."""
        self._prev_turn_claims = []