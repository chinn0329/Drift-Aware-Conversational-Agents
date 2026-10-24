"""
Confidence tracker — v2

Changes from v1:
- Added a hedging-language heuristic that scores uncertainty *directly from the
  response text* without an extra API call. This gives a continuous signal
  (0.0–1.0) that is not discretized to {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
- The verbalized confidence (from the LLM self-report) is blended 60/40 with the
  hedging-language score, producing a finer-grained composite.
- Hedging language: hedged phrasing raises uncertainty; assertive / resolved
  phrasing lowers it. Net score is the normalized difference.
"""

import re
from typing import Tuple

# Phrases that signal *lower* confidence / more uncertainty
_HEDGE_PHRASES = [
    "i think", "i believe", "i'm not sure", "not certain", "might be",
    "could be", "possibly", "perhaps", "probably", "it seems", "it appears",
    "i'm unsure", "i am unsure", "unclear", "not entirely sure",
    "i may be wrong", "assuming", "i assume", "roughly", "approximately",
    "around", "or so", "i guess", "tend to", "generally",
    "in most cases", "typically", "usually", "often", "sometimes",
    "hard to say", "difficult to say", "not 100%", "not fully",
]

# Phrases that signal *higher* confidence / resolution
_ASSERT_PHRASES = [
    "the answer is", "the result is", "therefore", "thus", "clearly",
    "definitely", "certainly", "this means", "we know", "it is",
    "the solution is", "final answer", "in conclusion", "to summarize",
    "confirmed", "proven", "exactly", "precisely",
]


def _hedging_score(text: str) -> float:
    """
    Returns a float in [0, 1] representing *uncertainty* from hedging language.
    0.0 = very assertive, 1.0 = very hedged.
    """
    text_lower = text.lower()
    hedge_count = sum(1 for p in _HEDGE_PHRASES if p in text_lower)
    assert_count = sum(1 for p in _ASSERT_PHRASES if p in text_lower)

    total = hedge_count + assert_count
    if total == 0:
        return 0.5  # neutral — no strong signal
    # High hedge ratio → high uncertainty → low confidence
    hedge_ratio = hedge_count / total
    return hedge_ratio  # 0 = all assertions, 1 = all hedges


class ConfidenceTracker:
    def __init__(self):
        self.prev_confidence = 1.0

    def query_verbalized_confidence(self, conversation_history: list, backbone) -> float:
        """Ask the model for its self-reported confidence (0.0–1.0)."""
        recent_history = (
            conversation_history[-4:]
            if len(conversation_history) > 4
            else conversation_history
        )
        prompt = recent_history + [
            {
                "role": "user",
                "content": (
                    "On a scale from 0.0 to 1.0, how confident are you in your "
                    "understanding of requirements so far? "
                    "Output ONLY a decimal number (e.g. 0.73)."
                ),
            }
        ]
        raw = backbone.generate(messages=prompt, temperature=0.0, max_tokens=8)
        match = re.search(r"[01]\.\d+|0|1", raw)
        return float(match.group(0)) if match else 0.8

    def compute_composite_confidence(
        self, verbalized: float, response_text: str
    ) -> float:
        """
        Blend verbalized self-report with hedging-language heuristic.
        hedge_score is *uncertainty* → confidence = 1 - hedge_score.
        Returns composite confidence in [0, 1].
        """
        hedge_uncertainty = _hedging_score(response_text)
        hedge_confidence = 1.0 - hedge_uncertainty
        # 60% weight on verbalized self-report, 40% on text heuristic
        composite = 0.6 * verbalized + 0.4 * hedge_confidence
        return max(0.0, min(1.0, composite))

    def process_turn(
        self, raw_confidence_score: float
    ) -> Tuple[float, float]:
        score = max(0.0, min(1.0, raw_confidence_score))
        delta = score - self.prev_confidence
        self.prev_confidence = score
        return score, delta