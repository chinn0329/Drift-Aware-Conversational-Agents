"""
Clarification-ask rate tracker — v2

Replaces the silent_commitment_ratio (v1) which depended on an unreliable
`is_underspecified` heuristic that fired on almost every short follow-up turn.

New signal: clarification_ask_rate
  = (number of turns where the assistant asked ≥1 clarifying question)
    / (total turns in the conversation so far)

Range: [0, 1]
Interpretation:
  - Low rate (near 0) = assistant ploughs ahead without seeking clarification
    → higher risk of silent commitment / drift
  - High rate (near 1) = assistant consistently checks its understanding
    → lower risk of drift (assuming turns aren't just filler questions)

No dependency on `is_underspecified`, so it generalises across all task sources.
"""


class CommitmentRatioTracker:
    """
    Tracks how often the assistant proactively asked a clarifying question.
    Kept named `CommitmentRatioTracker` for backwards-compatibility with
    FeatureBuilder; the metric it computes is `clarification_ask_rate`.
    """

    def __init__(self):
        self.total_turns: int = 0
        self.clarification_turns: int = 0

    def check_if_asked_clarification(self, response_text: str) -> bool:
        """
        Returns True if the assistant response contains at least one
        genuine clarifying question.

        Conditions (both must hold):
        1. The response contains a "?" character.
        2. The response contains at least one known clarification-seeking phrase.
        """
        if "?" not in response_text:
            return False
        text_lower = response_text.lower()
        return any(
            kw in text_lower
            for kw in [
                "clarify", "specify", "which", "what do you mean",
                "do you want", "could you", "can you clarify", "can you specify",
                "what exactly", "are you referring", "did you mean",
                "please provide", "need more", "not sure what you",
                "ambiguous", "unclear", "could you elaborate",
                "what kind of", "which one", "which type",
            ]
        )

    def update(
        self,
        is_underspecified: bool,  # kept for API compat, unused
        asked_clarification: bool,
    ) -> float:
        """
        Record this turn and return the running clarification_ask_rate.
        `is_underspecified` is accepted but ignored (no longer required).
        """
        self.total_turns += 1
        if asked_clarification:
            self.clarification_turns += 1
        return (
            self.clarification_turns / self.total_turns
            if self.total_turns > 0
            else 0.0
        )