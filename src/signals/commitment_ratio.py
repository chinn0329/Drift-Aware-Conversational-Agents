def update(silent_commitments: int, underspecified_turns: int) -> float:
    """
    Calculate the commitment ratio.

    Args:
        underspecified_turns: Total underspecified turns.
        silent_commitments: Number of silent commitments.

    Returns:
        Commitment ratio between 0 and 1.
    """
    if underspecified_turns == 0:
        return 0.0

    return silent_commitments / underspecified_turns

class CommitmentRatioTracker:
    def __init__(self):
        self.underspecified_turns = 0
        self.silent_commitments = 0

    def check_if_asked_clarification(self, response_text: str) -> bool:
        return "?" in response_text and any(
            kw in response_text.lower() for kw in ["clarify", "specify", "mean", "which", "do you want"]
        )

    def update(self, is_underspecified: bool, asked_clarification: bool) -> float:
        if is_underspecified:
            self.underspecified_turns += 1
            if not asked_clarification:
                self.silent_commitments += 1
        return (self.silent_commitments / self.underspecified_turns) if self.underspecified_turns > 0 else 0.0