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