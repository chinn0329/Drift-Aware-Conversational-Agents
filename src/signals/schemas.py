from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DriftFeatureVector:
    conversation_id: str
    turn_index: int
    self_consistency_variance: float
    contradiction_score: float
    confidence: float
    confidence_delta: float
    # Replaces silent_commitment_ratio (v1).
    # clarification_ask_rate = (turns where assistant asked ≥1 clarifying question)
    # divided by (total turns so far). Range [0, 1]. Higher = more clarification-seeking.
    clarification_ask_rate: float
    silent_commitment_ratio: float = 0.0
    eventual_success: Optional[bool] = None

    def __post_init__(self):
        if self.silent_commitment_ratio == 0.0 and self.clarification_ask_rate != 0.0:
            self.silent_commitment_ratio = self.clarification_ask_rate

    def to_dict(self) -> dict:
        return asdict(self)