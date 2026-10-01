from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class DriftFeatureVector:
    conversation_id: str
    turn_index: int
    self_consistency_variance: float
    contradiction_score: float
    confidence: float
    confidence_delta: float
    silent_commitment_ratio: float
    eventual_success: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)