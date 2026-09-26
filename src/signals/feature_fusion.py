"""
Section 5.5 — Fusing the signals

Per turn, the four signals form a feature vector:

    features_t = [
        self_consistency_variance_t,
        contradiction_score_t,
        confidence_t,
        confidence_delta_t,
        silent_commitment_ratio_t,
    ]

This vector is passed to the trained drift classifier (src/classifier/) which
outputs a single calibrated drift_score_t in [0, 1].
"""

from dataclasses import dataclass


@dataclass
class TurnFeatures:
    self_consistency_variance_t: float
    contradiction_score_t: float
    confidence_t: float
    confidence_delta_t: float
    silent_commitment_ratio_t: float

    def as_vector(self) -> list:
        return [
            self.self_consistency_variance_t,
            self.contradiction_score_t,
            self.confidence_t,
            self.confidence_delta_t,
            self.silent_commitment_ratio_t,
        ]


def build_feature_vector(**signal_outputs) -> TurnFeatures:
    """Assemble the five-element feature vector for one turn from the
    individual signal computations in self_consistency.py, contradiction.py,
    and confidence.py / commitment_ratio.py.
    """
    return TurnFeatures(**signal_outputs)
