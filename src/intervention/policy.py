"""
Section 7 — Intervention Logic

When the drift score crosses the calibrated threshold, a rule-based policy
layer (not a trained model) selects one of three interventions based on
which signal contributed most to the score:

  Contradiction dominant / general high score        -> Re-grounding
  Contradiction traceable to a specific earlier turn  -> Rollback
  Underspecification ratio dominant (silent guessing) -> Targeted clarification

Stretch goal (reported as ablation, not default): replace this fixed rule
with a second small classifier trained on which intervention historically
recovered performance best for a given feature profile.
"""

from src.signals.feature_fusion import TurnFeatures


def should_intervene(drift_score: float, threshold: float) -> bool:
    return drift_score >= threshold


def select_intervention(features: TurnFeatures, dominant_signal: str) -> str:
    """Return one of 'regrounding', 'rollback', 'targeted_clarification'
    based on which signal dominates the drift score, per the trigger table
    in Section 7.
    """
    raise NotImplementedError
