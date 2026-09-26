"""
Section 7 — Threshold calibration

Too low a threshold -> constant unnecessary intervention.
Too high a threshold -> catches drift only after it is unrecoverable.

The threshold is swept against a validation set; precision/recall of
"intervened when it should have" vs. "intervened when it did not need to"
is reported explicitly rather than fixed arbitrarily.
"""


def calibrate_probabilities(raw_scores, method: str = "platt"):
    """Calibrate raw classifier outputs into a genuine probability
    (Platt scaling or isotonic regression) so drift_score_t is meaningful
    as P(eventual failure).
    """
    raise NotImplementedError


def sweep_threshold(calibrated_scores, labels, thresholds=None) -> list:
    """Sweep candidate thresholds and return precision/recall at each,
    for the intervention gate (Section 7).
    """
    raise NotImplementedError
