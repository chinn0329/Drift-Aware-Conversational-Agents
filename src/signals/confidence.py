"""
Section 5.3 — Verbalized-confidence trend

After each turn, ask the model to self-rate its confidence in its current
understanding of the task. A single value is noisy; the trend across turns
(steady decline or sudden drop) correlates with the model sensing drift
before it can fully articulate it.
"""


def get_verbalized_confidence(turn_response_context: str, backbone) -> float:
    """Ask the backbone to self-rate confidence in its current task understanding.

    Returns:
        confidence_t in [0, 1]
    """
    raise NotImplementedError


def compute_confidence_delta(confidence_history: list) -> float:
    """Compute confidence_delta_t from the running confidence history
    (e.g. confidence_t - confidence_{t-1}, or a windowed slope).
    """
    raise NotImplementedError
