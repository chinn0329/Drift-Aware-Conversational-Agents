"""
Section 5.1 — Self-consistency divergence

At each turn, resample the assistant's response N times (~5) at moderate
temperature (0.7-1.0) with conversation history held fixed. Embed the N
responses and measure their pairwise spread (e.g. average cosine distance).

Low spread  -> model converges on the same answer regardless of sampling (solid grounding)
High spread -> model is effectively guessing (precedes a wrong commitment)
"""

from typing import List


def resample_responses(conversation_history: list, backbone, n: int = 5,
                        temperature: float = 0.85) -> List[str]:
    """Resample the next assistant turn N times with history fixed.

    Args:
        conversation_history: prior turns (fixed context).
        backbone: LLM backend wrapper (see src/pipeline/backends.py).
        n: number of resamples.
        temperature: sampling temperature.

    Returns:
        List of N response strings.
    """
    raise NotImplementedError


def compute_variance(responses: List[str], embedding_model) -> float:
    """Embed responses and compute average pairwise cosine distance.

    Returns:
        self_consistency_variance_t (float)
    """
    raise NotImplementedError
