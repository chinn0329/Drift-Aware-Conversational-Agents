"""
Self-consistency variance — v2

Changes:
- Increased default resamples from 1 → 3 (gives a proper variance estimate;
  variance from a single pair is just distance, not a distribution statistic).
- Increased resample temperature 0.85 → 0.9 to widen the response distribution
  and make the signal more discriminative for confident vs uncertain outputs.
- Added length-adaptive max_tokens so long-context tasks don't get cut off.
"""

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def resample_responses(
    conversation_history: list,
    backbone,
    n: int = 3,
    temperature: float = 0.9,
) -> List[str]:
    """
    Generate `n` alternative responses at high temperature from the same context.
    A higher temperature spreads the distribution more, making semantic variance
    a more reliable uncertainty proxy.
    """
    # Estimate a reasonable max_tokens from the last assistant turn length
    last_assistant = next(
        (m["content"] for m in reversed(conversation_history) if m.get("role") == "assistant"),
        None,
    )
    max_tok = min(150, max(50, len((last_assistant or "").split()) + 30))

    responses = []
    for _ in range(n):
        resp = backbone.generate(
            messages=conversation_history,
            temperature=temperature,
            max_tokens=max_tok,
        )
        if resp and resp.strip():
            responses.append(resp)
    return responses


def compute_variance(responses: List[str], embedding_model) -> float:
    """
    Mean pairwise cosine distance over all unique pairs in `responses`.
    Returns 0.0 if fewer than 2 valid responses are provided.
    """
    valid = [r for r in responses if r and r.strip()]
    if len(valid) < 2:
        return 0.0
    embeddings = embedding_model.encode(valid, show_progress_bar=False)
    dist_matrix = cosine_distances(embeddings)
    triu_indices = np.triu_indices_from(dist_matrix, k=1)
    return float(np.mean(dist_matrix[triu_indices]))