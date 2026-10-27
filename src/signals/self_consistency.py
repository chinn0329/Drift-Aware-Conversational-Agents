from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def resample_responses(
    conversation_history: list, backbone, n: int = 2, temperature: float = 0.85,
    max_tokens: int = 256,
) -> List[str]:
    responses = []
    for _ in range(n):
        resp = backbone.generate(
            messages=conversation_history, temperature=temperature, max_tokens=max_tokens
        )
        responses.append(resp)
    return responses


def compute_variance(responses: List[str], embedding_model) -> float:
    if not responses or len(responses) < 2:
        return 0.0
    embeddings = embedding_model.encode(responses)
    dist_matrix = cosine_distances(embeddings)
    triu_indices = np.triu_indices_from(dist_matrix, k=1)
    return float(np.mean(dist_matrix[triu_indices]))