"""
Section 5.2 — Assumption-contradiction tracking

After each turn, a cheap extraction prompt lists the assumptions the response
makes about the task. Each new turn's assumptions are checked against the
running list of prior assumptions using an NLI model (entailment / neutral /
contradiction) or embedding similarity as a cheaper proxy.

A detected contradiction = the model's own understanding of the task has
flipped -> strong, largely unambiguous drift signal.
"""

from typing import List


def extract_assumptions(turn_response: str, backbone) -> List[str]:
    """Prompt the backbone to list assumptions made in this turn's response."""
    raise NotImplementedError


def check_contradiction(new_assumptions: List[str], prior_assumptions: List[str],
                         nli_model) -> float:
    """Compare new assumptions against the running list via NLI (or embedding
    similarity proxy) and return a contradiction_score_t in [0, 1].
    """
    raise NotImplementedError
