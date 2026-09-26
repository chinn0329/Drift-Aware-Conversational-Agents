"""
Section 7 — The three intervention strategies.
"""


def re_grounding(conversation_history: list, backbone) -> str:
    """Agent explicitly restates its current understanding of the task and
    asks for confirmation before proceeding. Cheapest, least disruptive,
    the default response.
    """
    raise NotImplementedError


def rollback(conversation_history: list, contradiction_turn_index: int, backbone) -> list:
    """Discard everything built on top of the identified turn and retry from
    that point using all information now available.
    """
    raise NotImplementedError


def targeted_clarification(conversation_history: list, ambiguous_element: str, backbone) -> str:
    """Ask a specific question about the exact ambiguous element identified
    by the extraction step — not a generic 'can you clarify?'.
    """
    raise NotImplementedError
