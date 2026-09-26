"""
Section 5.4 — Commitment-to-underspecification ratio

For every turn where the instruction is genuinely underspecified (known from
the benchmark's own shard structure), check whether the model asked a
clarifying question or silently picked an interpretation and proceeded.
A running ratio of silent commitments to total underspecified turns is a
direct behavioral fingerprint of accumulating drift risk.
"""


def is_underspecified_turn(shard_metadata: dict) -> bool:
    """Look up whether this turn's shard is flagged as underspecified
    in the benchmark's own shard structure.
    """
    raise NotImplementedError


def did_ask_clarifying_question(turn_response: str) -> bool:
    """Classify whether the turn's response asked a clarifying question
    vs. silently picked an interpretation and proceeded.
    """
    raise NotImplementedError


def update_commitment_ratio(silent_commitments: int, total_underspecified_turns: int) -> float:
    """Return the running silent_commitment_ratio_t."""
    if total_underspecified_turns == 0:
        return 0.0
    return silent_commitments / total_underspecified_turns
