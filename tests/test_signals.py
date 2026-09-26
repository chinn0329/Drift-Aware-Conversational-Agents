"""Unit tests for the four drift signals (Section 5)."""

from src.signals.commitment_ratio import update_commitment_ratio


def test_commitment_ratio_zero_when_no_underspecified_turns():
    assert update_commitment_ratio(0, 0) == 0.0


def test_commitment_ratio_basic():
    assert update_commitment_ratio(2, 4) == 0.5


# TODO once implemented:
# - test self_consistency.compute_variance on identical vs. divergent responses
# - test contradiction.check_contradiction on a known entailment/contradiction pair
# - test confidence.compute_confidence_delta on a declining trend
