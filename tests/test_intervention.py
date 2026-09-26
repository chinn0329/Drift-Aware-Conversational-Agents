"""Unit tests for the intervention policy and strategies (Section 7)."""

from src.intervention.policy import should_intervene


def test_should_intervene_above_threshold():
    assert should_intervene(0.8, 0.5) is True


def test_should_intervene_below_threshold():
    assert should_intervene(0.3, 0.5) is False


# TODO once implemented:
# - test select_intervention routes contradiction-dominant -> re_grounding/rollback correctly
# - test select_intervention routes underspecification-dominant -> targeted_clarification
