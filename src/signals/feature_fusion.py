"""
FeatureBuilder — v2

Changes from v1:
- Contradiction: calls tracker.reset() between conversations; the tracker now
  only compares adjacent turns (not full history vs current), preventing saturation.
- Confidence: blends verbalized self-report with the hedging-language heuristic
  from the response text for a finer-grained, less-discretized score.
- Self-consistency: resamples 3 responses (n=3) instead of 1, giving a real
  variance estimate over 3 pairwise distances.
- Commitment: field renamed silent_commitment_ratio → clarification_ask_rate.
  No longer depends on is_underspecified; just tracks whether the assistant
  asked a clarifying question each turn.
"""

from sentence_transformers import SentenceTransformer
from src.signals.schemas import DriftFeatureVector
from src.signals.self_consistency import resample_responses, compute_variance
from src.signals.contradiction import ContradictionTracker
from src.signals.confidence import ConfidenceTracker
from src.signals.commitment_ratio import CommitmentRatioTracker


class FeatureBuilder:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        # Heavy models load ONCE; reset() clears per-conversation state.
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.contradiction_tracker = ContradictionTracker()
        self.reset()

    def reset(self):
        """Reset per-conversation state without reloading models."""
        self.confidence_tracker = ConfidenceTracker()
        self.commitment_tracker = CommitmentRatioTracker()
        self.contradiction_tracker.reset()   # clears the prev-turn window

    def compute_turn_features(
        self,
        conversation_id: str,
        turn_index: int,
        conversation_history: list,
        assistant_response: str,
        is_underspecified: bool,
        backbone,
        n_resamples: int = 3,
        eventual_success: bool = None,
    ) -> DriftFeatureVector:

        # ── Self-consistency variance (3 resamples at T=0.9) ──────────────────
        alt_samples = resample_responses(
            conversation_history, backbone, n=n_resamples, temperature=0.9
        )
        samples = [assistant_response] + alt_samples
        var_score = compute_variance(samples, self.embedding_model)

        # ── Contradiction (adjacent-turn only) ────────────────────────────────
        new_assumptions = self.contradiction_tracker.extract_assumptions(assistant_response)
        contra_score = self.contradiction_tracker.check_contradiction(
            [],  # historical_assumptions arg unused in v2; passing empty for compat
            new_assumptions,
        )

        # ── Confidence (verbalized + hedging heuristic blend) ─────────────────
        raw_conf = self.confidence_tracker.query_verbalized_confidence(
            conversation_history, backbone
        )
        composite_conf = self.confidence_tracker.compute_composite_confidence(
            verbalized=raw_conf,
            response_text=assistant_response,
        )
        conf_score, conf_delta = self.confidence_tracker.process_turn(composite_conf)

        # ── Clarification-ask rate ────────────────────────────────────────────
        asked_clarify = self.commitment_tracker.check_if_asked_clarification(
            assistant_response
        )
        clarification_rate = self.commitment_tracker.update(
            is_underspecified=is_underspecified,  # kept for compat, unused
            asked_clarification=asked_clarify,
        )

        return DriftFeatureVector(
            conversation_id=conversation_id,
            turn_index=turn_index,
            self_consistency_variance=var_score,
            contradiction_score=contra_score,
            confidence=conf_score,
            confidence_delta=conf_delta,
            clarification_ask_rate=clarification_rate,
            eventual_success=eventual_success,
        )