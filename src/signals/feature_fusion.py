from sentence_transformers import SentenceTransformer
from src.signals.schemas import DriftFeatureVector
from src.signals.self_consistency import resample_responses, compute_variance
from src.signals.contradiction import ContradictionTracker
from src.signals.confidence import ConfidenceTracker
from src.signals.commitment_ratio import CommitmentRatioTracker

class FeatureBuilder:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        # Models load into memory ONLY ONCE when FeatureBuilder is created
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.contradiction_tracker = ContradictionTracker()
        self.reset()

    def reset(self):
        """Resets conversation state between tasks without reloading models."""
        self.confidence_tracker = ConfidenceTracker()
        self.commitment_tracker = CommitmentRatioTracker()
        self.historical_assumptions = []

    def compute_turn_features(
        self,
        conversation_id: str,
        turn_index: int,
        conversation_history: list,
        assistant_response: str,
        is_underspecified: bool,
        backbone,
        n_resamples: int = 2,
        eventual_success: bool = None
    ) -> DriftFeatureVector:
        samples = resample_responses(conversation_history, backbone, n=n_resamples, temperature=0.85)
        var_score = compute_variance(samples, self.embedding_model)

        new_assumptions = self.contradiction_tracker.extract_assumptions(assistant_response, backbone)
        contra_score = self.contradiction_tracker.check_contradiction(
            self.historical_assumptions, new_assumptions
        )
        self.historical_assumptions.extend(new_assumptions)

        raw_conf = self.confidence_tracker.query_verbalized_confidence(conversation_history, backbone)
        conf_score, conf_delta = self.confidence_tracker.process_turn(raw_conf)

        asked_clarify = self.commitment_tracker.check_if_asked_clarification(assistant_response)
        commit_ratio = self.commitment_tracker.update(is_underspecified, asked_clarify)

        return DriftFeatureVector(
            conversation_id=conversation_id,
            turn_index=turn_index,
            self_consistency_variance=var_score,
            contradiction_score=contra_score,
            confidence=conf_score,
            confidence_delta=conf_delta,
            silent_commitment_ratio=commit_ratio,
            eventual_success=eventual_success
        )