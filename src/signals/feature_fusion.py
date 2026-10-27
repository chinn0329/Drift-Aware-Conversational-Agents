from sentence_transformers import SentenceTransformer
from src.signals.schemas import DriftFeatureVector
from src.signals.self_consistency import resample_responses, compute_variance
from src.signals.contradiction import ContradictionTracker
from src.signals.confidence import ConfidenceTracker
from src.signals.commitment_ratio import CommitmentRatioTracker


def _dedupe(items):
    """Drop empty and repeated assumptions, keeping order."""
    seen, out = set(), []
    for a in items or []:
        key = str(a).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


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
        self.historical_assumptions = []   # kept for debugging / compatibility
        self.prev_assumptions = []         # assumptions from the most recent turn that had any

    def compute_turn_features(
        self,
        conversation_id: str,
        turn_index: int,
        conversation_history: list,
        assistant_response: str,
        is_underspecified: bool,
        backbone,
        n_resamples: int = 2,
        eventual_success: bool = None,
        resample_max_tokens: int = 256,
    ) -> DriftFeatureVector:
        # Never record features for an empty reply: raise so the caller skips the
        # conversation instead of logging imputed zeros.
        if not assistant_response or not assistant_response.strip():
            raise ValueError("empty assistant_response passed to compute_turn_features")

        # Work on copies so no helper can mutate the live conversation history.
        history = [dict(m) for m in conversation_history]

         # 1) self-consistency: re-answer the SAME user message (drop the model's own reply)
        resample_history = (
            history[:-1] if history and history[-1]["role"] == "assistant" else history
        )
        samples = resample_responses(
            resample_history, backbone, n=n_resamples, temperature=0.85,
            max_tokens=resample_max_tokens,
        )
        var_score = compute_variance(samples, self.embedding_model)

        # 2) contradiction: compare this turn's assumptions with the PREVIOUS turn's,
        #    not with the whole history. Scoring against all history keeps returning the
        #    same high value once any old assumption was revised, even if the model stops
        #    changing its mind.
        new_assumptions = _dedupe(
            self.contradiction_tracker.extract_assumptions(assistant_response, backbone)
        )
        if self.prev_assumptions and new_assumptions:
                contra_score = self.contradiction_tracker.check_contradiction(
                self.prev_assumptions, new_assumptions, embedder=self.embedding_model
            )
        else:
            contra_score = 0.0
        if new_assumptions:
            self.prev_assumptions = new_assumptions
            self.historical_assumptions.extend(new_assumptions)

        # 3) verbalized confidence
        raw_conf = self.confidence_tracker.query_verbalized_confidence(
            [dict(m) for m in history], backbone
        )
        conf_score, conf_delta = self.confidence_tracker.process_turn(raw_conf)
        if turn_index == 1:
            conf_delta = 0.0  # no previous turn: the tracker's delta is just confidence - 1.0

        # 4) commitment ratio
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
            eventual_success=eventual_success,
        )