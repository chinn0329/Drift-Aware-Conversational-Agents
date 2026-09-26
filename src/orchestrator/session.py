"""
Section 10.1 / 11.2 — Live/runtime workflow.

Runs the baseline and drift-aware agent in parallel on the identical
simulated conversation. Per turn:

  1. New instruction/clarification revealed.
  2. Each agent generates its response.
  3. Drift-aware agent computes features -> classifier -> drift_score.
  4. If drift_score >= threshold: intervention controller acts, then proceeds.
     Else: proceed normally.
  5. Every turn/score/decision logged for replay + the demo's feature panel.
"""


class ConversationSession:
    def __init__(self, backbone, classifier, threshold: float):
        self.backbone = backbone
        self.classifier = classifier
        self.threshold = threshold
        self.history = []
        self.log = []

    def step_baseline(self, new_instruction: str) -> str:
        """Advance the vanilla (no drift-awareness) agent by one turn."""
        raise NotImplementedError

    def step_drift_aware(self, new_instruction: str) -> str:
        """Advance the drift-aware agent by one turn: generate, score,
        intervene if needed, log, return final response for this turn.
        """
        raise NotImplementedError
