import re


class ConfidenceTracker:
    def __init__(self):
        self.prev_confidence = 1.0

    def query_verbalized_confidence(self, conversation_history: list, backbone) -> float:
        recent_history = (
            conversation_history[-4:]
            if len(conversation_history) > 4
            else conversation_history
        )
        prompt = recent_history + [
            {
                "role": "user",
                "content": (
                    "On a scale from 0.0 to 1.0, how confident are you in your understanding "
                    "of requirements so far? Output ONLY a float number."
                ),
            }
        ]
        raw = backbone.generate(messages=prompt, temperature=0.0, max_tokens=6)
        match = re.search(r"0\.\d+|1\.0|0", raw)
        return float(match.group(0)) if match else 0.8

    def process_turn(self, raw_confidence_score: float) -> tuple[float, float]:
        score = max(0.0, min(1.0, raw_confidence_score))
        delta = score - self.prev_confidence
        self.prev_confidence = score
        return score, delta