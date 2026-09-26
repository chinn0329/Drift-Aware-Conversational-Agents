"""
Section 10.3 / 8.3 — LLM backend wrappers.

Multiple assistant backbones are used to multiply usable rows and test
generalization across models: frontier APIs (Claude / GPT) for the
head-to-head comparison, open-weight models (Llama, Qwen) for cheap
bulk data-generation runs.
"""

from abc import ABC, abstractmethod


class Backbone(ABC):
    @abstractmethod
    def generate(self, conversation_history: list, temperature: float = 0.0) -> str:
        ...


class ClaudeBackbone(Backbone):
    def generate(self, conversation_history: list, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OpenAIBackbone(Backbone):
    def generate(self, conversation_history: list, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OpenWeightBackbone(Backbone):
    """Wraps a local/self-hosted endpoint for Llama, Qwen, etc. (bulk generation)."""
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def generate(self, conversation_history: list, temperature: float = 0.0) -> str:
        raise NotImplementedError
