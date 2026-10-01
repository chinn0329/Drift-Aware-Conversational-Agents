import os
from openai import OpenAI

class LLMBackend:
    def __init__(self, provider: str = "ollama", model_name: str = "llama3.2:3b"):
        self.provider = provider.lower()
        self.model_name = model_name

        if self.provider == "ollama":
            self.client = OpenAI(
                base_url=os.getenv("LOCAL_MODEL_ENDPOINT", "http://localhost:11434/v1"),
                api_key="ollama"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(self, messages: list, temperature: float = 0.7) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"Ollama generation error: {e}")
            return ""