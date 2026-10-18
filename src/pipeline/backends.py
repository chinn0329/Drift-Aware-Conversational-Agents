import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _load_groq_keys() -> list[str]:
    """Load all available Groq API keys from environment (supports up to 4)."""
    keys = []
    for env_var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"):
        key = os.getenv(env_var)
        if key and key.strip():
            keys.append(key.strip())
    return keys


class LLMBackend:
    def __init__(self, provider: str = None, model_name: str = None):
        groq_keys = _load_groq_keys()

        if provider is None:
            provider = "groq" if groq_keys else "ollama"

        self.provider = provider.lower()

        if self.provider == "groq":
            if not groq_keys:
                raise ValueError("No GROQ_API_KEY found in environment or .env file.")
            self.model_name = model_name or "qwen/qwen3.8-27b"
            self._groq_keys = groq_keys
            self._key_index = 0
            print(f"Groq backend initialised with {len(groq_keys)} API key(s).", flush=True)
            self.client = self._make_groq_client(self._groq_keys[0])

        elif self.provider == "ollama":
            self.model_name = model_name or "llama3.2:3b"
            endpoint = os.getenv("LOCAL_MODEL_ENDPOINT", "http://localhost:11434/v1")
            self.client = OpenAI(
                base_url=endpoint,
                api_key="ollama",
                timeout=120.0,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _make_groq_client(self, api_key: str) -> OpenAI:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            timeout=60.0,
        )

    def _rotate_key(self) -> None:
        """Switch to the next Groq API key in round-robin order."""
        if len(self._groq_keys) <= 1:
            return
        self._key_index = (self._key_index + 1) % len(self._groq_keys)
        self.client = self._make_groq_client(self._groq_keys[self._key_index])
        print(
            f"Rotated to Groq key #{self._key_index + 1} / {len(self._groq_keys)}",
            flush=True,
        )

    def generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 512) -> str:
        max_retries = 8
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = resp.choices[0]
                content = choice.message.content
                if not content and hasattr(choice.message, "reasoning") and choice.message.reasoning:
                    content = choice.message.reasoning
                return content or ""
            except Exception as e:
                err_str = str(e).lower()
                status_code = getattr(e, "status_code", None)
                if status_code == 429 or "rate limit" in err_str:
                    # Rotate key first, then wait a short time
                    self._rotate_key()
                    wait_time = min(15, 2 ** attempt)
                    print(f"Rate limited. Switched key, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})", flush=True)
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    print(f"Retrying request ({attempt + 1}/{max_retries}) due to: {e}", flush=True)
                    time.sleep(1)
                else:
                    print(f"{self.provider.capitalize()} generation error: {e}", flush=True)
                    return ""
        return ""