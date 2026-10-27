import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import re
from openai import RateLimitError

class LLMError(RuntimeError): pass
class DailyLimitError(LLMError): pass

THINK_RE = re.compile(r"<think>.*?</think>", re.S)

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
        self.last_finish_reason = None

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
            max_retries=0,
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
        for attempt in range(6):
            try:
                    extra = {}
                    budget = max_tokens
                    if "gpt-oss" in self.model_name:
                        extra["extra_body"] = {"reasoning_effort": "low"}   # keep hidden thinking short
                    budget = max_tokens + 1024                          # headroom for the thinking tokens
                    resp = self.client.chat.completions.create(
                        model=self.model_name, messages=messages,
                        temperature=temperature, max_tokens=budget, **extra,
                )
            except RateLimitError as e:
                msg = str(e).lower()
                if "per day" in msg or "(tpd)" in msg:
                    raise DailyLimitError(str(e))      # retrying cannot help
                hdr = getattr(getattr(e, "response", None), "headers", {}) or {}
                wait = float(hdr.get("retry-after", min(30, 2 ** attempt)))
                print(f"Rate limited, waiting {wait:.0f}s (attempt {attempt+1}/6)", flush=True)
                time.sleep(wait + 0.5)
                continue
            except Exception as e:
                if attempt == 5:
                    raise LLMError(str(e))
                time.sleep(1 + attempt)
                continue

            choice = resp.choices[0]
            self.last_finish_reason = choice.finish_reason
            text = THINK_RE.sub("", choice.message.content or "").strip()
            if not text:
                raise LLMError(f"empty content (finish_reason={choice.finish_reason})")
            return text
        raise LLMError("retries exhausted")