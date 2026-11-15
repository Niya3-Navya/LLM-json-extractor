"""Thin wrappers around the Gemini and OpenAI APIs.
Anything callable as  llm(prompt: str) -> str  works with extract.py."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


class GeminiClient:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from google import genai  # imported here so tests run without the SDK
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=self.temperature,
                # Ask for JSON, but we still clean + validate ourselves.
                response_mime_type="application/json",
            ),
        )
        return response.text or ""


class OpenAIClient:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from openai import OpenAI  # imported here so Gemini users don't need it

        self._client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  # prompt already says "JSON"
        )
        return response.choices[0].message.content or ""


def make_client(provider: str, model: str | None = None):
    if provider == "gemini":
        return GeminiClient(model=model)
    if provider == "openai":
        return OpenAIClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")
