"""Turn a raw LLM reply into a Python dict.

LLMs often wrap JSON in markdown fences, add chatter before/after it, or leave
trailing commas. This module handles those cases. It does NOT validate the
structure -- that is Pydantic's job (see schemas.py).
"""
import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class JSONExtractionError(ValueError):
    """Raised when no valid JSON object can be recovered from the reply."""


def extract_json(raw: str) -> dict:
    if not raw or not raw.strip():
        raise JSONExtractionError("The reply was empty.")

    text = raw.strip()

    # 1. Prefer the contents of a ```json ... ``` fence if present.
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()

    # 2. Keep only the outermost {...} (drops "Here is the JSON:" chatter).
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JSONExtractionError("No JSON object ({...}) found in the reply.")
    candidate = text[start : end + 1]

    # 3. Parse; on failure try once more after removing trailing commas.
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            data = json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
        except json.JSONDecodeError as exc:
            raise JSONExtractionError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise JSONExtractionError("Top-level JSON value must be an object.")
    return data
