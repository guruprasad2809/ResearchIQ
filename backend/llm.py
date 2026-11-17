"""Thin, defensive wrapper around Gemini (google-genai SDK).

- API key only from the environment.
- Retries with exponential backoff on transient errors.
- JSON mode with one repair attempt; malformed output never crashes the app.
- The model's output is only ever parsed as data. It is never executed.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from config import settings

log = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The provider could not be reached / no key configured. Message is user-safe."""


class LLMOutputError(RuntimeError):
    """The model answered, but not in the expected structure."""


def extract_json(text: str) -> Any:
    """Parse JSON even when wrapped in ```json fences or surrounded by stray text."""
    if text is None:
        raise LLMOutputError("empty output")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = t.find(open_c), t.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(t[s : e + 1])
            except json.JSONDecodeError:
                continue
    raise LLMOutputError("model output was not valid JSON")


class GeminiClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.gemini_model
        self._api_key = api_key if api_key is not None else settings.gemini_api_key
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _get(self):
        if not self._api_key:
            raise LLMUnavailableError("The language model is not configured (GEMINI_API_KEY is missing).")
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = "",
        images: list[str | Path] | None = None,
        json_mode: bool = False,
        temperature: float = 0.1,
        model: str | None = None,
    ) -> str:
        from google.genai import types

        client = self._get()
        contents: list[Any] = []
        for img in images or []:
            p = Path(img)
            if p.exists():
                contents.append(types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png"))
        contents.append(prompt)
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        last_exc: Exception | None = None
        for attempt in range(settings.llm_max_retries):
            try:
                resp = client.models.generate_content(model=model or self.model, contents=contents, config=cfg)
                text = getattr(resp, "text", None)
                if not text:
                    raise LLMOutputError("empty response (possibly blocked by safety filters)")
                return text
            except LLMOutputError:
                raise
            except Exception as exc:  # network / quota / 5xx
                last_exc = exc
                wait = 2 ** attempt
                log.warning("Gemini call failed (attempt %s): %s; retrying in %ss", attempt + 1, type(exc).__name__, wait)
                time.sleep(wait)
        log.error("Gemini unavailable after retries: %s", last_exc)
        raise LLMUnavailableError("The language model is temporarily unavailable. Please try again.")

    def generate_json(self, prompt: str, system: str = "", images=None, model: str | None = None) -> Any:
        raw = self.generate(prompt, system=system, images=images, json_mode=True, model=model)
        try:
            return extract_json(raw)
        except LLMOutputError:
            repair = (
                "The following text was supposed to be a single valid JSON value but is malformed. "
                "Return ONLY the corrected JSON, with no commentary.\n\n" + raw[:12000]
            )
            return extract_json(self.generate(repair, json_mode=True, model=model))


_default: GeminiClient | None = None


def get_llm() -> GeminiClient:
    global _default
    if _default is None:
        _default = GeminiClient()
    return _default


def set_llm(client) -> None:
    """Used by tests to inject a fake client."""
    global _default
    _default = client
