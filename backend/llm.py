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
import threading
import time
from pathlib import Path
from typing import Any

from config import settings

log = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The provider could not be reached / no key configured. Message is user-safe."""


class LLMConfigError(LLMUnavailableError):
    """Wrong key or model: retrying cannot help until .env is fixed."""


class RateLimiter:
    """Keeps calls under N requests per minute (shared by every call in this process)."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)

    def pause(self, seconds: float) -> None:
        """Push every later call back (after the server told us to slow down)."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


_RETRY_RE = re.compile(r"retry in ([0-9.]+)\s*s|retryDelay'?\"?:\s*'?\"?([0-9.]+)s", re.IGNORECASE)


def _retry_after(msg: str) -> float | None:
    m = _RETRY_RE.search(msg)
    if not m:
        return None
    return float(m.group(1) or m.group(2))


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
        attempt, rate_limited = 0, 0
        while attempt < settings.llm_max_retries:
            _limiter.wait()
            try:
                resp = client.models.generate_content(model=model or self.model, contents=contents, config=cfg)
                text = getattr(resp, "text", None)
                if not text:
                    raise LLMOutputError("empty response (possibly blocked by safety filters)")
                return text
            except LLMOutputError:
                raise
            except Exception as exc:  # network / quota / 5xx
                msg = str(exc)
                if "API_KEY_INVALID" in msg or "API key not valid" in msg or "PERMISSION_DENIED" in msg:
                    # retrying cannot fix a wrong key -> fail fast with a clear, safe message
                    log.error("Gemini rejected the API key; check GEMINI_API_KEY in .env")
                    raise LLMConfigError(
                        "The Gemini API key was rejected. Check GEMINI_API_KEY in your .env file."
                    ) from None
                if " 404 " in msg or "NOT_FOUND" in msg:
                    log.error("Gemini model %s not found; check GEMINI_MODEL in .env", model or self.model)
                    raise LLMConfigError(
                        "The configured Gemini model was not found. Check GEMINI_MODEL in your .env file."
                    ) from None
                last_exc = exc
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    # rate limit: wait as long as the server asks, without using up normal retries
                    wait = (_retry_after(msg) or 10.0) + 1.0
                    rate_limited += 1
                    if wait > settings.llm_max_wait_s or rate_limited > 6:
                        log.error("Gemini quota exhausted (asked to wait %.0fs); giving up for now", wait)
                        raise LLMUnavailableError(
                            "The Gemini quota is used up for now (free-tier limit). Wait and try again, "
                            "or use a paid key / another model."
                        ) from None
                    log.warning("Gemini rate limit hit; waiting %.0fs as requested", wait)
                    _limiter.pause(wait)
                    continue
                wait = 2 ** attempt
                attempt += 1
                log.warning("Gemini call failed (attempt %s): %s; retrying in %ss", attempt, type(exc).__name__, wait)
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


_limiter = RateLimiter(settings.llm_rpm)
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
