"""Rate limiting / retry behaviour of the Gemini wrapper (no network: the client is faked)."""
import sys
import types

import pytest

import backend.llm as llm
from backend.llm import GeminiClient, LLMConfigError, LLMUnavailableError, RateLimiter, _retry_after


def test_retry_after_parses_both_formats():
    assert _retry_after("Please retry in 6.9s.") == pytest.approx(6.9)
    assert _retry_after("{'retryDelay': '6s'}") == 6.0
    assert _retry_after("nothing here") is None


def test_rate_limiter_spaces_calls(monkeypatch):
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    r = RateLimiter(rpm=60)  # one call per second
    for _ in range(3):
        r.wait()
    # the clock doesn't move while sleep is mocked, so waits stack up: ~1s, then ~2s
    assert len(slept) == 2 and 0.9 < slept[0] <= 1.0 and 1.9 < slept[1] <= 2.0


class _Resp:
    text = "OK"


def _fake_client(errors):
    calls = {"n": 0}

    def generate_content(model, contents, config):
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return _Resp()

    c = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    return c, calls


@pytest.fixture()
def fake_genai(monkeypatch):
    types_mod = types.SimpleNamespace(
        GenerateContentConfig=lambda **k: k, Part=types.SimpleNamespace(from_bytes=lambda **k: k))
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.types = types_mod
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    monkeypatch.setattr(llm, "_limiter", RateLimiter(0))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_429_waits_and_succeeds(fake_genai):
    g = GeminiClient(api_key="k")
    g._client, calls = _fake_client([Exception("429 RESOURCE_EXHAUSTED Please retry in 3s")] * 4)
    assert g.generate("hi") == "OK" and calls["n"] == 5  # rate-limit retries don't use normal attempts


def test_daily_quota_gives_up(fake_genai):
    g = GeminiClient(api_key="k")
    g._client, _ = _fake_client([Exception("429 RESOURCE_EXHAUSTED Please retry in 3600s")])
    with pytest.raises(LLMUnavailableError, match="quota"):
        g.generate("hi")


def test_bad_key_fails_fast(fake_genai):
    g = GeminiClient(api_key="k")
    g._client, calls = _fake_client([Exception("400 API_KEY_INVALID API key not valid")])
    with pytest.raises(LLMConfigError):
        g.generate("hi")
    assert calls["n"] == 1
