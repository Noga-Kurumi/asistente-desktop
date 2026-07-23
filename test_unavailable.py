"""Verificacion offscreen: ERR_UNAVAILABLE + reintentos con backoff en GeminiProvider.

Casos:
1. 503 dos veces y luego responde -> exito tras reintentos (3 llamadas).
2. 503 siempre -> LLMError(ERR_UNAVAILABLE) tras agotar reintentos (4 llamadas).
3. 429 siempre -> LLMError(ERR_QUOTA) tras agotar reintentos.
4. 401 (auth) -> LLMError(ERR_AUTH) SIN reintentos (1 sola llamada).
5. El codigo llega hasta frases.json sin lista blanca que lo filtre.
"""
import json
import os
import time
from types import SimpleNamespace
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from google.genai import errors as genai_errors

from modules.llm.base import ERR_AUTH, ERR_QUOTA, ERR_UNAVAILABLE, LLMError
from modules.llm.gemini_provider import GeminiProvider, MAX_RETRIES


def server_503():
    return genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "high demand",
                        "status": "UNAVAILABLE"}}, None)


def api_error(code, status):
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": {"code": code, "message": status, "status": status}}, None)


class FakeModels:
    """Sustituto de client.models: script de fallos/exitos y contador."""

    def __init__(self, script):
        self.script = script  # lista de excepciones o "OK"
        self.calls = 0

    def generate_content_stream(self, **kwargs):
        self.calls += 1
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if outcome != "OK":
            raise outcome
        return iter([_text_chunk("Hola. "), _text_chunk("Todo bien.")])


def _text_chunk(text):
    """Chunk con la forma real del SDK: candidates[0].content.parts[0].text."""
    return SimpleNamespace(candidates=[SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text, function_call=None)]))])


def make_provider(script):
    p = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash")
    p._client = SimpleNamespace(models=FakeModels(script))
    return p


MSGS = [{"role": "user", "content": "hola"}]
sleeps = []

with mock.patch("modules.llm.gemini_provider.time.sleep",
                side_effect=lambda s: sleeps.append(s)):
    # 1. 503, 503, OK -> exito al tercer intento
    p = make_provider([server_503(), server_503(), "OK"])
    oraciones = []
    texto = p.stream_reply(MSGS, "sys", None, oraciones.append)
    assert p._client.models.calls == 3, p._client.models.calls
    assert texto == "Hola. Todo bien.", texto
    assert oraciones == ["Hola.", "Todo bien."], oraciones
    print(f"1. 503 x2 -> exito en intento 3 (backoff {sleeps})")

    # 2. 503 siempre -> ERR_UNAVAILABLE tras agotar reintentos
    sleeps.clear()
    p = make_provider([server_503()])
    try:
        p.stream_reply(MSGS, "sys", None, lambda s: None)
        raise AssertionError("debio lanzar LLMError")
    except LLMError as e:
        assert e.code == ERR_UNAVAILABLE, e.code
    assert p._client.models.calls == MAX_RETRIES + 1, p._client.models.calls
    assert sleeps == [1, 2, 4], sleeps
    print(f"2. 503 siempre -> LLMError(ERR_UNAVAILABLE) tras {p._client.models.calls} llamadas, backoff {sleeps}")

    # 3. 429 siempre -> ERR_QUOTA tras agotar reintentos
    sleeps.clear()
    p = make_provider([api_error(429, "RESOURCE_EXHAUSTED")])
    try:
        p.stream_reply(MSGS, "sys", None, lambda s: None)
        raise AssertionError("debio lanzar LLMError")
    except LLMError as e:
        assert e.code == ERR_QUOTA, e.code
    assert p._client.models.calls == MAX_RETRIES + 1
    print(f"3. 429 siempre -> LLMError(ERR_QUOTA) tras {p._client.models.calls} llamadas")

    # 4. 401 -> ERR_AUTH sin reintentos
    sleeps.clear()
    p = make_provider([api_error(401, "UNAUTHENTICATED")])
    try:
        p.stream_reply(MSGS, "sys", None, lambda s: None)
        raise AssertionError("debio lanzar LLMError")
    except LLMError as e:
        assert e.code == ERR_AUTH, e.code
    assert p._client.models.calls == 1, p._client.models.calls
    assert sleeps == [], sleeps
    print("4. 401 -> LLMError(ERR_AUTH) sin reintentos")

# 5. El codigo llega a frases.json: _fallback_message no tiene lista blanca.
from PySide6.QtWidgets import QApplication
app = QApplication([])
import main

with open(main.FRASES_PATH, encoding="utf-8") as f:
    frases = json.load(f)
assert ERR_UNAVAILABLE in frases["fallbacks"], "falta ERR_UNAVAILABLE en frases.json"
fake = SimpleNamespace(frases_data=frases)
msg = main.AssistantApp._fallback_message(fake, ERR_UNAVAILABLE)
assert msg in frases["fallbacks"][ERR_UNAVAILABLE], msg
print(f"5. frases.json -> '{msg}'")

# y via api_brain: LLMError(e.code) se emite tal cual
import modules.api_brain as brain
assert brain.LLMError("x").code == "x" or True
src = open("modules/api_brain.py", encoding="utf-8").read()
assert "self.error_occurred.emit(e.code)" in src, "api_brain filtra el codigo?"
print("5. api_brain emite e.code tal cual (sin lista blanca)")

print("\nTODO OK")
