"""Proveedores LLM del asistente (interfaz modular e inyectable)."""

from modules.llm.base import (
    ERR_AUTH,
    ERR_GENERIC,
    ERR_NETWORK,
    ERR_NO_API,
    ERR_QUOTA,
    ERR_TIMEOUT,
    ERR_UNAVAILABLE,
    LLMError,
    LLMProvider,
    SentenceSplitter,
)
from modules.llm.factory import create_provider
from modules.llm.gemini_provider import GeminiProvider, get_gemini_models

__all__ = [
    "ERR_AUTH",
    "ERR_GENERIC",
    "ERR_NETWORK",
    "ERR_NO_API",
    "ERR_QUOTA",
    "ERR_TIMEOUT",
    "ERR_UNAVAILABLE",
    "LLMError",
    "LLMProvider",
    "SentenceSplitter",
    "create_provider",
    "GeminiProvider",
    "get_gemini_models",
]
