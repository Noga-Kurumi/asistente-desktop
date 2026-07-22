"""Factory de proveedores LLM.

create_provider(config) instancia el proveedor indicado por
config["api_provider"]. Para añadir un proveedor nuevo: crear su clase en
modules/llm/ y registrarla en _PROVIDERS con un builder.
"""

import logging
from typing import Any, Mapping

from modules.llm.base import LLMProvider
from modules.llm.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)


def _build_gemini(config: Mapping[str, Any]) -> GeminiProvider:
    return GeminiProvider(
        api_key=str(config.get("api_key", "")),
        model=str(config.get("gemini_model", "gemini-2.0-flash")),
    )


# Registro de proveedores disponibles: nombre → builder(config).
_PROVIDERS = {
    "gemini": _build_gemini,
}


def create_provider(config: Mapping[str, Any]) -> LLMProvider:
    """Crea el proveedor LLM según la configuración.

    Args:
        config: Mapping con al menos api_provider, api_key y las claves
            específicas del proveedor (p.ej. gemini_model).

    Raises:
        ValueError: Si api_provider no está soportado.
    """
    name = str(config.get("api_provider", "gemini")).strip().lower()
    builder = _PROVIDERS.get(name)
    if builder is None:
        soportados = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Proveedor LLM no soportado: '{name}'. Soportados: {soportados}")
    provider = builder(config)
    logger.info("🧠 [LLM] Proveedor creado: %s", provider.name)
    return provider
