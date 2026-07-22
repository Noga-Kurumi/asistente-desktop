"""Contrato base para proveedores LLM del asistente.

Un proveedor sabe hablar con una API concreta (Gemini, ...) y entrega la
respuesta en streaming, agrupada en oraciones listas para TTS. No conoce Qt,
ni historial de conversación, ni capturas de pantalla: eso es responsabilidad
del orquestador (modules/api_brain.py).

Formato de mensajes (agnóstico de proveedor):
    [{"role": "user" | "assistant", "content": "texto"}, ...]
Las imágenes NO viajan en el historial: solo la captura del turno actual se
pasa como image_bytes a stream_reply().
"""

import re
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Optional

# Códigos de error consumidos por main.py → frases.json (fallbacks).
ERR_NO_API = "ERR_NO_API"
ERR_AUTH = "ERR_AUTH"
ERR_QUOTA = "ERR_QUOTA"
ERR_NETWORK = "ERR_NETWORK"
ERR_TIMEOUT = "ERR_TIMEOUT"
ERR_UNAVAILABLE = "ERR_UNAVAILABLE"  # servicio saturado/no disponible temporalmente
ERR_GENERIC = "ERR_GENERIC"

# Callback que recibe cada oración completa, lista para sintetizar.
SentenceCallback = Callable[[str], None]

# Fin de oración: ., ?, ! seguidos de espacio o fin de texto.
SENTENCE_END_RE = re.compile(r"[.?!…]+(?:\s|$)")


class LLMError(Exception):
    """Error de un proveedor LLM, ya mapeado a un código estable.

    Attributes:
        code: Uno de los códigos ERR_* definidos arriba.
    """

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


class SentenceSplitter:
    """Agrupa chunks de streaming en oraciones completas para TTS.

    Uso: feed(chunk) por cada trozo; devuelve una lista de oraciones cerradas
    (posiblemente vacía). Al terminar el stream, flush() devuelve el resto.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> List[str]:
        self._buffer += chunk
        sentences: List[str] = []
        start = 0
        for match in SENTENCE_END_RE.finditer(self._buffer):
            sentence = self._buffer[start:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> List[str]:
        remainder = self._buffer.strip()
        self._buffer = ""
        return [remainder] if remainder else []


class LLMProvider(ABC):
    """Interfaz que debe implementar cada proveedor LLM."""

    #: Nombre corto del proveedor (logs, diagnóstico).
    name: str = "abstract"

    @abstractmethod
    def stream_reply(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        image_bytes: Optional[bytes],
        on_sentence: SentenceCallback,
    ) -> str:
        """Genera la respuesta en streaming.

        Args:
            messages: Historial de la conversación (sin la imagen actual).
            system_prompt: Prompt de sistema en español.
            image_bytes: JPEG de la captura del turno actual, o None.
            on_sentence: Callback invocado con cada oración completa, en orden.

        Returns:
            El texto completo de la respuesta (para guardar en el historial).

        Raises:
            LLMError: Con el código ERR_* correspondiente. Las excepciones del
                SDK deben mapearse aquí dentro; nunca propagarse crudas.
        """
