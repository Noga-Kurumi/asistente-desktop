"""Proveedor LLM para Google Gemini (SDK google-genai).

Streaming con generate_content_stream: los chunks se agrupan en oraciones
(SentenceSplitter) para TTS y se acumula el texto completo para el historial.

El cliente se crea de forma perezosa en la primera llamada: ningún I/O de red
ocurre en __init__.
"""

import logging
import time
from typing import Callable, Dict, List, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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
    SentenceCallback,
    SentenceSplitter,
)

logger = logging.getLogger(__name__)

# Reintentos ante errores transitorios del servidor (429/503): hasta 3
# reintentos con backoff exponencial (1s, 2s, 4s). Corre en un worker de
# QThreadPool, así que el sleep no congela la UI.
MAX_RETRIES = 3
RETRY_BACKOFF_S = (1, 2, 4)
RETRYABLE_HTTP_CODES = frozenset({429, 503})

# Function calling (Fase D): rondas máximas tool→respuesta→tool por query.
MAX_TOOL_ROUNDS = 3

# Declaraciones de las tools del timeline (contrato con Gemini; la ejecución
# la hace tool_executor inyectado por api_brain — MCP del timeline).
TIMELINE_TOOLS = [
    types.FunctionDeclaration(
        name="search_timeline_by_keywords",
        description=(
            "Busca en el timeline de actividad del usuario (apps que usó, texto "
            "visible en pantalla vía OCR, portapapeles, audio de reuniones "
            "transcrito) por palabras clave. Usala cuando la consulta NO tenga "
            "una franja horaria clara."),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="Palabras clave a buscar (texto libre, en español)."),
                "limit": types.Schema(
                    type=types.Type.INTEGER,
                    description="Máximo de registros a devolver (default 10)."),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_timeline_by_time_range",
        description=(
            "Devuelve la actividad del usuario en una franja horaria concreta "
            "(qué hizo entre dos horas). Usala cuando la consulta mencione horas "
            "o franjas ('esta mañana', 'de 9 a 10')."),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "start_time": types.Schema(
                    type=types.Type.STRING,
                    description="Hora de inicio: 'HH:MM' (hoy) o ISO 'YYYY-MM-DDTHH:MM'."),
                "end_time": types.Schema(
                    type=types.Type.STRING,
                    description="Hora de fin: 'HH:MM' (hoy) o ISO 'YYYY-MM-DDTHH:MM'."),
                "limit": types.Schema(
                    type=types.Type.INTEGER,
                    description="Máximo de registros (default 50)."),
            },
            required=["start_time", "end_time"],
        ),
    ),
]


class GeminiProvider(LLMProvider):
    """Proveedor Gemini. Sin I/O de red en __init__ (cliente perezoso)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model = model
        self._client: Optional[genai.Client] = None
        # Function calling (Fase D): ejecutor (name, args) -> str inyectado por
        # api_brain (modules/timeline_mcp_server.call_timeline_tool). None =
        # sin tools, comportamiento idéntico al de siempre.
        self.tool_executor: Optional[Callable[[str, dict], str]] = None

    def _get_client(self) -> genai.Client:
        if not self._api_key or not self._api_key.strip():
            raise LLMError(ERR_NO_API, "Falta la API key de Gemini")
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def stream_reply(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        image_bytes: Optional[bytes],
        on_sentence: SentenceCallback,
    ) -> str:
        client = self._get_client()
        contents = self._build_contents(messages, image_bytes)

        splitter = SentenceSplitter()
        full_text_parts: List[str] = []

        # Bucle manual de function calling (google-genai 2.10 no trae AFC en
        # streaming): mientras el modelo emita function_call, se ejecuta la
        # tool y se reenvía la conversación; el stream de texto sigue fluyendo
        # por on_sentence en cada ronda. Acotado a MAX_TOOL_ROUNDS.
        for round_n in range(MAX_TOOL_ROUNDS + 1):
            tool_calls = self._stream_with_retry(
                client, contents, system_prompt, splitter, full_text_parts, on_sentence)
            if not tool_calls or self.tool_executor is None or round_n == MAX_TOOL_ROUNDS:
                break
            logger.info("🔧 [GEMINI] %d function call(s) solicitadas (ronda %d)",
                        len(tool_calls), round_n + 1)
            contents.append(types.Content(
                role="model",
                parts=[types.Part(function_call=fc) for fc in tool_calls]))
            response_parts = []
            for fc in tool_calls:
                args = dict(fc.args or {})
                logger.info("🔧 [GEMINI] Ejecutando tool %s(%s)", fc.name, args)
                result = self.tool_executor(fc.name, args)
                response_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=response_parts))

        for sentence in splitter.flush():
            on_sentence(sentence)
        return "".join(full_text_parts)

    def _stream_with_retry(
        self,
        client: genai.Client,
        contents: List[types.Content],
        system_prompt: str,
        splitter: SentenceSplitter,
        full_text_parts: List[str],
        on_sentence: SentenceCallback,
    ) -> List[types.FunctionCall]:
        """Una ronda de streaming con reintentos ante 429/503.

        Devuelve las function calls pedidas por el modelo en esta ronda
        (vacía si respondió solo texto). El flush del splitter lo hace
        stream_reply al final de TODAS las rondas.
        """
        config_kwargs = dict(
            system_instruction=system_prompt,
            max_output_tokens=1000,
            temperature=0.7,
        )
        if self.tool_executor is not None:
            config_kwargs["tools"] = [types.Tool(function_declarations=TIMELINE_TOOLS)]

        for attempt in range(MAX_RETRIES + 1):
            tool_calls: List[types.FunctionCall] = []
            try:
                stream = client.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                for chunk in stream:
                    tool_calls.extend(self._chunk_function_calls(chunk))
                    text = self._chunk_text(chunk)
                    if not text:
                        continue
                    full_text_parts.append(text)
                    for sentence in splitter.feed(text):
                        on_sentence(sentence)
                return tool_calls
            except LLMError:
                raise
            except Exception as e:
                # Solo se reintenta si el error es transitorio (429/503) y aún
                # no se emitió texto (reintentar a mitad de stream duplicaría
                # las oraciones mandadas al TTS).
                if (full_text_parts or not self._is_retryable(e)
                        or attempt == MAX_RETRIES):
                    raise self._map_exception(e) from e
                wait = RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "⚠️ [GEMINI] Error transitorio (HTTP %s), reintento %d/%d en %ds",
                    getattr(e, "code", "?"), attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """True si el error es transitorio (servidor saturado o rate limit)."""
        return (isinstance(exc, genai_errors.APIError)
                and getattr(exc, "code", None) in RETRYABLE_HTTP_CODES)

    @staticmethod
    def _chunk_text(chunk) -> str:
        """Texto de un chunk, ignorando parts no-texto (function_call)."""
        texts = []
        for cand in (getattr(chunk, "candidates", None) or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        return "".join(texts)

    @staticmethod
    def _chunk_function_calls(chunk) -> List["types.FunctionCall"]:
        """Function calls de un chunk (parts con function_call)."""
        calls = []
        for cand in (getattr(chunk, "candidates", None) or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    calls.append(fc)
        return calls

    @staticmethod
    def _build_contents(
        messages: List[Dict[str, str]], image_bytes: Optional[bytes]
    ) -> List[types.Content]:
        """Convierte el historial agnóstico a types.Content de Gemini.

        La imagen (captura del turno actual) se antepone al ÚLTIMO mensaje del
        usuario; nunca se reenvían capturas de turnos anteriores.
        """
        contents: List[types.Content] = []
        last_user_index = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=None,
        )
        for i, msg in enumerate(messages):
            role = "user" if msg.get("role") == "user" else "model"
            parts: List[types.Part] = []
            if i == last_user_index and image_bytes:
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            parts.append(types.Part(text=msg.get("content", "")))
            contents.append(types.Content(role=role, parts=parts))
        return contents

    @staticmethod
    def _map_exception(exc: Exception) -> LLMError:
        """Mapea excepciones del SDK a códigos ERR_* estables (sin string-matching).

        google-genai lanza google.genai.errors.APIError con .code (HTTP status)
        tipado: 401/403 → auth, 429 → cuota, 503 → no disponible, 5xx restante
        → genérico. Los errores de transporte (httpx, usado por el SDK) se
        mapean por tipo.
        """
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code in (401, 403):
                return LLMError(ERR_AUTH, str(exc))
            if code == 429:
                return LLMError(ERR_QUOTA, str(exc))
            if code == 408:
                return LLMError(ERR_TIMEOUT, str(exc))
            if code == 503:
                return LLMError(ERR_UNAVAILABLE, str(exc))
            return LLMError(ERR_GENERIC, f"APIError {code}: {exc}")

        # Errores de transporte por tipo (httpx es dependencia de google-genai).
        try:
            import httpx

            if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
                return LLMError(ERR_NETWORK, str(exc))
            if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                return LLMError(ERR_TIMEOUT, str(exc))
        except ImportError:  # pragma: no cover - httpx siempre viene con google-genai
            pass

        if isinstance(exc, (ConnectionError, TimeoutError)):
            return LLMError(ERR_NETWORK, str(exc))

        logger.error("❌ [GEMINI] Excepción no mapeada: %r", exc, exc_info=True)
        return LLMError(ERR_GENERIC, str(exc))


def get_gemini_models(api_key: str) -> List[Dict[str, str]]:
    """Lista los modelos Gemini aptos para generateContent (familia Flash estable).

    Función de módulo: no requiere instanciar el proveedor. Devuelve [] si
    falla (error logueado).
    """
    if not api_key:
        return []

    try:
        client = genai.Client(api_key=api_key)
        available: List[Dict[str, str]] = []

        for model in client.models.list():
            model_name = model.name or ""
            if model_name.startswith("models/"):
                model_name = model_name[len("models/"):]
            name_lower = model_name.lower()

            actions = [a.lower() for a in (model.supported_actions or [])]
            if "generatecontent" not in actions:
                continue
            if "gemini" not in name_lower or "flash" not in name_lower:
                continue
            if "-exp" in name_lower or "preview" in name_lower or "experimental" in name_lower:
                continue
            if "-image" in name_lower or "-audio" in name_lower:
                continue

            available.append({
                "name": model_name,
                "display_name": getattr(model, "display_name", None) or model_name,
            })

        available.sort(key=lambda x: x["name"])
        logger.info("🔍 [GEMINI] Modelos disponibles: %d", len(available))
        return available
    except Exception as e:
        logger.error("❌ [GEMINI] Error obteniendo modelos: %s", e, exc_info=True)
        return []
