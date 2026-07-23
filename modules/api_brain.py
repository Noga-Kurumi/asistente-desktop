"""Orquestador del cerebro del asistente.

Capa delgada entre la UI y el proveedor LLM:
- Mantiene el historial de conversación (thread-safe, formato agnóstico).
- Captura la pantalla del monitor principal con mss.
- Parsea acciones especiales del modelo (<PointTo, x, y>).
- Garantiza UNA query activa: contador de generación; si llega una nueva, la
  anterior deja de emitir resultados.
- Sin hilos creados a mano: process_query() es síncrono; submit_query() lo
  envuelve en un QRunnable sobre QThreadPool.globalInstance().

El proveedor se inyecta por constructor (factory por defecto), lo que permite
testear con un provider falso sin tocar la red.
"""

import io
import logging
import re
import threading
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from modules.config_manager import get_config
from modules.llm import (
    ERR_GENERIC,
    ERR_NO_API,
    LLMError,
    LLMProvider,
    create_provider,
    get_gemini_models,
)

logger = logging.getLogger(__name__)

POINT_PATTERN = re.compile(r"<PointTo,\s*(\d+),\s*(\d+)>")


class AssistantBrain(QObject):
    """Orquestador LLM. Mantiene las señales Qt que consume main.py."""

    text_chunk_ready = Signal(str)
    point_action_ready = Signal(int, int)
    thinking_started = Signal()
    thinking_finished = Signal()
    error_occurred = Signal(str)

    MAX_HISTORY = 10  # turnos (mensajes) conservados en memoria

    def __init__(self, provider: Optional[LLMProvider] = None, config=None):
        """Sin I/O de red: el provider inyectado ya viene configurado y su
        cliente se crea perezosamente en la primera query."""
        super().__init__()
        self.config = config or get_config()
        if provider is None:
            provider = create_provider(self.config.as_dict())
        self.provider = provider

        self._history_lock = threading.Lock()
        self._message_history: List[Dict[str, str]] = []

        # Contador de generación: solo la query más reciente puede emitir.
        self._generation_lock = threading.Lock()
        self._generation = 0

        # Captura de pantalla tomada al iniciar la grabación (turno actual).
        self._capture_lock = threading.Lock()
        self._captured_image: Optional[bytes] = None

        self.system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------ setup

    def _build_system_prompt(self) -> str:
        custom_inst = self.config.get("custom_instructions", "")
        avatar_name = self.config.get("avatar_name", "Lindsay")
        username = self.config.get("username", "") or "el usuario"
        return f"""
        Eres un asistente virtual de escritorio llamado {avatar_name} que ayuda a {username}.
        Actualmente puedes ver la pantalla principal del usuario.

        REGLAS ESTRICTAS:
        1. PERSONA: Habla íntegramente en español neutro. NO uses modismos ni jerga regional. Mantén un tono directo, natural y profesional.
        2. IDENTIDAD: Tu nombre es {avatar_name}. Puedes presentarte y referirte a ti mismo por ese nombre cuando sea apropiado.
        3. FORMATO: NO uses Markdown complejo (ni negritas ni bloques de código) porque tu respuesta será leída en voz alta por un motor TTS. Habla con naturalidad.

        INSTRUCCIONES PERSONALIZADAS DEL USUARIO:
        {custom_inst if custom_inst else "Ninguna."}
        """

    def get_gemini_models(self, api_key: str = ""):
        """Compatibilidad con setup.py: lista modelos Gemini disponibles."""
        return get_gemini_models(api_key or self.config.get("api_key", ""))

    # --------------------------------------------------------------- captura

    @staticmethod
    def _get_primary_monitor(sct):
        """Elige el monitor principal: el que empieza en (0,0). Fallback: el primero real."""
        real_monitors = sct.monitors[1:]  # monitors[0] es el bounding box combinado
        for mon in real_monitors:
            if mon["left"] == 0 and mon["top"] == 0:
                return mon
        return real_monitors[0]

    def capture_screen_bytes(self) -> Optional[bytes]:
        """Captura el monitor principal y devuelve JPEG comprimido en bytes."""
        try:
            from mss import mss
            from PIL import Image

            with mss() as sct:
                monitor = self._get_primary_monitor(sct)
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

                # Compresión a 720p (suficiente para el OCR interno del LLM)
                target_width = 1280
                if img.size[0] > target_width:
                    ratio = target_width / float(img.size[0])
                    target_height = int(float(img.size[1]) * ratio)
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=60)
                return buffer.getvalue()
        except Exception as e:
            logger.error("❌ [VISIÓN] Error al capturar pantalla: %s", e, exc_info=True)
            return None

    def capture_and_store_screen(self) -> None:
        """Captura la pantalla actual (bytes JPEG) y la guarda para el próximo query."""
        image = self.capture_screen_bytes()
        with self._capture_lock:
            self._captured_image = image

    # --------------------------------------------------------------- historial

    def _append_history(self, role: str, content: str) -> None:
        with self._history_lock:
            self._message_history.append({"role": role, "content": content})
            if len(self._message_history) > self.MAX_HISTORY:
                self._message_history = self._message_history[-self.MAX_HISTORY:]

    def _history_snapshot(self) -> List[Dict[str, str]]:
        with self._history_lock:
            return list(self._message_history)

    # ----------------------------------------------------------------- queries

    def _next_generation(self) -> int:
        with self._generation_lock:
            self._generation += 1
            return self._generation

    def _is_current(self, generation: int) -> bool:
        with self._generation_lock:
            return generation == self._generation

    def cancel_current(self) -> None:
        """Invalida la query en curso: sus emisiones se descartan (generación
        obsoleta). Se llama cuando el usuario inicia una grabación nueva a
        mitad de una respuesta (MVP: el turno viejo se abandona)."""
        self._next_generation()

    def submit_query(self, user_prompt: str) -> None:
        """Encola una query en el QThreadPool global (sin hilos a mano).

        Si había otra query en curso, sus emisiones se descartan (generación
        obsoleta). Pensada para ser llamada desde el hilo de la UI.
        """
        if not str(self.config.get("api_key", "") or "").strip():
            self.error_occurred.emit(ERR_NO_API)
            return
        worker = _BrainWorker(self, user_prompt)
        QThreadPool.globalInstance().start(worker)

    # Alias compatible con el main.py actual.
    process_query_async = submit_query

    def process_query(self, user_prompt: str) -> None:
        """Procesa una query de forma síncrona (ejecutar dentro de un QRunnable).

        Emite thinking_started / text_chunk_ready / point_action_ready /
        error_occurred / thinking_finished según avanza.
        """
        generation = self._next_generation()
        self.thinking_started.emit()

        with self._capture_lock:
            image_bytes = self._captured_image

        self._append_history("user", user_prompt)
        sentence_count = 0

        def emit_sentence(sentence: str) -> None:
            nonlocal sentence_count
            if not self._is_current(generation):
                return  # generación obsoleta: descartar emisiones
            point_match = POINT_PATTERN.search(sentence)
            if point_match:
                x, y = int(point_match.group(1)), int(point_match.group(2))
                self.point_action_ready.emit(x, y)
                # Quitar el tag y colapsar los espacios que deja.
                sentence = " ".join(POINT_PATTERN.sub("", sentence).split())
            if sentence:
                sentence_count += 1
                logger.info("🧠 [BRAIN] Oración %d del LLM: '%.60s'",
                            sentence_count, sentence)
                self.text_chunk_ready.emit(sentence)

        try:
            full_text = self.provider.stream_reply(
                messages=self._history_snapshot(),
                system_prompt=self.system_prompt,
                image_bytes=image_bytes,
                on_sentence=emit_sentence,
            )
            if full_text.strip():
                self._append_history("assistant", full_text.strip())
        except LLMError as e:
            logger.error("❌ [BRAIN] Error LLM (%s): %s", e.code, e, exc_info=True)
            if self._is_current(generation):
                self.error_occurred.emit(e.code)
        except Exception as e:
            logger.error("❌ [BRAIN] Error inesperado en la query: %s", e, exc_info=True)
            if self._is_current(generation):
                self.error_occurred.emit(ERR_GENERIC)
        finally:
            self.thinking_finished.emit()


class _BrainWorker(QRunnable):
    """QRunnable que ejecuta AssistantBrain.process_query en el pool global."""

    def __init__(self, brain: AssistantBrain, user_prompt: str):
        super().__init__()
        self._brain = brain
        self._prompt = user_prompt

    def run(self) -> None:
        try:
            self._brain.process_query(self._prompt)
        except Exception as e:
            # Última línea de defensa: process_query ya mapea sus errores;
            # si algo se escapa, que quede logueado y no muerda el pool.
            logger.error("❌ [BRAIN] Excepción no controlada en worker: %s", e, exc_info=True)
