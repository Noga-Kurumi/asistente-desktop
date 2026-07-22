"""Núcleo de audio del asistente: transcripción final + streaming live.

- process_live_input() alimenta whisper-stream-pcm.exe (streaming live).
- process_voice_input() lanza la transcripción final en el QThreadPool global
  (sin hilos creados a mano).
- Si el modelo whisper no existe en disco NO se lanza FileNotFoundError: se
  emite model_missing(path) para que el caller lo descargue (model_manager).

Nota de coordinación: stop_live_transcription() es bloqueante —devuelve solo
cuando el subproceso de streaming ha terminado—, así que el caller puede
llamar a process_voice_input() inmediatamente después, sin el antiguo
QTimer.singleShot(100, ...) mágico que esperaba a que el pipe se vaciara.
"""

import logging
import os
import threading

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from modules.config_manager import get_config
from modules.whisper_wrapper import WhisperCppWrapper

logger = logging.getLogger(__name__)


def whisper_model_path(config) -> str:
    """Resuelve la ruta del modelo ggml según whisper_model/whisper_quantization."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    whisper_model = config.get("whisper_model", "tiny")
    whisper_quant = config.get("whisper_quantization", "q5_1")
    if whisper_quant in ("none", "", None):
        model_filename = f"ggml-{whisper_model}.bin"
    else:
        model_filename = f"ggml-{whisper_model}-{whisper_quant}.bin"
    return os.path.join(base_dir, "models", model_filename)


class _TranscribeRunnable(QRunnable):
    """Ejecuta la transcripción final en el pool global."""

    def __init__(self, core: "AssistantAudioCore", audio_array: np.ndarray):
        super().__init__()
        self._core = core
        self._audio = audio_array

    def run(self) -> None:
        self._core._run_final_transcription(self._audio)


class AssistantAudioCore(QObject):
    text_transcribed = Signal(str)
    live_text_ready = Signal(str)
    # Emitida cuando el modelo ggml no existe en disco (path esperado).
    model_missing = Signal(str)

    def __init__(self, config=None):
        super().__init__()
        self.config = config or get_config()

        # Protege is_live_transcribing y la secuencia start/stop del streaming.
        self._live_lock = threading.Lock()
        self._is_live_transcribing = False

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stream_exe_path = os.path.join(base_dir, "whisper_cpp", "Release", "whisper-stream-pcm.exe")
        model_path = whisper_model_path(self.config)
        n_threads = int(self.config.get("whisper_threads", 4) or 4)

        self.whisper = WhisperCppWrapper(
            stream_exe_path=stream_exe_path,
            model_path=model_path,
            n_threads=n_threads,
        )

        if not self.whisper.model_available:
            logger.error("❌ [AUDIO_CORE] Modelo whisper ausente: %s", model_path)
            # Emitimos en diferido: los connect() del caller aún no existen en
            # __init__. El caller puede consultar model_available al terminar
            # de conectar, o reconectar esta señal vía ensure_models().
            self._pending_model_missing = model_path
        else:
            self._pending_model_missing = None

        logger.info("✅ [AUDIO_CORE] Modelo whisper: %s (threads=%d)",
                    os.path.basename(model_path), n_threads)

    @property
    def model_available(self) -> bool:
        return self.whisper.model_available

    def notify_model_missing(self) -> None:
        """Emite model_missing si el modelo faltaba al construir el core.

        Pensada para que el caller la invoque tras conectar las señales.
        """
        if self._pending_model_missing:
            path, self._pending_model_missing = self._pending_model_missing, None
            self.model_missing.emit(path)

    # ----------------------------------------------------- transcripción final

    def process_voice_input(self, audio_array: np.ndarray) -> None:
        """Encola la transcripción final en el QThreadPool global."""
        logger.info("🎯 [AUDIO_CORE] Transcripción final solicitada: %d samples",
                    len(audio_array))
        QThreadPool.globalInstance().start(_TranscribeRunnable(self, audio_array))

    def _run_final_transcription(self, audio_array: np.ndarray) -> None:
        try:
            if not self.whisper.model_available:
                logger.error("❌ [AUDIO_CORE] Sin modelo whisper, no se puede transcribir")
                self.model_missing.emit(self.whisper.model_path)
                self.text_transcribed.emit("")
                return
            success, text = self.whisper.transcribe(audio_array)
            if success and text:
                logger.info("✅ [AUDIO_CORE] Transcripción final: '%s'", text)
                self.text_transcribed.emit(text)
            else:
                logger.warning("⚠️ [AUDIO_CORE] Transcripción final falló o vacía")
                self.text_transcribed.emit("")
        except Exception as e:
            logger.error("❌ [AUDIO_CORE] Error en transcripción final: %s", e, exc_info=True)
            self.text_transcribed.emit("")

    # ------------------------------------------------------------ streaming live

    def process_live_input(self, audio_array: np.ndarray) -> None:
        """Alimenta el streaming live; arranca el subproceso en el primer chunk."""
        with self._live_lock:
            if not self._is_live_transcribing:
                if not self.whisper.model_available:
                    # Sin modelo no hay streaming; se avisa una sola vez.
                    self.notify_model_missing()
                    return
                if not self.whisper.start_streaming(self._stream_callback):
                    logger.error("❌ [AUDIO_CORE] No se pudo iniciar streaming")
                    return
                self._is_live_transcribing = True
            self.whisper.send_audio_chunk(audio_array)

    def _stream_callback(self, text: str) -> None:
        self.live_text_ready.emit(text)

    def stop_live_transcription(self) -> None:
        """Detiene el streaming. Bloqueante: al volver, el subproceso terminó."""
        with self._live_lock:
            if not self._is_live_transcribing:
                return
            self._is_live_transcribing = False
            self.whisper.stop_streaming()
