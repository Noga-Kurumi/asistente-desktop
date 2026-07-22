"""Wrapper para whisper.cpp.

- Transcripción final: pywhispercpp (bindings en-proceso). El modelo se carga
  de forma PEREZOSA en la primera transcripción final (no en __init__):
  durante el streaming live el audio lo procesa el subproceso
  whisper-stream-pcm.exe, así que tener además el modelo pywhispercpp en RAM
  mientras se graba sería memoria desperdiciada. Trade-off: la primera
  transcripción final de la sesión paga el coste de carga (~1-2s).
- Transcripción live: whisper-stream-pcm.exe alimentado por stdin (PCM f32).
  stderr del subproceso se drena en un hilo dedicado para no llenar el pipe
  (64KB en Windows) y bloquear al hijo.
- Idioma fijo en español ('es') tanto en streaming como en transcripción
  final: el asistente es monolingüe y la detección automática por ventana
  añadía latencia e inconsistencias entre live y final.
"""

import atexit
import logging
import os
import subprocess
import threading
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Idioma fijo del asistente (streaming y transcripción final).
LANGUAGE = "es"

# Marcas no-habla que whisper emite entre corchetes y deben filtrarse.
# Configurable a nivel de módulo: asignar un nuevo set para cambiar el filtro.
FILTERED_MARKERS = frozenset({
    "[MÚSICA]", "[MUSIC]", "[BLANK_AUDIO]", "[SILENCIO]", "[SILENCE]",
    "[APPLAUSE]", "[APLAUSOS]", "[RISAS]", "[LAUGHTER]",
})

# Timeout de parada del subproceso (también en atexit: corto para no colgar
# el cierre del intérprete).
_STOP_TIMEOUT = 2.0


class WhisperCppWrapper:
    """Wrapper de whisper.cpp: transcripción final en-proceso + streaming nativo."""

    def __init__(self, stream_exe_path: str, model_path: str, n_threads: int = 4):
        """
        Args:
            stream_exe_path: Ruta al ejecutable whisper-stream-pcm.exe.
            model_path: Ruta al modelo ggml-*.bin (no se carga hasta la primera
                transcripción final; ver docstring del módulo).
            n_threads: Número de threads para whisper.
        """
        self.stream_pcm_exe = stream_exe_path
        self.model_path = model_path
        self.n_threads = n_threads

        # El modelo ausente NO es excepción: se marca y el caller decide
        # (p.ej. descargarlo vía modules/model_manager.py).
        self.model_available = os.path.exists(model_path)
        if not self.model_available:
            logger.error("❌ [WHISPER_WRAPPER] Modelo no encontrado: %s", model_path)

        # Modelo pywhispercpp perezoso (ver docstring del módulo).
        self._model = None
        self._model_lock = threading.Lock()
        self._model_load_failed = False

        # Estado del streaming, protegido por _stream_lock de punta a punta.
        self._stream_lock = threading.Lock()
        self.stream_process: Optional[subprocess.Popen] = None
        self._stream_callback: Optional[Callable[[str], None]] = None
        self.stream_running = False

        # Parada segura al salir del intérprete (timeout corto, sin colgar).
        atexit.register(self._atexit_stop)

    # ----------------------------------------------------- transcripción final

    def _get_model(self):
        """Carga perezosa del modelo pywhispercpp (thread-safe). None si falla."""
        if self._model is not None or self._model_load_failed:
            return self._model
        with self._model_lock:
            if self._model is not None or self._model_load_failed:
                return self._model
            if not self.model_available:
                self._model_load_failed = True
                return None
            try:
                from pywhispercpp.model import Model

                self._model = Model(
                    self.model_path,
                    n_threads=self.n_threads,
                    language=LANGUAGE,
                    print_realtime=False,
                    print_progress=False,
                    print_timestamps=False,
                )
                logger.info("✅ [WHISPER_WRAPPER] Modelo cargado (perezoso): %s", self.model_path)
            except Exception as e:
                logger.error("❌ [WHISPER_WRAPPER] Error cargando modelo: %s", e, exc_info=True)
                self._model_load_failed = True
        return self._model

    def transcribe(self, audio_array: np.ndarray, timeout: int = 30) -> Tuple[bool, str]:
        """Transcribe audio float32 16kHz con el modelo persistente.

        Returns:
            Tupla (success, text). (False, "") si el modelo falta o falla.
        """
        model = self._get_model()
        if model is None:
            return False, ""
        try:
            audio = np.ascontiguousarray(audio_array, dtype=np.float32)
            segments = model.transcribe(audio)
            text = " ".join(seg.text for seg in segments).strip()
            logger.info("✅ [WHISPER_WRAPPER] Transcripción completada: '%s'", text)
            return True, text
        except Exception as e:
            logger.error("❌ [WHISPER_WRAPPER] Error en transcripción: %s", e, exc_info=True)
            return False, ""

    # ---------------------------------------------------------------- streaming

    def start_streaming(self, callback: Callable[[str], None]) -> bool:
        """Inicia el streaming real con whisper-stream-pcm.exe."""
        with self._stream_lock:
            if self.stream_running:
                logger.warning("⚠️ [WHISPER_WRAPPER] Streaming ya está en curso")
                return False
            return self._start_streaming_locked(callback)

    def _start_streaming_locked(self, callback: Callable[[str], None]) -> bool:
        """Arranque real del subproceso. REQUIERE tener _stream_lock."""
        if not os.path.exists(self.stream_pcm_exe):
            logger.error("❌ [WHISPER_WRAPPER] whisper-stream-pcm.exe no encontrado: %s",
                         self.stream_pcm_exe)
            return False

        cmd = [
            self.stream_pcm_exe,
            "-m", self.model_path,
            "-l", LANGUAGE,
            "-t", str(self.n_threads),
            "-i", "-",  # stdin
            "--format", "f32",
            "--sample-rate", "16000",
            "--vad",
            "--step", "200",   # 200ms: segmentos frecuentes
            "--length", "2000"  # 2000ms: menor delay
        ]

        try:
            logger.info("🚀 [WHISPER_WRAPPER] Iniciando streaming: %s", " ".join(cmd))
            self.stream_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._stream_callback = callback

            # Hilo que lee stdout y despacha segmentos al callback.
            threading.Thread(
                target=self._read_stream_output,
                args=(self.stream_process, callback),
                daemon=True,
                name="whisper-stream-stdout",
            ).start()
            # Hilo que drena stderr: si nadie lo consume, el pipe (64KB en
            # Windows) se llena y el subproceso se bloquea escribiendo.
            threading.Thread(
                target=self._drain_stream_stderr,
                args=(self.stream_process,),
                daemon=True,
                name="whisper-stream-stderr",
            ).start()

            self.stream_running = True
            logger.info("✅ [WHISPER_WRAPPER] Streaming iniciado correctamente")
            return True
        except Exception as e:
            logger.error("❌ [WHISPER_WRAPPER] Error iniciando streaming: %s", e, exc_info=True)
            self.stream_process = None
            self.stream_running = False
            return False

    def _read_stream_output(self, proc: subprocess.Popen, callback: Callable[[str], None]) -> None:
        """Hilo: lee stdout del proceso y llama al callback con cada segmento."""
        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # Formato: [00:00:00.000 --> 00:00:05.000]   texto
                if "-->" in line and line.startswith("["):
                    parts = line.split("]")
                    if len(parts) > 1:
                        text = parts[1].strip()
                        if text and text.upper() not in {m.upper() for m in FILTERED_MARKERS}:
                            logger.info("📝 [WHISPER_WRAPPER] Segmento streaming: '%s'", text)
                            callback(text)
        except Exception as e:
            logger.error("❌ [WHISPER_WRAPPER] Error leyendo stream output: %s", e, exc_info=True)
        finally:
            logger.info("🛑 [WHISPER_WRAPPER] stdout del streaming cerrado (proceso terminado)")

    def _drain_stream_stderr(self, proc: subprocess.Popen) -> None:
        """Hilo: drena stderr para no bloquear al hijo; lo reenvía al log."""
        try:
            for raw_line in proc.stderr:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line:
                    logger.debug("[whisper-stream] %s", line)
        except Exception as e:
            logger.debug("[WHISPER_WRAPPER] Fin del drenado de stderr: %s", e)

    def send_audio_chunk(self, audio_array: np.ndarray) -> bool:
        """Envía un chunk de audio float32 al proceso de streaming.

        Verifica que el proceso siga vivo antes de escribir; si murió, loguea
        e intenta reiniciarlo con el callback almacenado.
        """
        with self._stream_lock:
            if not self.stream_running or self.stream_process is None:
                logger.warning("⚠️ [WHISPER_WRAPPER] Streaming no está iniciado")
                return False

            if self.stream_process.poll() is not None:
                logger.error("❌ [WHISPER_WRAPPER] El proceso de streaming murió "
                             "(exit=%s), reiniciando", self.stream_process.returncode)
                self.stream_running = False
                self.stream_process = None
                callback = self._stream_callback
                if callback is None or not self._start_streaming_locked(callback):
                    return False
                # Caer al envío con el proceso recién reiniciado.

            try:
                audio_bytes = audio_array.astype(np.float32).tobytes()
                self.stream_process.stdin.write(audio_bytes)
                self.stream_process.stdin.flush()
                return True
            except (BrokenPipeError, OSError) as e:
                logger.error("❌ [WHISPER_WRAPPER] Error enviando chunk: %s", e, exc_info=True)
                self.stream_running = False
                return False

    def stop_streaming(self, timeout: float = _STOP_TIMEOUT) -> None:
        """Detiene el streaming y limpia recursos (idempotente)."""
        with self._stream_lock:
            if not self.stream_running and self.stream_process is None:
                return

            logger.info("🛑 [WHISPER_WRAPPER] Deteniendo streaming")
            self.stream_running = False
            proc = self.stream_process
            self.stream_process = None
            self._stream_callback = None

        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except OSError as e:
                logger.debug("[WHISPER_WRAPPER] stdin ya cerrado: %s", e)
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ [WHISPER_WRAPPER] terminate() no bastó, matando proceso")
                try:
                    proc.kill()
                    proc.wait(timeout=timeout)
                except Exception as e:
                    logger.error("❌ [WHISPER_WRAPPER] No se pudo matar el proceso: %s", e)
            except Exception as e:
                logger.error("❌ [WHISPER_WRAPPER] Error deteniendo streaming: %s", e,
                             exc_info=True)

        logger.info("✅ [WHISPER_WRAPPER] Streaming detenido")

    def _atexit_stop(self) -> None:
        """Parada segura registrada en atexit: timeout corto, nunca cuelga."""
        try:
            self.stop_streaming(timeout=1.0)
        except Exception as e:
            # En el cierre del intérprete hasta el logging puede fallar.
            try:
                logger.error("❌ [WHISPER_WRAPPER] Error en atexit: %s", e)
            except Exception:
                pass
