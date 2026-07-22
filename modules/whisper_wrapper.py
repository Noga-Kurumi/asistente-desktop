"""
Wrapper para whisper.cpp.

- Transcripción final: pywhispercpp (bindings en-proceso). El modelo se carga
  una sola vez en memoria, eliminando el overhead de lanzar whisper-cli.exe
  por cada utterance. Idioma 'auto': whisper detecta el idioma en la misma
  pasada, sin heurísticas ni doble transcripción.
- Transcripción live: whisper-stream-pcm.exe alimentado por stdin (PCM f32).
"""

import os
import atexit
import subprocess
import logging
import numpy as np
import threading
import queue
from typing import Tuple, Callable

logger = logging.getLogger(__name__)


class WhisperCppWrapper:
    """Wrapper de whisper.cpp: transcripción final en-proceso + streaming nativo"""

    def __init__(self, stream_exe_path: str, model_path: str, language: str = 'auto', n_threads: int = 2):
        """
        Inicializa el wrapper.

        Args:
            stream_exe_path: Ruta al ejecutable whisper-stream-pcm.exe
            model_path: Ruta al modelo ggml-*.bin
            language: Idioma de transcripción (default: 'auto' = detección automática)
            n_threads: Número de threads (default: 2)
        """
        self.stream_pcm_exe = stream_exe_path
        self.model_path = model_path
        self.language = language
        self.n_threads = n_threads

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

        # Modelo persistente para transcripción final (cargado una sola vez)
        from pywhispercpp.model import Model
        self.model = Model(
            model_path,
            n_threads=n_threads,
            language=language,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
        )

        logger.info(f"✅ [WHISPER_WRAPPER] Modelo cargado en memoria: {model_path}")
        logger.info(f"✅ [WHISPER_WRAPPER] Stream PCM: {self.stream_pcm_exe} (existe: {os.path.exists(self.stream_pcm_exe)})")

        # Variables para streaming
        self.stream_process = None
        self.stream_queue = None
        self.stream_reader_thread = None
        self.stream_running = False

        # Matar el proceso de streaming si Python sale sin cleanup explícito
        atexit.register(self.stop_streaming)

    def transcribe(self, audio_array: np.ndarray, timeout: int = 30) -> Tuple[bool, str]:
        """
        Transcribe audio con el modelo persistente (idioma auto-detectado).

        Args:
            audio_array: Array numpy con audio float32 a 16kHz
            timeout: Ignorado (kept por compatibilidad de firma)

        Returns:
            Tupla (success, text)
        """
        try:
            audio = np.ascontiguousarray(audio_array, dtype=np.float32)
            segments = self.model.transcribe(audio)
            text = " ".join(seg.text for seg in segments).strip()
            logger.info(f"✅ [WHISPER_WRAPPER] Transcripción completada: '{text}'")
            return True, text
        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error en transcripción: {e}", exc_info=True)
            return False, ""

    def start_streaming(self, callback: Callable[[str], None]) -> bool:
        """
        Inicia el streaming real con whisper-stream-pcm.exe

        Args:
            callback: Función que será llamada con cada segmento de transcripción

        Returns:
            True si el streaming se inició correctamente
        """
        if not os.path.exists(self.stream_pcm_exe):
            logger.error(f"❌ [WHISPER_WRAPPER] whisper-stream-pcm.exe no encontrado")
            return False

        if self.stream_running:
            logger.warning("⚠️ [WHISPER_WRAPPER] Streaming ya está en curso")
            return False

        try:
            # El streaming live queda fijo en español (la detección automática
            # por ventana lo haría más lento); la transcripción final sí usa
            # el idioma auto-detectado del modelo persistente
            stream_language = 'es' if self.language == 'auto' else self.language
            cmd = [
                self.stream_pcm_exe,
                '-m', self.model_path,
                '-l', stream_language,
                '-t', str(self.n_threads),
                '-i', '-',  # stdin
                '--format', 'f32',
                '--sample-rate', '16000',
                '--vad',  # Habilitar VAD
                '--step', '200',  # Reducido de 500ms a 200ms para segmentos más frecuentes
                '--length', '2000'  # Reducido de 5000ms a 2000ms para menor delay
            ]

            logger.info(f"🚀 [WHISPER_WRAPPER] Iniciando streaming real: {' '.join(cmd)}")

            # Iniciar proceso
            self.stream_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Crear cola para comunicación
            self.stream_queue = queue.Queue()

            # Iniciar hilo para leer stdout
            self.stream_reader_thread = threading.Thread(
                target=self._read_stream_output,
                args=(callback,),
                daemon=True
            )
            self.stream_reader_thread.start()

            self.stream_running = True
            logger.info("✅ [WHISPER_WRAPPER] Streaming iniciado correctamente")
            return True

        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error iniciando streaming: {e}", exc_info=True)
            return False

    def _read_stream_output(self, callback: Callable[[str], None]):
        """
        Hilo que lee stdout del proceso de streaming y llama al callback

        Args:
            callback: Función que será llamada con cada segmento de transcripción
        """
        try:
            for line in self.stream_process.stdout:
                line = line.decode('utf-8', errors='ignore').strip()
                if line:
                    logger.debug(f"📄 [WHISPER_WRAPPER] Línea stdout: '{line}'")

                    # Parsear línea de transcripción
                    # Formato: [00:00:00.000 --> 00:00:05.000]   texto
                    if '-->' in line and line.startswith('['):
                        # Extraer texto después del timestamp
                        parts = line.split(']')
                        if len(parts) > 1:
                            text = parts[1].strip()
                            if text and text != '[MÚSICA]' and text != '[MUSIC]':
                                logger.info(f"📝 [WHISPER_WRAPPER] Segmento streaming: '{text}'")
                                callback(text)
        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error leyendo stream output: {e}", exc_info=True)

    def send_audio_chunk(self, audio_array: np.ndarray) -> bool:
        """
        Envía un chunk de audio al proceso de streaming

        Args:
            audio_array: Array numpy con audio float32

        Returns:
            True si el chunk se envió correctamente
        """
        if not self.stream_running or not self.stream_process:
            logger.warning("⚠️ [WHISPER_WRAPPER] Streaming no está iniciado")
            return False

        try:
            # Convertir a bytes
            audio_bytes = audio_array.astype(np.float32).tobytes()

            # Escribir a stdin
            self.stream_process.stdin.write(audio_bytes)
            self.stream_process.stdin.flush()

            logger.debug(f"📡 [WHISPER_WRAPPER] Chunk enviado: {len(audio_array)} samples")
            return True

        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error enviando chunk: {e}", exc_info=True)
            return False

    def stop_streaming(self):
        """Detiene el streaming y limpia recursos"""
        if not self.stream_running and not self.stream_process:
            return

        logger.info("🛑 [WHISPER_WRAPPER] Deteniendo streaming")

        self.stream_running = False

        if self.stream_process:
            try:
                self.stream_process.stdin.close()
                self.stream_process.terminate()
                self.stream_process.wait(timeout=5)
            except Exception:
                try:
                    self.stream_process.kill()
                except Exception:
                    pass
            self.stream_process = None

        self.stream_queue = None
        self.stream_reader_thread = None
        logger.info("✅ [WHISPER_WRAPPER] Streaming detenido")
