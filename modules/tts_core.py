import os
import queue
import logging
import threading
import time
import numpy as np
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    from kokoro_onnx import Kokoro
except ImportError as e:
    raise ImportError(
        "Faltan dependencias de TTS. Instálalas con: pip install -r requirements.txt"
    ) from e


class AssistantTTS(QObject):
    speech_started = Signal()
    speech_ended = Signal()
    text_to_speak = Signal(str, float)  # Señal para enviar texto y duración al avatar (thread-safe)

    def __init__(self):
        super().__init__()
        # Asumimos que los modelos están en la raíz del repo (junto a main.py)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_path = os.path.join(self.base_dir, "kokoro-v1.0.onnx")
        self.model_quant_path = os.path.join(self.base_dir, "kokoro-v1.0.int8.onnx")
        self.voices_path = os.path.join(self.base_dir, "voices-v1.0.bin")

        self.kokoro = None
        self.is_ready = False
        self.avatar_widget = None  # Referencia al avatar_widget para enviar visemes

        # Cola serializada: un único hilo worker reproduce los textos en orden,
        # evitando que sd.play() de dos textos se solapen o se corten
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker.start()

        threading.Thread(target=self._init_engine, daemon=True).start()

    def set_avatar_widget(self, avatar_widget):
        """Establece la referencia al avatar_widget para enviar visemes"""
        self.avatar_widget = avatar_widget

    def _init_engine(self):
        inicio = time.perf_counter()

        # Prioridad al modelo normal, fallback al quantizado
        modelo_a_usar = self.model_path
        if not os.path.exists(self.model_path) and os.path.exists(self.model_quant_path):
            modelo_a_usar = self.model_quant_path

        if not os.path.exists(modelo_a_usar) or not os.path.exists(self.voices_path):
            logger.error(f"❌ [TTS] Modelo o voces no encontrados: {modelo_a_usar}, {self.voices_path}")
            return

        try:
            self.kokoro = Kokoro(modelo_a_usar, self.voices_path)

            fin = time.perf_counter()
            self.is_ready = True
            logger.info(f"✅ [TTS] Motor Kokoro listo en {fin - inicio:.2f}s ({os.path.basename(modelo_a_usar)})")
        except Exception as e:
            logger.error(f"❌ [TTS] Error inicializando Kokoro: {e}", exc_info=True)

    def process_text_async(self, text, voice="af_bella"):
        if not self.is_ready or not text.strip():
            if not self.is_ready:
                logger.warning("⚠️ [TTS] Motor no listo, se descarta el texto a sintetizar")
            return
        self._queue.put((text, voice))

    def _tts_worker(self):
        """Worker único: consume la cola y reproduce cada texto en orden."""
        while not self._stop_event.is_set():
            try:
                text, voice = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._generate_and_play(text, voice)
            except Exception as e:
                logger.error(f"❌ [TTS] Error en worker: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    def _generate_and_play(self, text, voice):
        """Genera audio y reproduce (llamado solo desde el worker)"""
        try:
            samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="es")

            # Calcular duración del audio en segundos
            audio_duration = len(samples) / sample_rate

            # Emitir señal con texto y duración para lip sync (thread-safe)
            self.text_to_speak.emit(text, audio_duration)

            self.speech_started.emit()
            sd.play(samples, sample_rate)
            sd.wait()
            self.speech_ended.emit()

        except Exception as e:
            logger.error(f"❌ [TTS] Error generando/reproduciendo audio: {e}", exc_info=True)
            self.speech_ended.emit()

    def cleanup(self):
        """Detiene el worker y cualquier reproducción en curso."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass


def get_available_voices(voices_path=None):
    """Extrae la lista de voces disponibles desde voices-v1.0.bin con sus idiomas."""
    if voices_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        voices_path = os.path.join(base_dir, "voices-v1.0.bin")

    if not os.path.exists(voices_path):
        return []

    try:
        voices = np.load(voices_path)
        voice_names = list(sorted(voices.keys()))

        lang_map = {
            'a': 'Inglés (US)',
            'b': 'Inglés (UK)',
            'e': 'Español',
            'f': 'Francés',
            'i': 'Italiano',
            'j': 'Japonés',
            'p': 'Portugués',
            'h': 'Hindi',
            'k': 'Coreano',
            'z': 'Chino'
        }

        formatted_voices = []
        for voice in voice_names:
            if len(voice) >= 2:
                prefix = voice[0].lower()
                lang = lang_map.get(prefix, 'Desconocido')
                formatted_name = f"{voice} - {lang}"
                formatted_voices.append((voice, formatted_name))
            else:
                formatted_voices.append((voice, voice))

        return formatted_voices
    except Exception as e:
        logger.error(f"❌ [TTS] Error leyendo voces desde {voices_path}: {e}", exc_info=True)
        return []
