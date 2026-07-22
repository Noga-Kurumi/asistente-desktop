"""Motor TTS del asistente (Kokoro ONNX) con lip sync por análisis de audio.

Cambios de interfaz respecto a la versión anterior:

- Señal text_to_speak(texto: str, duracion: float, timeline: list).
  `timeline` es la línea de tiempo de visemas calculada del audio generado:
      [{"t": <segundos desde el inicio>, "viseme": <str>, "weight": <0..1>}, ...]
  Visemas posibles: 'sil', 'aa', 'e', 'ih', 'oh', 'ou'.
  COMPATIBILIDAD: los consumidores (avatar_window.on_text_to_speak y el JS
  window.startSpeaking) deben actualizarse para aceptar el tercer argumento
  (lo hace el agente que reescribe main.py/avatar_window.py).

- process_text_async() NUNCA descarta texto: si el motor aún no está listo,
  el texto queda encolado y se sintetiza cuando termine la inicialización.

- Voz por defecto: config.active_voice (fallback 'ef_dora', español).
  lang='es' fijo, consistente con el idioma del asistente.
"""

import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional

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

DEFAULT_VOICE = "ef_dora"
KOKORO_SAMPLE_RATE = 24000

# --- Parámetros del lip sync (documentados y calibrados para voz 24kHz) ---
FRAME_MS = 40          # ventana de análisis
HOP_MS = 20            # salto entre frames
# Silencio: RMS por debajo de este factor del RMS máximo del clip → 'sil'.
SILENCE_REL_THRESHOLD = 0.06
# Bandas de centroide espectral (Hz) → visema. Heurística: el centroide sube
# con F1 (apertura) y F2 (anterioridad): /u/ y /o/ (cerradas, posteriores)
# tienen centroide bajo; /a/ (abierta) medio; /e/, /i/ (anteriores) alto.
# Calibrado a oído sobre ef_dora; ajustar si se cambia de voz.
CENTROID_BANDS = [
    (900, "ou"),
    (1500, "oh"),
    (2200, "aa"),
    (3200, "e"),
    (float("inf"), "ih"),
]


def compute_viseme_timeline(
    samples: np.ndarray, sample_rate: int = KOKORO_SAMPLE_RATE
) -> List[Dict]:
    """Calcula la línea de tiempo de visemas de un clip PCM float32 mono.

    Por cada frame de FRAME_MS con hop de HOP_MS calcula energía RMS y
    centroide espectral (numpy puro, sin dependencias nuevas) y mapea a un
    visema según SILENCE_REL_THRESHOLD y CENTROID_BANDS. weight es la energía
    normalizada al pico del clip (0..1): cuanto más abierta/fuerte la boca.

    Returns:
        [{"t": float, "viseme": str, "weight": float}, ...] — lista vacía si
        el clip es demasiado corto para un solo frame.
    """
    frame_len = int(sample_rate * FRAME_MS / 1000)
    hop = int(sample_rate * HOP_MS / 1000)
    if len(samples) < frame_len or frame_len <= 0:
        return []

    n_frames = 1 + (len(samples) - frame_len) // hop
    # Ventanas sin copiar: (n_frames, frame_len)
    shape = (n_frames, frame_len)
    strides = (samples.strides[0] * hop, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    max_rms = float(rms.max())
    if max_rms <= 1e-8:
        return [{"t": 0.0, "viseme": "sil", "weight": 0.0}]

    window = np.hanning(frame_len).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    mag_sum = spectrum.sum(axis=1)
    mag_sum[mag_sum <= 1e-12] = 1e-12  # evitar división por cero en frames mudos
    centroid = (spectrum * freqs).sum(axis=1) / mag_sum

    timeline: List[Dict] = []
    for i in range(n_frames):
        rel = float(rms[i]) / max_rms
        if rel < SILENCE_REL_THRESHOLD:
            viseme, weight = "sil", 0.0
        else:
            c = float(centroid[i])
            viseme = next(v for limit, v in CENTROID_BANDS if c < limit)
            weight = min(1.0, rel)
        timeline.append({
            "t": round(i * hop / sample_rate, 3),
            "viseme": viseme,
            "weight": round(weight, 3),
        })
    return timeline


class AssistantTTS(QObject):
    speech_started = Signal()
    speech_ended = Signal()
    # (texto, duración_segundos, timeline_visemas) — ver docstring del módulo.
    text_to_speak = Signal(str, float, list)

    def __init__(self, config=None):
        super().__init__()
        from modules.config_manager import get_config

        self.config = config or get_config()
        # Modelos en la raíz del repo (junto a main.py).
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_path = os.path.join(self.base_dir, "kokoro-v1.0.onnx")
        self.model_quant_path = os.path.join(self.base_dir, "kokoro-v1.0.int8.onnx")
        self.voices_path = os.path.join(self.base_dir, "voices-v1.0.bin")

        self.kokoro: Optional[Kokoro] = None
        self.avatar_widget = None  # compat: referencia al avatar (visemes legacy)

        # Se activa cuando el motor termina de inicializarse. El worker espera
        # en este evento, así los textos encolados antes de tiempo no se
        # descartan: se sintetizan en cuanto el motor está listo.
        self._ready_event = threading.Event()

        # Cola serializada: un único hilo worker reproduce los textos en orden,
        # evitando que sd.play() de dos textos se solapen o se corten.
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._tts_worker, daemon=True, name="tts-worker")
        self._worker.start()

        threading.Thread(target=self._init_engine, daemon=True, name="tts-init").start()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def set_avatar_widget(self, avatar_widget) -> None:
        """Compatibilidad: referencia al avatar_widget (lip sync legacy)."""
        self.avatar_widget = avatar_widget

    def _init_engine(self) -> None:
        inicio = time.perf_counter()

        # Prioridad al modelo normal, fallback al cuantizado.
        modelo_a_usar = self.model_path
        if not os.path.exists(self.model_path) and os.path.exists(self.model_quant_path):
            modelo_a_usar = self.model_quant_path

        if not os.path.exists(modelo_a_usar) or not os.path.exists(self.voices_path):
            logger.error("❌ [TTS] Modelo o voces no encontrados: %s, %s",
                         modelo_a_usar, self.voices_path)
            return

        try:
            self.kokoro = Kokoro(modelo_a_usar, self.voices_path)
            fin = time.perf_counter()
            self._ready_event.set()
            logger.info("✅ [TTS] Motor Kokoro listo en %.2fs (%s)",
                        fin - inicio, os.path.basename(modelo_a_usar))
        except Exception as e:
            logger.error("❌ [TTS] Error inicializando Kokoro: %s", e, exc_info=True)

    def process_text_async(self, text: str, voice: Optional[str] = None) -> None:
        """Encola un texto para sintetizar. Nunca descarta: si el motor aún no
        está listo, el texto espera en cola hasta que termine la init."""
        if not text or not text.strip():
            return
        if voice is None:
            voice = self.config.get("active_voice", DEFAULT_VOICE) or DEFAULT_VOICE
        if not self.is_ready:
            logger.info("⏳ [TTS] Motor no listo, texto encolado a la espera")
        self._queue.put((text, voice))

    def _tts_worker(self) -> None:
        """Worker único: consume la cola y reproduce cada texto en orden."""
        while not self._stop_event.is_set():
            try:
                text, voice = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # Esperar al motor (con timeout para seguir siendo cancelable).
                while not self._ready_event.wait(timeout=0.5):
                    if self._stop_event.is_set():
                        return
                self._generate_and_play(text, voice)
            except Exception as e:
                logger.error("❌ [TTS] Error en worker: %s", e, exc_info=True)
            finally:
                self._queue.task_done()

    def _generate_and_play(self, text: str, voice: str) -> None:
        """Genera audio, calcula visemas y reproduce (solo desde el worker)."""
        try:
            samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="es")

            audio_duration = len(samples) / sample_rate
            timeline = compute_viseme_timeline(
                np.ascontiguousarray(samples, dtype=np.float32), sample_rate
            )

            # Señal con texto, duración y timeline para lip sync (thread-safe).
            self.text_to_speak.emit(text, audio_duration, timeline)

            self.speech_started.emit()
            sd.play(samples, sample_rate)
            sd.wait()
            self.speech_ended.emit()

        except Exception as e:
            logger.error("❌ [TTS] Error generando/reproduciendo audio: %s", e, exc_info=True)
            self.speech_ended.emit()

    def cleanup(self) -> None:
        """Detiene el worker y cualquier reproducción en curso."""
        # TODO(pendiente): cleanup puede quedar bloqueado en sd.wait(); revisar
        # cuando se aborde el cierre limpio de la app.
        self._stop_event.set()
        try:
            sd.stop()
        except Exception as e:
            logger.warning("⚠️ [TTS] Error en sd.stop() durante cleanup: %s", e)


def get_available_voices(voices_path: Optional[str] = None):
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
            "a": "Inglés (US)",
            "b": "Inglés (UK)",
            "e": "Español",
            "f": "Francés",
            "i": "Italiano",
            "j": "Japonés",
            "p": "Portugués",
            "h": "Hindi",
            "k": "Coreano",
            "z": "Chino",
        }

        formatted_voices = []
        for voice in voice_names:
            if len(voice) >= 2:
                prefix = voice[0].lower()
                lang = lang_map.get(prefix, "Desconocido")
                formatted_name = f"{voice} - {lang}"
                formatted_voices.append((voice, formatted_name))
            else:
                formatted_voices.append((voice, voice))

        return formatted_voices
    except Exception as e:
        logger.error("❌ [TTS] Error leyendo voces desde %s: %s", voices_path, e, exc_info=True)
        return []
