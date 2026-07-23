"""Motor TTS del asistente (Kokoro ONNX) con lip sync por análisis de audio.

Cambios de interfaz respecto a la versión anterior:

- Señal text_to_speak(texto: str, duracion: float, timeline: list).
  `timeline` es la línea de tiempo de visemas calculada del audio generado:
      [{"t": <segundos desde el inicio>, "viseme": <str>, "weight": <0..1>}, ...]
  Visemas posibles: 'sil', 'aa', 'e', 'ih', 'oh', 'ou'.

- Síntesis en streaming: se usa Kokoro.create_stream (kokoro-onnx >= 0.4),
  que entrega el audio por lotes de fonemas. Si el paquete instalado no tiene
  create_stream, se cae a create() de una sola pieza.

- Pipeline productor-consumidor (anti-pausas entre frases), dos hilos:

      _text_queue → [hilo SINTETIZADOR] → _audio_queue → [hilo REPRODUCTOR]

  El sintetizador convierte cada texto en lotes de audio (con su timeline de
  visemas ya calculada) y los deposita en una cola acotada (maxsize=4 lotes):
  la síntesis de la oración N+1 ocurre MIENTRAS suena la N, y el bloqueo por
  cola llena es el backpressure natural para no acumular RAM en respuestas
  largas. El reproductor hace sd.play/sd.wait por lote y emite text_to_speak
  AL REPRODUCIR (no al sintetizar), así el lip sync del JS sigue alineado con
  lo que suena. Por cada texto el sintetizador deposita además un centinela
  "end" (incluso si la síntesis falla) para que el reproductor nunca espere
  lotes que no llegarán.

- Coordinación de fin de turno (anti-parpadeo): pending_count cuenta TEXTOS
  encolados que aún no terminaron de sonar (esperando síntesis + en síntesis
  + lotes esperando sonar + el que suena). queue_drained se emite cuando ese
  contador llega a 0 (verdad del reproductor Y del sintetizador sin trabajo).
  clear_queue() vacía la cola de textos + sd.stop() e invalida por época la
  cola de audio y lo que estuviera a medio sintetizar.

- process_text_async() NUNCA descarta texto: si el motor aún no está listo,
  el texto queda encolado y se sintetiza cuando termine la inicialización.

- Voz por defecto: config.active_voice (fallback 'ef_dora', español).
  lang='es' fijo, consistente con el idioma del asistente.
  Modelo según config.tts_model ("int8" | "fp32", default "int8").
"""

import asyncio
import logging
import os
import queue
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

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
    # La cola quedó vacía tras terminar un texto: main.py la usa para cerrar
    # el turno SOLO cuando ya no queda nada por sonar (anti-parpadeo).
    queue_drained = Signal()
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

        # Pipeline productor-consumidor (ver docstring del módulo):
        #   _text_queue → [hilo sintetizador] → _audio_queue → [hilo reproductor]
        self._text_queue: "queue.Queue[tuple]" = queue.Queue()
        # Acotada: backpressure natural para no acumular RAM en respuestas largas.
        self._audio_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()

        # Verdad del trabajo pendiente (thread-safe): textos encolados que aún
        # no terminaron de sonar. _epoch invalida lo encolado al clear_queue.
        self._state_lock = threading.Lock()
        self._pending_texts = 0
        self._epoch = 0

        self._synth_thread = threading.Thread(
            target=self._synth_worker, daemon=True, name="tts-synth")
        self._play_thread = threading.Thread(
            target=self._play_worker, daemon=True, name="tts-player")
        self._synth_thread.start()
        self._play_thread.start()

        threading.Thread(target=self._init_engine, daemon=True, name="tts-init").start()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    @property
    def pending_count(self) -> int:
        """Textos encolados que aún no terminaron de sonar (0 = todo sonó).

        Cuenta textos, no lotes: incluye los que esperan síntesis, el que se
        está sintetizando, los lotes ya sintetizados esperando sonar y el que
        suena ahora. main.py cierra el turno solo cuando llega a 0.
        """
        with self._state_lock:
            return self._pending_texts

    def clear_queue(self) -> None:
        """Cancela TODO lo pendiente (nuevo turno del usuario, MVP).

        Vacia la cola de TEXTOS (lo no sintetizado jamás se sintetiza) y frena
        la reproducción en curso con sd.stop(). La cola de audio NO se vacía:
        se invalida por época y el reproductor descarta los lotes sin sonarlos
        (es acotada: se drena al instante), así el centinela "end" del texto
        interrumpido llega y su speech_ended se emite igual (contrato). Lo que
        el sintetizador tenga a medias queda invalidado y se descarta sin
        tocar el contador (ya reseteado acá).
        """
        with self._state_lock:
            self._epoch += 1
            self._pending_texts = 0
        with self._text_queue.mutex:
            self._text_queue.queue.clear()
        try:
            sd.stop()  # sd.wait() del reproductor retorna al instante
        except Exception as e:
            logger.warning("⚠️ [TTS] Error en sd.stop() al limpiar la cola: %s", e)

    def set_avatar_widget(self, avatar_widget) -> None:
        """Compatibilidad: referencia al avatar_widget (lip sync legacy)."""
        self.avatar_widget = avatar_widget

    def _init_engine(self) -> None:
        inicio = time.perf_counter()

        # Variante del modelo según config (tts_model: "int8" | "fp32"),
        # con fallback a la otra si falta la pedida.
        prefer = str(self.config.get("tts_model", "int8") or "int8")
        candidatos = ([self.model_quant_path, self.model_path] if prefer == "int8"
                      else [self.model_path, self.model_quant_path])
        modelo_a_usar = next((p for p in candidatos if os.path.exists(p)), None)

        if modelo_a_usar is None or not os.path.exists(self.voices_path):
            logger.error("❌ [TTS] Modelo o voces no encontrados: %s, %s",
                         candidatos[0], self.voices_path)
            return
        if modelo_a_usar != candidatos[0]:
            logger.warning("⚠️ [TTS] tts_model='%s' pero falta %s; usando %s",
                           prefer, os.path.basename(candidatos[0]),
                           os.path.basename(modelo_a_usar))

        try:
            self.kokoro = Kokoro(modelo_a_usar, self.voices_path)
            fin = time.perf_counter()
            logger.info("✅ [TTS] Motor Kokoro listo en %.2fs (%s)",
                        fin - inicio, os.path.basename(modelo_a_usar))
            # Pre-calentar: la primera síntesis real pagaba el warm-up de
            # ONNX/espeak (observado ~11s de espera en la primera frase).
            warm = time.perf_counter()
            try:
                voice = self.config.get("active_voice", DEFAULT_VOICE) or DEFAULT_VOICE
                for _ in self._stream_chunks("Hola.", voice):
                    pass
                logger.info("🔥 [TTS] Warm-up completado en %.2fs",
                            time.perf_counter() - warm)
            except Exception as e:
                logger.warning("⚠️ [TTS] Warm-up falló (no crítico): %s", e)
            self._ready_event.set()
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
        with self._state_lock:
            self._pending_texts += 1
            epoch = self._epoch
        self._text_queue.put((text, voice, epoch))

    def _synth_worker(self) -> None:
        """Hilo sintetizador: consume textos y deposita lotes de audio listos.

        Por cada texto deposita 0..N items ("chunk", payload) y SIEMPRE un
        centinela ("end", None) — incluso si la síntesis falla — para que el
        reproductor nunca quede esperando lotes que no llegarán.
        """
        while not self._stop_event.is_set():
            try:
                text, voice, epoch = self._text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                with self._state_lock:
                    if epoch != self._epoch:
                        continue  # texto cancelado por clear_queue
                # Esperar al motor (con timeout para seguir siendo cancelable).
                while not self._ready_event.wait(timeout=0.5):
                    if self._stop_event.is_set():
                        return
                t0 = time.perf_counter()
                audio_secs = 0.0
                for samples, sample_rate in self._stream_chunks(text, voice):
                    timeline = compute_viseme_timeline(
                        np.ascontiguousarray(samples, dtype=np.float32), sample_rate
                    )
                    duration = len(samples) / sample_rate
                    audio_secs += duration
                    # Bloqueo por cola llena = backpressure natural.
                    self._audio_queue.put(
                        (epoch, "chunk", (text, samples, sample_rate, timeline, duration)))
                synth_secs = time.perf_counter() - t0
                logger.info(
                    "🗣️ [TTS] Síntesis '%.40s': %.2fs para %.2fs de audio (RTF %.2f)",
                    text, synth_secs, audio_secs,
                    synth_secs / audio_secs if audio_secs > 0 else 0.0)
            except Exception as e:
                logger.error("❌ [TTS] Error sintetizando '%.40s': %s", text, e, exc_info=True)
            finally:
                self._audio_queue.put((epoch, "end", None))
                self._text_queue.task_done()

    def _play_worker(self) -> None:
        """Hilo reproductor: suena los lotes en orden y emite las señales.

        text_to_speak se emite AL REPRODUCIR (no al sintetizar) para que el
        lip sync del JS siga alineado con lo que suena; speech_started al
        primer lote de cada texto y speech_ended SIEMPRE una vez por texto que
        empezó a sonar (incluido el interrumpido por clear_queue). Al terminar
        un texto decrementa el pendiente; si llega a 0 emite queue_drained
        (no queda NADA por sonar: el contador cubre ambas colas).
        """
        started = False
        while not self._stop_event.is_set():
            try:
                epoch, kind, payload = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._state_lock:
                stale = epoch != self._epoch
            if stale:
                # Item cancelado por clear_queue: se descarta sin decrementar
                # (el contador ya se reseteó), pero el texto interrumpido a
                # medio sonar cierra su señal igual.
                if kind == "end" and started:
                    self.speech_ended.emit()
                    started = False
                continue
            try:
                if kind == "chunk":
                    text, samples, sample_rate, timeline, duration = payload
                    self.text_to_speak.emit(text, duration, timeline)
                    if not started:
                        logger.info("▶️ [TTS] Reproduciendo '%.40s'", text)
                        self.speech_started.emit()
                        started = True
                    sd.play(samples, sample_rate)
                    sd.wait()
                else:  # "end": último lote del texto ya sonó
                    if started:
                        logger.info("⏹️ [TTS] Fin de reproducción")
                        self.speech_ended.emit()
                        started = False
                    with self._state_lock:
                        self._pending_texts -= 1
                        drained = self._pending_texts == 0
                    if drained:
                        self.queue_drained.emit()
            except Exception as e:
                logger.error("❌ [TTS] Error reproduciendo audio: %s", e, exc_info=True)
            finally:
                self._audio_queue.task_done()

    def _stream_chunks(
        self, text: str, voice: str
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Genera (samples, sample_rate) por lotes, a medida que se sintetizan.

        Usa Kokoro.create_stream (async generator) consumido con un event loop
        propio en este hilo: cada __anext__ espera el siguiente lote mientras
        el executor del loop sintetiza en segundo plano. Fallback a create()
        de una pieza si el paquete no tiene create_stream.
        """
        create_stream = getattr(self.kokoro, "create_stream", None)
        if create_stream is None:
            logger.warning("⚠️ [TTS] kokoro_onnx sin create_stream; síntesis de una pieza")
            yield self.kokoro.create(text, voice=voice, speed=1.0, lang="es")
            return

        loop = asyncio.new_event_loop()
        agen = create_stream(text, voice=voice, speed=1.0, lang="es")
        try:
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception as e:
                logger.warning("⚠️ [TTS] Error cerrando el executor del stream: %s", e)
            loop.close()

    def cleanup(self) -> None:
        """Frena los hilos (sintetizador y reproductor) y el audio en curso."""
        # TODO(pendiente): cleanup puede quedar bloqueado en sd.wait() o en un
        # put a la cola de audio llena; los hilos son daemon, así que el
        # proceso igual puede salir. Revisar cuando se aborde el cierre limpio.
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
