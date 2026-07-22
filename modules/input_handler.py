"""Gestor de entrada de voz: hotkey push-to-talk + captura de micrófono.

Windows-only (pynput + sounddevice tal cual); no se invierte en crossplatform.

Lee la configuración exclusivamente vía modules/config_manager (nada de
json.load propio) y loguea con logging.getLogger (nada de basicConfig: el
root logger lo configura modules/log_setup.setup_logging() desde el entry
point).

Si el dispositivo de audio falla al abrirse (desconectado, driver caído), no
crashea: loguea, emite error_occurred("ERR_MIC") y queda sin stream hasta que
se reintente con reopen_stream().
"""

import logging
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from pynput import keyboard
from PySide6.QtCore import QObject, QTimer, Signal

from modules.config_manager import get_config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1


class VoiceInputManager(QObject):
    recording_started = Signal()
    recording_stopped = Signal()
    recording_canceled = Signal()
    audio_ready = Signal(np.ndarray)
    audio_live_ready = Signal(np.ndarray)
    volume_level = Signal(float)  # nivel de volumen (0-1)
    error_occurred = Signal(str)  # código ERR_* (p.ej. ERR_MIC)

    # Señales internas para puentear los hilos sin que Qt explote.
    _start_timer_sig = Signal()
    _stop_timer_sig = Signal()

    def __init__(self, hotkey=keyboard.Key.alt_r, device=None, config=None):
        super().__init__()
        self.config = config or get_config()
        self.hotkey = hotkey
        self.is_recording = False
        self.is_locked = True  # Bloqueado inicialmente hasta que el sistema cargue
        self.start_time = 0
        self.audio_buffer = []
        self.stream: Optional[sd.InputStream] = None

        if device is None:
            device = self.config.get("audio_device")
            if device is not None:
                logger.info("🎤 [INPUT] Dispositivo de audio desde config: %s", device)
        if device is None:
            device = self._autodetect_input_device()

        self._open_stream(device)

        self.live_timer = QTimer()
        self.live_timer.setInterval(500)  # 500ms para streaming con whisper-stream-pcm.exe
        self.live_timer.timeout.connect(self.emit_live_audio)

        # Conectamos las señales internas al timer en el hilo principal.
        self._start_timer_sig.connect(self.live_timer.start)
        self._stop_timer_sig.connect(self.live_timer.stop)

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    # ------------------------------------------------------------- dispositivo

    @staticmethod
    def _autodetect_input_device() -> Optional[int]:
        """Elige un dispositivo de entrada razonable (no virtual)."""
        try:
            devices = sd.query_devices()
        except Exception as e:
            logger.error("❌ [INPUT] No se pudieron enumerar dispositivos: %s", e, exc_info=True)
            return None

        def usable(i, dev):
            return dev["max_input_channels"] > 0 and "voicemeeter" not in dev["name"].lower()

        for i, dev in enumerate(devices):
            name = dev["name"].lower()
            if usable(i, dev) and ("micrófono" in name or "microphone" in name):
                logger.info("🎤 [INPUT] Dispositivo autodetectado: [%d] %s", i, dev["name"])
                return i
        for i, dev in enumerate(devices):
            if usable(i, dev):
                logger.info("🎤 [INPUT] Dispositivo seleccionado: [%d] %s", i, dev["name"])
                return i
        return None

    def _open_stream(self, device: Optional[int]) -> None:
        """Abre el InputStream; ante fallo loguea y emite ERR_MIC (sin crash)."""
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                callback=self.audio_callback, device=device,
            )
            logger.info("🎤 [INPUT] Stream de audio abierto (dispositivo: %s)", device)
        except (sd.PortAudioError, ValueError) as e:
            logger.error("❌ [INPUT] No se pudo abrir el dispositivo %s: %s", device, e,
                         exc_info=True)
            self.stream = None
            self.error_occurred.emit("ERR_MIC")

    def reopen_stream(self, device: Optional[int] = None) -> None:
        """Reintenta abrir el stream (p.ej. tras reconectar el micrófono)."""
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception as e:
                logger.warning("⚠️ [INPUT] Error cerrando stream previo: %s", e)
            self.stream = None
        if device is None:
            device = self.config.get("audio_device")
        self._open_stream(device)

    # ------------------------------------------------------------------ eventos

    def set_locked(self, state: bool) -> None:
        logger.info("🔒 [INPUT] Estado de bloqueo cambiado a: %s", state)
        self.is_locked = state

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("[INPUT] Estado del stream: %s", status)
        if self.is_recording:
            self.audio_buffer.append(indata.copy())
            # Nivel de volumen (RMS) normalizado a 0-1 (máximo normal ~0.1).
            rms = np.sqrt(np.mean(indata ** 2))
            level = min(rms * 10, 1.0)
            self.volume_level.emit(level)

    def emit_live_audio(self) -> None:
        if self.is_recording and self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer).flatten()
            self.audio_live_ready.emit(audio_data)

    def on_press(self, key) -> None:
        if self.is_locked:
            return

        if key == keyboard.Key.esc and self.is_recording:
            logger.info("⏹️ [INPUT] ESC presionado, cancelando grabación")
            self.is_recording = False
            self._stop_stream()
            self._stop_timer_sig.emit()
            self.audio_buffer = []
            self.recording_canceled.emit()
            return

        if key == self.hotkey and not self.is_recording:
            if self.stream is None:
                logger.error("❌ [INPUT] Sin stream de audio; no se puede grabar")
                self.error_occurred.emit("ERR_MIC")
                return
            logger.info("🎙️ [INPUT] Hotkey presionado, iniciando grabación: %s", key)
            self.is_recording = True
            self.start_time = time.time()
            self.audio_buffer = []
            try:
                self.stream.start()
            except sd.PortAudioError as e:
                logger.error("❌ [INPUT] Error arrancando el stream: %s", e, exc_info=True)
                self.is_recording = False
                self.error_occurred.emit("ERR_MIC")
                return
            self._start_timer_sig.emit()
            self.recording_started.emit()

    def on_release(self, key) -> None:
        if key == self.hotkey and self.is_recording:
            logger.info("🛑 [INPUT] Hotkey soltado, deteniendo grabación: %s", key)
            self.is_recording = False
            self._stop_stream()
            self._stop_timer_sig.emit()
            self.recording_stopped.emit()

            duration = time.time() - self.start_time
            logger.info("⏱️ [INPUT] Duración de grabación: %.2fs", duration)
            if duration >= 1.0:
                if self.audio_buffer:
                    audio_data = np.concatenate(self.audio_buffer).flatten()
                    logger.info("✅ [INPUT] Audio listo para transcripción: %d samples",
                                len(audio_data))
                    self.audio_ready.emit(audio_data)
            else:
                logger.info("❌ [INPUT] Grabación muy corta (<1s), cancelando")
                self.audio_buffer = []
                self.recording_canceled.emit()

    def _stop_stream(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
            except sd.PortAudioError as e:
                logger.warning("⚠️ [INPUT] Error deteniendo el stream: %s", e)

    def cleanup(self) -> None:
        """Limpia recursos antes de salir."""
        if self.listener:
            self.listener.stop()
        self._stop_stream()
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception as e:
                logger.warning("⚠️ [INPUT] Error cerrando el stream: %s", e)
        self.live_timer.stop()
