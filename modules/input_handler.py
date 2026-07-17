import time
import os
import numpy as np
import sounddevice as sd
from pynput import keyboard
from PySide6.QtCore import QObject, Signal, QTimer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1

class VoiceInputManager(QObject):
    recording_started = Signal()
    recording_stopped = Signal()
    recording_canceled = Signal()
    audio_ready = Signal(np.ndarray)     
    audio_live_ready = Signal(np.ndarray) 
    volume_level = Signal(float)  # Nueva señal para nivel de volumen (0-1)

    # Señales internas para puentear los hilos sin que Qt explote
    _start_timer_sig = Signal()
    _stop_timer_sig = Signal()

    def __init__(self, hotkey=keyboard.Key.alt_r, device=None):
        super().__init__()
        self.hotkey = hotkey
        self.is_recording = False
        self.is_locked = True # Bloqueado inicialmente hasta que el sistema cargue
        self.start_time = 0
        self.audio_buffer = []
        
        # Configurar dispositivo de audio
        if device is None:
            # Leer dispositivo del config.json
            import json
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        device = config.get("audio_device")
                        if device is not None:
                            logger.info(f"🎤 [INPUT] Dispositivo de audio desde config: {device}")
                except Exception as e:
                    logger.warning(f"⚠️ [INPUT] Error leyendo config.json: {e}")
            
            # Si no hay dispositivo en config, buscar automáticamente
            if device is None:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0 and 'voicemeeter' not in dev['name'].lower():
                        # Preferir dispositivos con "micrófono" o "microphone" en el nombre
                        if 'micrófono' in dev['name'].lower() or 'microphone' in dev['name'].lower():
                            device = i
                            logger.info(f"🎤 [INPUT] Dispositivo de audio seleccionado automáticamente: [{i}] {dev['name']}")
                            break
                # Si no se encontró micrófono específico, usar el primer dispositivo de entrada que no sea virtual
                if device is None:
                    for i, dev in enumerate(devices):
                        if dev['max_input_channels'] > 0 and 'voicemeeter' not in dev['name'].lower():
                            device = i
                            logger.info(f"🎤 [INPUT] Dispositivo de audio seleccionado: [{i}] {dev['name']}")
                            break
        
        if device is not None:
            logger.info(f"🎤 [INPUT] Usando dispositivo de audio: {device}")
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=self.audio_callback, device=device)
        else:
            logger.warning("⚠️ [INPUT] No se encontró dispositivo de audio adecuado, usando por defecto")
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=self.audio_callback)
        
        self.live_timer = QTimer()
        self.live_timer.setInterval(500)  # 500ms para streaming real con whisper-stream-pcm.exe
        self.live_timer.timeout.connect(self.emit_live_audio)
        
        # Conectamos las señales internas al timer en el hilo principal
        self._start_timer_sig.connect(self.live_timer.start)
        self._stop_timer_sig.connect(self.live_timer.stop)
        
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def set_locked(self, state):
        logger.info(f"🔒 [INPUT] Estado de bloqueo cambiado a: {state}")
        self.is_locked = state

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_buffer.append(indata.copy())
            
            # Calcular nivel de volumen (RMS)
            rms = np.sqrt(np.mean(indata ** 2))
            # Normalizar a 0-1 (asumiendo que el máximo normal es ~0.1)
            level = min(rms * 10, 1.0)
            self.volume_level.emit(level)

    def emit_live_audio(self):
        if self.is_recording and self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer).flatten()
            self.audio_live_ready.emit(audio_data)

    def on_press(self, key):
        if self.is_locked: 
            return
            
        if key == keyboard.Key.esc and self.is_recording:
            logger.info("⏹️ [INPUT] ESC presionado, cancelando grabación")
            self.is_recording = False
            self.stream.stop()
            self._stop_timer_sig.emit()
            self.audio_buffer = []
            self.recording_canceled.emit()
            return

        if key == self.hotkey and not self.is_recording:
            logger.info(f"🎙️ [INPUT] Hotkey presionado, iniciando grabación: {key}")
            self.is_recording = True
            self.start_time = time.time()
            self.audio_buffer = []
            self.stream.start()
            self._start_timer_sig.emit()
            logger.info("📡 [INPUT] Señal _start_timer_sig emitida")
            self.recording_started.emit()

    def on_release(self, key):
        if key == self.hotkey and self.is_recording:
            logger.info(f"🛑 [INPUT] Hotkey soltado, deteniendo grabación: {key}")
            self.is_recording = False
            self.stream.stop()
            self._stop_timer_sig.emit()
            logger.info("📡 [INPUT] Señal _stop_timer_sig emitida")
            self.recording_stopped.emit()
            logger.info("📡 [INPUT] Señal recording_stopped emitida")
            
            duration = time.time() - self.start_time
            logger.info(f"⏱️ [INPUT] Duración de grabación: {duration:.2f}s")
            if duration >= 1.0:
                if self.audio_buffer:
                    audio_data = np.concatenate(self.audio_buffer).flatten()
                    logger.info(f"✅ [INPUT] Audio listo para transcripción: {len(audio_data)} samples")
                    self.audio_ready.emit(audio_data)
                    logger.info("📡 [INPUT] Señal audio_ready emitida")
            else:
                logger.info(f"❌ [INPUT] Grabación muy corta (<1s), cancelando")
                self.audio_buffer = []
                self.recording_canceled.emit()
                logger.info("📡 [INPUT] Señal recording_canceled emitida")
    
    def cleanup(self):
        """Limpia recursos antes de salir"""
        if self.listener:
            self.listener.stop()
        if self.stream:
            self.stream.stop()
        self.live_timer.stop()