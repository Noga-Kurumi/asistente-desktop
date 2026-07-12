import time
import numpy as np
import sounddevice as sd
from pynput import keyboard
from PySide6.QtCore import QObject, Signal, QTimer

SAMPLE_RATE = 16000
CHANNELS = 1

class VoiceInputManager(QObject):
    recording_started = Signal()
    recording_stopped = Signal()
    recording_canceled = Signal()
    audio_ready = Signal(np.ndarray)     
    audio_live_ready = Signal(np.ndarray) 

    # Señales internas para puentear los hilos sin que Qt explote
    _start_timer_sig = Signal()
    _stop_timer_sig = Signal()

    def __init__(self, hotkey=keyboard.Key.alt_r):
        super().__init__()
        self.hotkey = hotkey
        self.is_recording = False
        self.is_locked = True # Bloqueado inicialmente hasta que el sistema cargue
        self.start_time = 0
        self.audio_buffer = []
        
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=self.audio_callback)
        
        self.live_timer = QTimer()
        self.live_timer.setInterval(1500)
        self.live_timer.timeout.connect(self.emit_live_audio)
        
        # Conectamos las señales internas al timer en el hilo principal
        self._start_timer_sig.connect(self.live_timer.start)
        self._stop_timer_sig.connect(self.live_timer.stop)
        
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def set_locked(self, state):
        self.is_locked = state

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_buffer.append(indata.copy())

    def emit_live_audio(self):
        if self.is_recording and self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer).flatten()
            self.audio_live_ready.emit(audio_data)

    def on_press(self, key):
        if self.is_locked: 
            return # BLOQUEO ABSOLUTO: No hace nada si está pensando o hablando
            
        if key == keyboard.Key.esc and self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self._stop_timer_sig.emit()
            self.audio_buffer = []
            self.recording_canceled.emit()
            return

        if key == self.hotkey and not self.is_recording:
            self.is_recording = True
            self.start_time = time.time()
            self.audio_buffer = []
            self.stream.start()
            self._start_timer_sig.emit()
            self.recording_started.emit()

    def on_release(self, key):
        if key == self.hotkey and self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self._stop_timer_sig.emit()
            self.recording_stopped.emit()
            
            duration = time.time() - self.start_time
            if duration >= 1.0:
                if self.audio_buffer:
                    audio_data = np.concatenate(self.audio_buffer).flatten()
                    self.audio_ready.emit(audio_data)
            else:
                self.audio_buffer = []
                self.recording_canceled.emit()