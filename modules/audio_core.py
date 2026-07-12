import os
import json
import time
import threading
from faster_whisper import WhisperModel
from PySide6.QtCore import QObject, Signal

CONFIG_FILE = "config.json"

class AssistantAudioCore(QObject):
    text_transcribed = Signal(str)      
    live_text_ready = Signal(str)       

    def __init__(self):
        super().__init__()
        self.config = self._load_config()
        
        # cpu_threads=2 -> Dejamos núcleos libres para que la UI no se congele
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2)
        self.is_live_transcribing = False

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}

    def process_voice_input(self, audio_array):
        threading.Thread(target=self._final_transcribe_thread, args=(audio_array,), daemon=True).start()

    def _final_transcribe_thread(self, audio_array):
        try:
            segments, _ = self.whisper.transcribe(audio_array, language="es", beam_size=1, condition_on_previous_text=False)
            text = "".join([segment.text for segment in segments]).strip()
            self.text_transcribed.emit(text)
        except Exception as e:
            print(f"❌ [AUDIO_CORE] Error: {e}")
            self.text_transcribed.emit("")

    def process_live_input(self, audio_array):
        if self.is_live_transcribing:
            return
        self.is_live_transcribing = True
        threading.Thread(target=self._live_transcribe_thread, args=(audio_array,), daemon=True).start()

    def _live_transcribe_thread(self, audio_array):
        try:
            segments, _ = self.whisper.transcribe(audio_array, language="es", beam_size=1, condition_on_previous_text=False)
            text = "".join([s.text for s in segments]).strip()
            if text:
                self.live_text_ready.emit(text)
        except Exception:
            pass
        finally:
            self.is_live_transcribing = False