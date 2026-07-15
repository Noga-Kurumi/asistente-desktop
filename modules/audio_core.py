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
        
        # Optimizaciones para balance velocidad/precisión:
        # - Modelo "base" para mejor precisión (vs "tiny")
        # - cpu_threads=4 para mayor paralelismo
        # - num_workers=1 para evitar overhead
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)
        self.is_live_transcribing = False

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}

    def _is_valid_spanish(self, text):
        """Verifica si el texto parece ser español basándose en caracteres comunes"""
        if not text:
            return False
        
        # Caracteres comunes en español
        spanish_chars = set('áéíóúñü¿¡')
        
        # Si contiene caracteres típicos del español, es válido
        if any(char in text for char in spanish_chars):
            return True
        
        # Si no, verificar palabras comunes en español
        spanish_words = {'que', 'para', 'por', 'con', 'una', 'como', 'estar', 'todo', 'pero', 'más', 'hacer', 'puede', 'ser', 'tiene', 'este', 'hasta', 'donde', 'cuando', 'muy', 'sobre', 'otros', 'después', 'sin', 'entre', 'tiempo', 'años', 'parte', 'bien', 'gracias', 'hola', 'buenos', 'días', 'tarde', 'noche'}
        
        words = text.lower().split()
        if any(word in spanish_words for word in words):
            return True
        
        # Si el texto es corto y no tiene características de español, asumir que no es válido
        return len(text) > 20  # Solo aceptar textos largos sin características claras

    def process_voice_input(self, audio_array):
        threading.Thread(target=self._final_transcribe_thread, args=(audio_array,), daemon=True).start()

    def _final_transcribe_thread(self, audio_array):
        try:
            # Intentar primero con español
            segments, _ = self.whisper.transcribe(audio_array, language="es", beam_size=1, condition_on_previous_text=False)
            text = "".join([segment.text for segment in segments]).strip()
            
            # Verificar si el texto parece ser español (caracteres válidos)
            if self._is_valid_spanish(text):
                self.text_transcribed.emit(text)
            else:
                # Si no parece español, intentar con inglés
                segments, _ = self.whisper.transcribe(audio_array, language="en", beam_size=1, condition_on_previous_text=False)
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