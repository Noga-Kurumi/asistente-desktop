import os
import sys
import subprocess
import threading
import time
import numpy as np
from PySide6.QtCore import QObject, Signal

def _instalar_dependencias():
    try:
        import kokoro_onnx
        import sounddevice
    except ImportError:
        print("⏳ [TTS] Instalando dependencias kokoro-onnx y sounddevice...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kokoro-onnx", "sounddevice"])

_instalar_dependencias()

import sounddevice as sd
from kokoro_onnx import Kokoro

class AssistantTTS(QObject):
    speech_started = Signal()
    speech_ended = Signal()

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
            print(f"❌ [TTS] Falta el modelo o las voces en la raíz del proyecto")
            return

        try:
            self.kokoro = Kokoro(modelo_a_usar, self.voices_path)
            
            fin = time.perf_counter()
            self.is_ready = True
            print(f"✅ [TTS] Motor listo en {(fin-inicio):.2f}s")
        except Exception as e:
            print(f"❌ [TTS] Error al cargar: {e}")

    def process_text_async(self, text, voice="af_bella"):
        if not self.is_ready or not text.strip():
            return
        threading.Thread(target=self._generate_and_play, args=(text, voice), daemon=True).start()

    def _generate_and_play(self, text, voice):
        """Genera audio y reproduce"""
        try:
            samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="es")
            
            self.speech_started.emit()
            sd.play(samples, sample_rate)
            sd.wait()
            self.speech_ended.emit()
            
        except Exception as e:
            print(f"❌ [TTS] Error en generación/reproducción: {e}")
            self.speech_ended.emit()

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
        print(f"❌ [TTS] Error al cargar voces: {e}")
        return []