import os
import io
import base64
import urllib.request
import threading
import soundfile as sf
from kokoro_onnx import Kokoro
from PySide6.QtCore import QObject, Signal

class AssistantTTS(QObject):
    audio_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "tts"))
        # Actualizamos a los archivos de la v1.0
        self.model_path = os.path.join(self.model_dir, "kokoro-v1.0.onnx")
        self.voices_path = os.path.join(self.model_dir, "voices-v1.0.json")
        
        self.kokoro = None
        self.is_ready = False
        
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _download_file(self, url, dest):
        # Camuflaje anti-bots para que no nos tiren 403 o 404 por rashear
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())

    def _init_engine(self):
        os.makedirs(self.model_dir, exist_ok=True)
        
        if not os.path.exists(self.model_path):
            print("⏳ [TTS] Descargando modelo ONNX v1.0 (82MB)... Bancame.")
            try:
                self._download_file("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model/kokoro-v1.0.onnx", self.model_path)
            except Exception:
                print("⚠️ GitHub tiró error. Bajando desde el mirror de HuggingFace...")
                self._download_file("https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx", self.model_path)
            
        if not os.path.exists(self.voices_path):
            print("⏳ [TTS] Descargando perfiles de voces v1.0...")
            try:
                self._download_file("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model/voices-v1.0.json", self.voices_path)
            except Exception:
                print("⚠️ GitHub tiró error en voces. Bajando desde el mirror de HuggingFace...")
                self._download_file("https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices.json", self.voices_path)

        print("🧠 [TTS] Cargando motor Kokoro v1.0 en CPU...")
        try:
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            self.is_ready = True
            print("✅ [TTS] Motor de voz nativo 100% operativo.")
        except Exception as e:
            print(f"❌ [TTS] Error al cargar el motor: {e}")

    def process_text_async(self, text, voice="es_gs"):
        if not self.is_ready or not text.strip():
            return
        threading.Thread(target=self._generate_and_emit, args=(text, voice), daemon=True).start()

    def _generate_and_emit(self, text, voice):
        try:
            # Generación en tiempo récord
            samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="es")
            buffer = io.BytesIO()
            sf.write(buffer, samples, sample_rate, format='WAV')
            audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            self.audio_ready.emit(audio_b64)
        except Exception as e:
            print(f"❌ [TTS] Error generando voz: {e}")