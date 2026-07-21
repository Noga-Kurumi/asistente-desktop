import os
import json
import time
import threading
import numpy as np
from PySide6.QtCore import QObject, Signal
import logging
from modules.whisper_wrapper import WhisperCppWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

class AssistantAudioCore(QObject):
    text_transcribed = Signal(str)      
    live_text_ready = Signal(str)       

    def __init__(self):
        super().__init__()
        self.config = self._load_config()
        
        # Lock para evitar conflictos de hilos
        self.whisper_lock = threading.Lock()
        
        # Flag para evitar múltiples transcripciones live simultáneas
        self.is_live_transcribing = False
        
        # Buffer de audio para streaming
        self.live_audio_buffer = []
        
        # Inicializar wrapper de whisper.cpp
        base_dir = os.path.dirname(os.path.dirname(__file__))
        exe_path = os.path.join(base_dir, "whisper_cpp", "Release", "whisper-cli.exe")
        
        # Leer modelo y cuantización desde configuración
        whisper_model = self.config.get("whisper_model", "tiny")
        whisper_quant = self.config.get("whisper_quantization", "none")
        
        # Construir nombre del archivo según cuantización
        if whisper_quant == "none":
            model_filename = f"ggml-{whisper_model}.bin"
        else:
            model_filename = f"ggml-{whisper_model}-{whisper_quant}.bin"
        
        model_path = os.path.join(base_dir, "models", model_filename)
        
        self.whisper = WhisperCppWrapper(
            exe_path=exe_path,
            model_path=model_path,
            language='es',
            n_threads=2
        )
        
        logger.info(f"✅ [AUDIO_CORE] Modelo de whisper: {whisper_model}, Cuantización: {whisper_quant}")

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
        logger.info(f"🎯 [AUDIO_CORE] Iniciando transcripción final: {len(audio_array)} samples")
        threading.Thread(target=self._final_transcribe_thread, args=(audio_array,), daemon=True).start()

    def _final_transcribe_thread(self, audio_array):
        try:
            # Usar lock para evitar conflictos de hilos
            with self.whisper_lock:
                # Transcribir usando el wrapper
                success, text = self.whisper.transcribe(audio_array, timeout=30)
                
                if success and text:
                    # Verificar si el texto parece ser español
                    if self._is_valid_spanish(text):
                        logger.info(f"🇪🇸 [AUDIO_CORE] Texto válido en español, emitiendo")
                        self.text_transcribed.emit(text)
                    else:
                        logger.info(f"🔄 [AUDIO_CORE] Texto no parece español, intentando inglés")
                        # Cambiar idioma temporalmente
                        self.whisper.language = 'en'
                        success_en, text_en = self.whisper.transcribe(audio_array, timeout=30)
                        self.whisper.language = 'es'  # Restaurar español
                        
                        if success_en and text_en:
                            logger.info(f"✅ [AUDIO_CORE] Transcripción en inglés: '{text_en}'")
                            self.text_transcribed.emit(text_en)
                        else:
                            logger.warning("⚠️ [AUDIO_CORE] Transcripción en inglés falló")
                            self.text_transcribed.emit("")
                else:
                    logger.warning("⚠️ [AUDIO_CORE] Transcripción final falló o vacía")
                    self.text_transcribed.emit("")
                    
        except Exception as e:
            logger.error(f"❌ [AUDIO_CORE] Error en transcripción final: {e}", exc_info=True)
            self.text_transcribed.emit("")

    def process_live_input(self, audio_array):
        # Iniciar streaming real con whisper-stream-pcm.exe
        if self.is_live_transcribing:
            # Si ya está transcribiendo, enviar chunk de audio
            if self.whisper.stream_running:
                self.whisper.send_audio_chunk(audio_array)
            return
        
        self.is_live_transcribing = True
        
        # Iniciar streaming con callback
        success = self.whisper.start_streaming(self._stream_callback)
        
        if success:
            # Enviar primer chunk
            self.whisper.send_audio_chunk(audio_array)
        else:
            logger.error("❌ [AUDIO_CORE] No se pudo iniciar streaming")
            self.is_live_transcribing = False
    
    def _stream_callback(self, text: str):
        """Callback llamado cuando se recibe un segmento de transcripción"""
        self.live_text_ready.emit(text)
    
    def stop_live_transcription(self):
        """Detiene el streaming real"""
        if self.is_live_transcribing:
            self.whisper.stop_streaming()
            self.is_live_transcribing = False
            self.live_audio_buffer = []