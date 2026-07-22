import os
import threading
import logging
from PySide6.QtCore import QObject, Signal
from modules.whisper_wrapper import WhisperCppWrapper
from modules.config_manager import get_config

logger = logging.getLogger(__name__)


class AssistantAudioCore(QObject):
    text_transcribed = Signal(str)
    live_text_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.config = get_config()

        # Lock para evitar conflictos de hilos
        self.whisper_lock = threading.Lock()

        # Flag para evitar múltiples transcripciones live simultáneas
        self.is_live_transcribing = False

        # Buffer de audio para streaming
        self.live_audio_buffer = []

        # Inicializar wrapper de whisper.cpp
        base_dir = os.path.dirname(os.path.dirname(__file__))
        stream_exe_path = os.path.join(base_dir, "whisper_cpp", "Release", "whisper-stream-pcm.exe")

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
            stream_exe_path=stream_exe_path,
            model_path=model_path,
            language='auto',  # Detección automática en la transcripción final
            n_threads=2
        )

        logger.info(f"✅ [AUDIO_CORE] Modelo de whisper: {whisper_model}, Cuantización: {whisper_quant}")

    def process_voice_input(self, audio_array):
        logger.info(f"🎯 [AUDIO_CORE] Iniciando transcripción final: {len(audio_array)} samples")
        threading.Thread(target=self._final_transcribe_thread, args=(audio_array,), daemon=True).start()

    def _final_transcribe_thread(self, audio_array):
        try:
            # Usar lock para evitar conflictos de hilos
            with self.whisper_lock:
                # Una sola pasada: whisper detecta el idioma automáticamente
                success, text = self.whisper.transcribe(audio_array)

                if success and text:
                    logger.info(f"✅ [AUDIO_CORE] Transcripción final: '{text}'")
                    self.text_transcribed.emit(text)
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
