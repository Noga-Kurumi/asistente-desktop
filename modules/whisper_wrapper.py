"""
Wrapper limpio y robusto para whisper.cpp nativo
Encapsula toda la lógica de interacción con whisper-cli.exe y whisper-stream-pcm.exe
"""

import os
import subprocess
import tempfile
import logging
import wave
import numpy as np
import threading
import queue
from typing import Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class WhisperCppWrapper:
    """Wrapper robusto para whisper.cpp nativo"""
    
    def __init__(self, exe_path: str, model_path: str, language: str = 'es', n_threads: int = 2):
        """
        Inicializa el wrapper de whisper.cpp
        
        Args:
            exe_path: Ruta al ejecutable whisper-cli.exe
            model_path: Ruta al modelo ggml-tiny.bin
            language: Idioma de transcripción (default: 'es')
            n_threads: Número de threads (default: 2)
        """
        self.exe_path = exe_path
        self.model_path = model_path
        self.language = language
        self.n_threads = n_threads
        
        # Buscar whisper-stream-pcm.exe en el mismo directorio
        exe_dir = os.path.dirname(exe_path)
        self.stream_pcm_exe = os.path.join(exe_dir, "whisper-stream-pcm.exe")
        
        # Validar que los archivos existan
        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"whisper-cli.exe no encontrado en {exe_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        
        logger.info(f"✅ [WHISPER_WRAPPER] Inicializado: {exe_path}")
        logger.info(f"✅ [WHISPER_WRAPPER] Modelo: {model_path}")
        logger.info(f"✅ [WHISPER_WRAPPER] Stream PCM: {self.stream_pcm_exe} (existe: {os.path.exists(self.stream_pcm_exe)})")
        
        # Variables para streaming
        self.stream_process = None
        self.stream_queue = None
        self.stream_reader_thread = None
        self.stream_running = False
    
    def _save_audio_to_wav(self, audio_array: np.ndarray, temp_path: str) -> None:
        """
        Guarda audio numpy como WAV temporal
        
        Args:
            audio_array: Array numpy con audio float32
            temp_path: Ruta donde guardar el archivo WAV
        """
        # Convertir a int16 para WAV
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_int16.tobytes())
        
        logger.debug(f"💾 [WHISPER_WRAPPER] Audio guardado en {temp_path}")
    
    def _build_command(self, wav_path: str, output_prefix: str) -> list:
        """
        Construye el comando para whisper-cli.exe
        
        Args:
            wav_path: Ruta al archivo WAV de entrada
            output_prefix: Prefijo para archivos de salida
            
        Returns:
            Lista de argumentos para subprocess
        """
        return [
            self.exe_path,
            '-m', self.model_path,
            '-f', wav_path,
            '-l', self.language,
            '-otxt',
            '-of', output_prefix,
            '-t', str(self.n_threads)
        ]
    
    def transcribe(self, audio_array: np.ndarray, timeout: int = 30) -> Tuple[bool, str]:
        """
        Transcribe audio usando whisper-cli.exe
        
        Args:
            audio_array: Array numpy con audio float32
            timeout: Timeout en segundos (default: 30)
            
        Returns:
            Tupla (success, text) donde success es True si la transcripción fue exitosa
        """
        temp_wav = None
        try:
            # Crear directorio temporal local para evitar problemas de rutas
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Crear archivo temporal WAV en directorio local
            import uuid
            wav_filename = f"temp_{uuid.uuid4().hex}.wav"
            wav_path = os.path.join(temp_dir, wav_filename)
            output_prefix = os.path.join(temp_dir, f"temp_{uuid.uuid4().hex}")
            
            # Guardar audio como WAV
            self._save_audio_to_wav(audio_array, wav_path)
            
            # Construir comando
            cmd = self._build_command(wav_path, output_prefix)
            
            logger.info(f"🚀 [WHISPER_WRAPPER] Ejecutando transcripción: {' '.join(cmd)}")
            
            # Ejecutar whisper-cli.exe
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            logger.debug(f"📄 [WHISPER_WRAPPER] Stdout: {result.stdout}")
            logger.debug(f"📄 [WHISPER_WRAPPER] Stderr: {result.stderr}")
            
            # Leer resultado del archivo de texto generado
            txt_path = output_prefix + '.txt'
            logger.debug(f"🔍 [WHISPER_WRAPPER] Buscando archivo: {txt_path}")
            logger.debug(f"🔍 [WHISPER_WRAPPER] Archivo existe: {os.path.exists(txt_path)}")
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                os.unlink(txt_path)
                
                logger.info(f"✅ [WHISPER_WRAPPER] Transcripción completada: '{text}'")
                return True, text
            else:
                logger.warning(f"⚠️ [WHISPER_WRAPPER] No se generó archivo de texto")
                logger.debug(f"📄 [WHISPER_WRAPPER] Stdout: {result.stdout}")
                logger.debug(f"📄 [WHISPER_WRAPPER] Stderr: {result.stderr}")
                return False, ""
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ [WHISPER_WRAPPER] Timeout después de {timeout}s")
            return False, ""
        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error en transcripción: {e}", exc_info=True)
            return False, ""
        finally:
            # Limpiar archivos temporales
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                    logger.debug(f"🗑️ [WHISPER_WRAPPER] Archivo WAV temporal eliminado: {wav_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [WHISPER_WRAPPER] Error eliminando WAV temporal: {e}")
            
            if txt_path and os.path.exists(txt_path):
                try:
                    os.unlink(txt_path)
                    logger.debug(f"🗑️ [WHISPER_WRAPPER] Archivo TXT temporal eliminado: {txt_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [WHISPER_WRAPPER] Error eliminando TXT temporal: {e}")
    
    
    def start_streaming(self, callback: Callable[[str], None]) -> bool:
        """
        Inicia el streaming real con whisper-stream-pcm.exe
        
        Args:
            callback: Función que será llamada con cada segmento de transcripción
            
        Returns:
            True si el streaming se inició correctamente
        """
        if not os.path.exists(self.stream_pcm_exe):
            logger.error(f"❌ [WHISPER_WRAPPER] whisper-stream-pcm.exe no encontrado")
            return False
        
        if self.stream_running:
            logger.warning("⚠️ [WHISPER_WRAPPER] Streaming ya está en curso")
            return False
        
        try:
            # Construir comando para whisper-stream-pcm.exe
            cmd = [
                self.stream_pcm_exe,
                '-m', self.model_path,
                '-l', self.language,
                '-t', str(self.n_threads),
                '-i', '-',  # stdin
                '--format', 'f32',
                '--sample-rate', '16000',
                '--vad',  # Habilitar VAD
                '--step', '200',  # Reducido de 500ms a 200ms para segmentos más frecuentes
                '--length', '2000'  # Reducido de 5000ms a 2000ms para menor delay
            ]
            
            logger.info(f"🚀 [WHISPER_WRAPPER] Iniciando streaming real: {' '.join(cmd)}")
            
            # Iniciar proceso
            self.stream_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Crear cola para comunicación
            self.stream_queue = queue.Queue()
            
            # Iniciar hilo para leer stdout
            self.stream_reader_thread = threading.Thread(
                target=self._read_stream_output,
                args=(callback,),
                daemon=True
            )
            self.stream_reader_thread.start()
            
            self.stream_running = True
            logger.info("✅ [WHISPER_WRAPPER] Streaming iniciado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error iniciando streaming: {e}", exc_info=True)
            return False
    
    def _read_stream_output(self, callback: Callable[[str], None]):
        """
        Hilo que lee stdout del proceso de streaming y llama al callback
        
        Args:
            callback: Función que será llamada con cada segmento de transcripción
        """
        try:
            for line in self.stream_process.stdout:
                line = line.decode('utf-8', errors='ignore').strip()
                if line:
                    logger.debug(f"📄 [WHISPER_WRAPPER] Línea stdout: '{line}'")
                    
                    # Parsear línea de transcripción
                    # Formato: [00:00:00.000 --> 00:00:05.000]   texto
                    if '-->' in line and line.startswith('['):
                        # Extraer texto después del timestamp
                        parts = line.split(']')
                        if len(parts) > 1:
                            text = parts[1].strip()
                            if text and text != '[MÚSICA]' and text != '[MUSIC]':
                                logger.info(f"📝 [WHISPER_WRAPPER] Segmento streaming: '{text}'")
                                callback(text)
        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error leyendo stream output: {e}", exc_info=True)
    
    def send_audio_chunk(self, audio_array: np.ndarray) -> bool:
        """
        Envía un chunk de audio al proceso de streaming
        
        Args:
            audio_array: Array numpy con audio float32
            
        Returns:
            True si el chunk se envió correctamente
        """
        if not self.stream_running or not self.stream_process:
            logger.warning("⚠️ [WHISPER_WRAPPER] Streaming no está iniciado")
            return False
        
        try:
            # Convertir a bytes
            audio_bytes = audio_array.astype(np.float32).tobytes()
            
            # Escribir a stdin
            self.stream_process.stdin.write(audio_bytes)
            self.stream_process.stdin.flush()
            
            logger.debug(f"📡 [WHISPER_WRAPPER] Chunk enviado: {len(audio_array)} samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ [WHISPER_WRAPPER] Error enviando chunk: {e}", exc_info=True)
            return False
    
    def stop_streaming(self):
        """Detiene el streaming y limpia recursos"""
        if not self.stream_running:
            return
        
        logger.info("🛑 [WHISPER_WRAPPER] Deteniendo streaming")
        
        self.stream_running = False
        
        if self.stream_process:
            try:
                self.stream_process.stdin.close()
                self.stream_process.terminate()
                self.stream_process.wait(timeout=5)
            except:
                self.stream_process.kill()
            self.stream_process = None
        
        self.stream_queue = None
        self.stream_reader_thread = None
        logger.info("✅ [WHISPER_WRAPPER] Streaming detenido")
