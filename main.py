import sys
import os
import json
import random
import subprocess
import logging
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import QTimer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from modules.input_handler import VoiceInputManager
from modules.audio_core import AssistantAudioCore
from modules.api_brain import AssistantBrain
from modules.tts_core import AssistantTTS
from avatar_window import AvatarWindow
import setup

def run_app():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_dir, "app.ico")

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    app.setQuitOnLastWindowClosed(False)

    config = {}
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

    frases_data = {"fallbacks": {}}
    if os.path.exists("frases.json"):
        with open("frases.json", "r", encoding="utf-8") as f:
            frases_data = json.load(f)

    # Inicialización de módulos
    audio_core = AssistantAudioCore()
    api_brain = AssistantBrain()
    tts_core = AssistantTTS()
    
    avatar_widget = AvatarWindow()
    avatar_widget.show()
    
    # Conectar TTS con avatar para visemes
    tts_core.set_avatar_widget(avatar_widget)

    # Función para abrir ventana de configuraciones
    def open_config_window():
        setup.run_setup_window(from_system_tray=True)

    # System Tray
    tray_icon = QSystemTrayIcon(QIcon(icon_path), app)
    menu = QMenu()
    config_action = QAction("Configuraciones", app)
    config_action.triggered.connect(open_config_window)
    quit_action = QAction("Cerrar Asistente", app)
    quit_action.triggered.connect(app.quit)
    menu.addAction(config_action)
    menu.addAction(quit_action)
    tray_icon.setContextMenu(menu)
    tray_icon.show()
    
    # Establecer icono de la aplicación
    app.setWindowIcon(QIcon(icon_path))

    # Input Manager
    hotkey_str = config.get("hotkey", "Key.alt_r")
    from pynput.keyboard import Key, KeyCode
    if hotkey_str.startswith("Key."):
        hotkey_obj = getattr(Key, hotkey_str.split(".")[1])
    else:
        hotkey_obj = KeyCode.from_char(hotkey_str)
    input_manager = VoiceInputManager(hotkey=hotkey_obj)

    # Funciones de control
    def on_js_event(msg):
        logger.info(f"🔌 [MAIN] JS event recibido: {msg}")
        if msg == "TTS_ENDED":
            logger.info("🔓 [MAIN] TTS terminado, desbloqueando input")
            input_manager.set_locked(False)

    def on_recording_started():
        logger.info("🎙️ [MAIN] Grabación iniciada")
        input_manager.set_locked(True)
        logger.info("📡 [MAIN] Llamando a toggle_recording_ui(True)")
        avatar_widget.toggle_recording_ui(True)
        avatar_widget.webview.page().runJavaScript("window.hideReadyNotification();")

    def on_recording_canceled():
        logger.info("❌ [MAIN] Grabación cancelada")
        audio_core.stop_live_transcription()  # Detener streaming
        avatar_widget.toggle_recording_ui(False)
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("hidden");')
        input_manager.set_locked(False)

    def on_valid_audio_input(audio_array):
        logger.info(f"🎯 [MAIN] Audio válido recibido: {len(audio_array)} samples")
        logger.info("📡 [MAIN] Llamando a toggle_recording_ui(False) para cambiar a Procesando")
        # Cambiar a estado Procesando
        avatar_widget.toggle_recording_ui(False)
        # Detener streaming primero
        audio_core.stop_live_transcription()
        # No ocultar UI inmediatamente, esperar transcripción final
        QTimer.singleShot(100, lambda: audio_core.process_voice_input(audio_array))

    def on_live_text(text):
        logger.info(f"📝 [MAIN] Texto live recibido: '{text}'")
        avatar_widget.update_live_transcription(text)
    
    def on_volume_level(level):
        """Actualiza el medidor de volumen en la UI"""
        avatar_widget.webview.page().runJavaScript(f'window.updateVolumeMeter({level});')

    def on_text_transcribed(text):
        logger.info(f"✅ [MAIN] Texto transcribido: '{text}'")
        if not text.strip():
            logger.warning("⚠️ [MAIN] Texto vacío, cancelando")
            on_recording_canceled()
            return
        
        # Actualizar transcripción final en la caja
        logger.info("📝 [MAIN] Actualizando transcripción final en UI")
        avatar_widget.update_transcription(text)
        
        # Ocultar UI después de 2 segundos usando función separada
        logger.info("⏱️ [MAIN] Programando ocultamiento de UI en 2 segundos")
        def hide_ui():
            logger.info("🔽 [MAIN] Ejecutando ocultamiento de UI")
            avatar_widget.hide_recording_ui()
        
        QTimer.singleShot(2000, hide_ui)
        
        # Solo poner el avatar en thinking si hay transcripción válida
        logger.info("🤔 [MAIN] Poniendo avatar en estado thinking")
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("thinking");')
            
        api_key = config.get("api_key", "").strip()
        voz_activa = config.get("active_voice", "es_gs")

        if api_key:
            logger.info("🔑 [MAIN] API key presente, procesando respuesta")
            pool = frases_data.get("inmediatos", []) + frases_data.get("largos", [])
            if pool:
                frase_random = random.choice(pool)
                logger.info(f"🎵 [MAIN] Frase inmediata seleccionada: '{frase_random}'")
                tts_core.process_text_async(frase_random, voz_activa)
                
        logger.info("🧠 [MAIN] Enviando query al cerebro")
        api_brain.process_query_async(text)

    def on_error(error_code):
        fallbacks = frases_data.get("fallbacks", {})
        fallback_data = fallbacks.get(error_code, ["Algo reventó en el backend y no sé qué es. Revisá la consola."])

        if isinstance(fallback_data, list) and fallback_data:
            msg = random.choice(fallback_data)
        elif isinstance(fallback_data, str):
            msg = fallback_data
        else:
            msg = "Error desconocido."

        voz_activa = config.get("active_voice", "es_gs")
        tts_core.process_text_async(msg, voz_activa)

    def on_llm_text_ready(text):
        # Cuando Haiku devuelve la respuesta real, la pasamos al TTS
        voz_activa = config.get("active_voice", "es_gs")
        tts_core.process_text_async(text, voz_activa)

    def on_speech_started():
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("speaking");')

    def on_speech_ended():
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("idle");')
        input_manager.set_locked(False)

    # Pipeline
    avatar_widget.js_event.connect(on_js_event)
    avatar_widget.system_ready.connect(lambda: input_manager.set_locked(False))
    input_manager.recording_started.connect(on_recording_started)
    input_manager.recording_canceled.connect(on_recording_canceled)
    input_manager.audio_ready.connect(on_valid_audio_input)
    input_manager.audio_live_ready.connect(audio_core.process_live_input)
    input_manager.volume_level.connect(on_volume_level)
    audio_core.live_text_ready.connect(on_live_text)
    audio_core.text_transcribed.connect(on_text_transcribed)
    
    # Cerebro -> TTS -> Frontend
    api_brain.text_chunk_ready.connect(on_llm_text_ready)
    api_brain.point_action_ready.connect(lambda x, y: avatar_widget.webview.page().runJavaScript('window.apuntarFalso();'))
    api_brain.error_occurred.connect(on_error)

    tts_core.speech_started.connect(on_speech_started)
    tts_core.speech_ended.connect(on_speech_ended)
    tts_core.text_to_speak.connect(avatar_widget.on_text_to_speak)

    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()