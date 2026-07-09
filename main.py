import sys
import os
import json
import random
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import QTimer

from modules.input_handler import VoiceInputManager
from modules.audio_core import AssistantAudioCore
from modules.api_brain import AssistantBrain
from modules.tts_core import AssistantTTS
from avatar_window import AvatarWindow
import setup

def run_app():
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

    # System Tray
    tray_icon = QSystemTrayIcon(QIcon("app.ico"), app)
    menu = QMenu()
    config_action = QAction("Configuraciones", app)
    config_action.triggered.connect(setup.run_setup_window)
    quit_action = QAction("Cerrar Asistente", app)
    quit_action.triggered.connect(app.quit)
    menu.addAction(config_action)
    menu.addAction(quit_action)
    tray_icon.setContextMenu(menu)
    tray_icon.show()

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
        if msg == "TTS_ENDED":
            input_manager.set_locked(False)

    def on_recording_started():
        input_manager.set_locked(True)
        avatar_widget.toggle_recording_ui(True)

    def on_recording_canceled():
        avatar_widget.toggle_recording_ui(False)
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("idle");')
        input_manager.set_locked(False)

    def on_valid_audio_input(audio_array):
        avatar_widget.toggle_recording_ui(False)
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("thinking");')
        QTimer.singleShot(100, lambda: audio_core.process_voice_input(audio_array))

    def on_live_text(text):
        avatar_widget.update_transcription(text)

    def on_text_transcribed(text):
        if not text.strip():
            on_recording_canceled()
            return
            
        api_key = config.get("api_key", "").strip()
        voz_activa = config.get("active_voice", "es_gs")

        if api_key:
            pool = frases_data.get("inmediatos", []) + frases_data.get("largos", [])
            if pool:
                frase_random = random.choice(pool)
                # Mandamos a generar la voz en Python
                tts_core.process_text_async(frase_random, voz_activa)
                
        api_brain.process_query_async(text)

    def on_error(error_code):
        fallbacks = frases_data.get("fallbacks", {})
        msg = fallbacks.get(error_code, "Algo reventó en el backend.")
        voz_activa = config.get("active_voice", "es_gs")
        tts_core.process_text_async(msg, voz_activa)

    def on_llm_text_ready(text):
        # Cuando Haiku devuelve la respuesta real, la pasamos al TTS
        voz_activa = config.get("active_voice", "es_gs")
        tts_core.process_text_async(text, voz_activa)

    def on_audio_ready(b64_audio):
        # El TTS nativo generó el audio, lo pasamos al JS para reproducir
        avatar_widget.reproducir_base64(b64_audio)

    # Pipeline
    avatar_widget.js_event.connect(on_js_event)
    input_manager.recording_started.connect(on_recording_started)
    input_manager.recording_canceled.connect(on_recording_canceled)
    input_manager.audio_ready.connect(on_valid_audio_input)
    input_manager.audio_live_ready.connect(audio_core.process_live_input)
    audio_core.live_text_ready.connect(on_live_text)
    audio_core.text_transcribed.connect(on_text_transcribed)
    
    # Cerebro -> TTS -> Frontend
    api_brain.text_chunk_ready.connect(on_llm_text_ready)
    api_brain.point_action_ready.connect(lambda x, y: avatar_widget.webview.page().runJavaScript('window.apuntarFalso();'))
    api_brain.error_occurred.connect(on_error)
    
    tts_core.audio_ready.connect(on_audio_ready)

    print("✅ [MAIN] Pipeline nativo conectado.")
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()