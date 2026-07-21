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
    config_path = os.path.join(base_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    frases_data = {"fallbacks": {}}
    frases_path = os.path.join(base_dir, "frases.json")
    if os.path.exists(frases_path):
        with open(frases_path, "r", encoding="utf-8") as f:
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
    try:
        if hotkey_str.startswith("Key."):
            hotkey_obj = getattr(Key, hotkey_str.split(".")[1])
        else:
            hotkey_obj = KeyCode.from_char(hotkey_str)
    except (AttributeError, ValueError) as e:
        logger.warning(f"⚠️ [MAIN] Hotkey inválida '{hotkey_str}' ({e}), usando 'Key.alt_r' por defecto")
        hotkey_obj = Key.alt_r
    input_manager = VoiceInputManager(hotkey=hotkey_obj)

    # Variables para el modo chat
    last_transcribed_text = None
    username = config.get("username", "Usuario")
    hide_timer = None
    is_chat_mode = False
    is_immediate_phrase = False

    # Funciones de control
    def on_js_event(msg):
        logger.info(f"🔌 [MAIN] JS event recibido: {msg}")
        if msg == "TTS_ENDED":
            logger.info("🔓 [MAIN] TTS terminado, desbloqueando input")
            input_manager.set_locked(False)

    def on_recording_started():
        logger.info("🎙️ [MAIN] Grabación iniciada")
        input_manager.set_locked(True)
        
        # Capturar pantalla en el momento exacto de la grabación
        api_brain.capture_and_store_screen()
        
        # Cancelar timer de ocultamiento si existe
        nonlocal hide_timer, is_chat_mode
        if hide_timer:
            hide_timer.stop()
            hide_timer = None
            logger.info("⏱️ [MAIN] Timer de ocultamiento cancelado")
        
        # Salir del modo chat al iniciar grabación
        is_chat_mode = False
        logger.info("🔄 [MAIN] Saliendo del modo chat")
        
        # Resetear estado de chat a modo transcripción con animación
        logger.info("🔄 [MAIN] Resetando estado de chat a modo transcripción con animación")
        avatar_widget.webview.page().runJavaScript("""
            (function() {
                const chatContent = document.getElementById('chat-content');
                const transStatus = document.getElementById('transcription-status');
                const transText = document.getElementById('transcription-text');
                const volumeMeter = document.getElementById('volume-meter');
                
                console.log("🎭 [WEB] Iniciando transición de chat a transcripción");
                
                // Resetear clases de fade
                chatContent.classList.remove('fade-in', 'fade-out');
                transStatus.classList.remove('fade-in', 'fade-out');
                transText.classList.remove('fade-in', 'fade-out');
                volumeMeter.classList.remove('fade-in', 'fade-out');
                
                // Ocultar chat inmediatamente
                chatContent.style.display = 'none';
                
                // Mostrar elementos de transcripción
                transStatus.style.display = 'inline';
                transText.style.display = 'block';
                volumeMeter.style.display = 'block';
                
                // Forzar reflow para asegurar que los cambios de display se apliquen
                void transStatus.offsetWidth;
                
                // Aplicar fade in a los elementos de transcripción
                transStatus.style.opacity = '0';
                transText.style.opacity = '0';
                volumeMeter.style.opacity = '0';
                
                setTimeout(() => {
                    transStatus.style.opacity = '1';
                    transText.style.opacity = '1';
                    volumeMeter.style.opacity = '1';
                }, 50);
                
                console.log("🎭 [WEB] Transición de chat a transcripción completada");
            })();
        """)
        
        logger.info("📡 [MAIN] Llamando a toggle_recording_ui(True)")
        avatar_widget.toggle_recording_ui(True)
        avatar_widget.webview.page().runJavaScript("window.hideReadyNotification();")
        
        # Resetear estilos de la caja por si estaba oculta
        avatar_widget.webview.page().runJavaScript("""
            (function() {
                const transBox = document.getElementById('transcription-box');
                if (transBox) {
                    transBox.style.visibility = 'visible';
                    transBox.style.opacity = '1';
                    transBox.classList.remove('hiding');
                    console.log("🎭 [WEB] Estilos de caja reseteados");
                }
            })();
        """)

    def on_recording_canceled():
        logger.info("❌ [MAIN] Grabación cancelada")
        audio_core.stop_live_transcription()  # Detener streaming
        
        nonlocal is_chat_mode
        if is_chat_mode:
            # Si estamos en modo chat, solo reiniciar timer de ocultamiento
            logger.info("💬 [MAIN] En modo chat, reiniciando timer de ocultamiento")
            nonlocal hide_timer
            if hide_timer:
                hide_timer.stop()
            
            hide_timer = QTimer()
            hide_timer.setSingleShot(True)
            hide_timer.timeout.connect(lambda: hide_ui_and_avatar())
            hide_timer.start(3000)
            logger.info("⏱️ [MAIN] Timer de ocultamiento reiniciado en 3 segundos")
        else:
            # Si no estamos en modo chat, esconder todo
            logger.info("🔽 [MAIN] Ocultando caja y avatar (cancelación normal)")
            avatar_widget.hide_chat()
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
        
        # Verificar si el texto contiene algo entre corchetes (ej: [BLANK_AUDIO])
        if '[' in text and ']' in text:
            logger.warning(f"⚠️ [MAIN] Texto contiene marcadores entre corchetes: '{text}', cancelando")
            on_recording_canceled()
            return
        
        # Guardar texto transcribido para usar en modo chat
        nonlocal last_transcribed_text
        last_transcribed_text = text
        
        # Actualizar transcripción final en la caja
        logger.info("📝 [MAIN] Actualizando transcripción final en UI")
        avatar_widget.update_transcription(text)
        
        # Ya no ocultamos UI automáticamente, esperamos respuesta de API
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
                nonlocal is_immediate_phrase
                is_immediate_phrase = True
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

        # Flash de error en la ventana
        avatar_widget.flash_error()

        # Transicionar a modo chat con el texto del usuario y mostrar fallback
        nonlocal last_transcribed_text, is_chat_mode
        if last_transcribed_text:
            logger.info("💬 [MAIN] Transicionando a modo chat (error)")
            is_chat_mode = True
            avatar_widget.transition_to_chat_mode(username, last_transcribed_text)
            avatar_widget.show_assistant_response(msg, is_fallback=True)
        
        voz_activa = config.get("active_voice", "es_gs")
        tts_core.process_text_async(msg, voz_activa)

    def on_llm_text_ready(text):
        # Marcar que ahora vamos a reproducir la respuesta real (no frase inmediata)
        nonlocal is_immediate_phrase
        is_immediate_phrase = False
        
        # Transicionar a modo chat con el texto del usuario y mostrar respuesta
        nonlocal last_transcribed_text, is_chat_mode
        if last_transcribed_text:
            logger.info("💬 [MAIN] Transicionando a modo chat (respuesta)")
            is_chat_mode = True
            avatar_widget.transition_to_chat_mode(username, last_transcribed_text)
            avatar_widget.show_assistant_response(text)
        
        # Cuando Haiku devuelve la respuesta real, la pasamos al TTS
        voz_activa = config.get("active_voice", "es_gs")
        tts_core.process_text_async(text, voz_activa)

    def on_speech_started():
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("speaking");')

    def on_speech_ended():
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("idle");')
        
        # Solo programar timer si NO es frase inmediata (es respuesta real o fallback)
        nonlocal is_immediate_phrase, hide_timer
        if not is_immediate_phrase:
            input_manager.set_locked(False)
            
            # Programar ocultamiento después de 3 segundos
            hide_timer = QTimer()
            hide_timer.setSingleShot(True)
            hide_timer.timeout.connect(lambda: hide_ui_and_avatar())
            hide_timer.start(3000)
            logger.info("⏱️ [MAIN] Timer de ocultamiento programado en 3 segundos (respuesta real/fallback)")
        else:
            logger.info("⏭️ [MAIN] TTS de frase inmediata terminado, esperando respuesta del modelo")
            is_immediate_phrase = False

    def hide_ui_and_avatar():
        logger.info("🔽 [MAIN] Ocultando caja de chat y avatar")
        avatar_widget.hide_chat()
        avatar_widget.webview.page().runJavaScript('window.setAvatarState("hidden");')

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