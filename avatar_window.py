import os
import json
import random
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from PySide6.QtCore import Qt, QUrl, Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtGui import QIcon
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile, QWebEnginePage

class CustomWebPage(QWebEnginePage):
    console_message_signal = Signal(str)
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        self.console_message_signal.emit(message)
        super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)

class LocalHTTPServer:
    """Servidor HTTP local para servir archivos estáticos del directorio web/"""
    def __init__(self, web_dir, port=None):
        self.web_dir = web_dir
        self.port = port if port else random.randint(8000, 9000)
        self.server = None
        self.thread = None
        
    def start(self):
        """Inicia el servidor en un hilo separado"""
        os.chdir(self.web_dir)
        self.server = HTTPServer(('localhost', self.port), SimpleHTTPRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"🌐 [AVATAR] Servidor HTTP iniciado en http://localhost:{self.port}")
        
    def stop(self):
        """Detiene el servidor"""
        if self.server:
            self.server.shutdown()
            print(f"🌐 [AVATAR] Servidor HTTP detenido")
            
    def get_url(self, path="index.html"):
        """Retorna la URL completa para un archivo"""
        return f"http://localhost:{self.port}/{path}"

class AvatarWindow(QWidget):
    js_event = Signal(str)
    system_ready = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().geometry())
        
        # Establecer icono de la ventana
        icon_path = os.path.join(os.path.dirname(__file__), "app.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Cargar configuración para obtener el avatar activo y nombre
        self.config = {}
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        self.active_avatar = self.config.get("active_avatar", "avatar.vrm")
        self.avatar_name = self.config.get("avatar_name", "Kurumi")
        self.hotkey = self.config.get("hotkey", "Key.alt_r")
        
        # Convertir hotkey a formato legible
        hotkey_display = self.hotkey.replace("Key.", "").replace("_", " ").upper()
        if hotkey_display == "ALT R":
            hotkey_display = "Alt+R"
        self.hotkey_display = hotkey_display

        # Copiar el avatar seleccionado a web/ ANTES de iniciar el servidor
        import shutil
        source_path = os.path.join(os.path.dirname(__file__), "avatars", self.active_avatar)
        dest_path = os.path.join(os.path.dirname(__file__), "web", "avatar.vrm")
        
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                print(f"🎭 [AVATAR] Avatar copiado: {self.active_avatar} -> web/avatar.vrm")
            except Exception as e:
                print(f"❌ [AVATAR] Error copiando avatar: {e}")
        else:
            print(f"❌ [AVATAR] Avatar no encontrado: {source_path}")

        # Iniciar servidor HTTP local
        web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web'))
        self.http_server = LocalHTTPServer(web_dir)
        self.http_server.start()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.webview = QWebEngineView()
        self.profile = QWebEngineProfile("AvatarProfile", self.webview)
        self.page = CustomWebPage(self.profile, self.webview)
        self.page.setBackgroundColor(Qt.transparent)
        self.page.console_message_signal.connect(self.js_event.emit)
        self.webview.setPage(self.page)

        settings = self.webview.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)

        # Cargar desde servidor HTTP local en lugar de archivo local
        html_url = self.http_server.get_url("index.html")
        self.webview.setUrl(QUrl(html_url))
        layout.addWidget(self.webview)
        
        # Inyectar el nombre del avatar y hotkey después de cargar
        self.webview.loadFinished.connect(lambda: self.webview.page().runJavaScript(f"window.setAvatarName('{self.avatar_name}'); window.setHotkey('{self.hotkey_display}');"))
        
        # Verificar periódicamente si el sistema está listo
        self.check_ready_timer = QTimer()
        self.check_ready_timer.timeout.connect(self.check_system_ready)
        self.check_ready_timer.start(500)

    def closeEvent(self, event):
        """Cerrar sin limpieza"""
        super().closeEvent(event)

    def toggle_recording_ui(self, is_recording):
        self.webview.page().runJavaScript(f"window.toggleRecordingUI({'true' if is_recording else 'false'});")

    def update_transcription(self, text):
        texto_limpio = text.replace("'", "\\'").replace('\n', ' ')
        self.webview.page().runJavaScript(f"window.updateTranscription('{texto_limpio}');")

    def on_text_to_speak(self, text, duration):
        """Slot para recibir texto y duración del TTS y enviar al frontend (thread-safe)"""
        texto_limpio = text.replace("'", "\\'").replace('\n', ' ')
        self.webview.page().runJavaScript(f"window.startSpeaking('{texto_limpio}', {duration});")

    def check_system_ready(self):
        """Verifica si el sistema está listo y emite la señal correspondiente"""
        self.webview.page().runJavaScript("window.systemReady", self.on_ready_check)

    def on_ready_check(self, result):
        """Callback para verificar si el sistema está listo"""
        if result:
            self.check_ready_timer.stop()
            self.system_ready.emit()