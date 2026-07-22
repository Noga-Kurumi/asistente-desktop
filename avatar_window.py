"""Ventana del avatar 3D (PySide6 + QWebEngineView).

Sirve web/ y avatars/ por HTTP local, aloja la página three.js + three-vrm y
expone la API Python → JS (set_avatar_state, on_text_to_speak, etc.). Los
eventos JS → Python viajan por QWebChannel (AvatarBridge.registerObject como
"bridge"); javaScriptConsoleMessage queda solo como log de depuración (DEBUG).

Reglas de la casa: configuración solo vía modules.config_manager, errores
atrapados y logueados, nada de prints.
"""

import functools
import json
import logging
import os
import random
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import quote, unquote, urlsplit

from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from modules.config_manager import get_config

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
AVATARS_DIR = os.path.join(BASE_DIR, "avatars")
PORT_MIN, PORT_MAX = 8000, 9000
PORT_MAX_ATTEMPTS = 20


class CustomWebPage(QWebEnginePage):
    """Página que deja los console.log del JS como log DEBUG (ya no son canal
    de eventos: para eso está el QWebChannel)."""

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        logger.debug("[WEB] %s (%s:%s)", message, sourceID, lineNumber)
        super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)


class AvatarHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Sirve web/ y mapea además /avatars/<file> al directorio avatars/.

    Protección contra path traversal: se resuelve la ruta real y se verifica
    que quede dentro de avatars/; si no, se devuelve una ruta inexistente
    (el servidor responde 404).
    """

    def __init__(self, *args, web_dir, avatars_dir, **kwargs):
        self.avatars_root = os.path.realpath(avatars_dir)
        super().__init__(*args, directory=web_dir, **kwargs)

    def translate_path(self, path):
        clean = urlsplit(path).path
        if clean.startswith("/avatars/"):
            rel = unquote(clean[len("/avatars/"):])
            candidate = os.path.realpath(os.path.join(self.avatars_root, rel))
            if candidate.startswith(self.avatars_root + os.sep):
                return candidate
            # Fuera de avatars/: ruta inexistente dentro de web/ → 404.
            logger.warning("⚠️ [AVATAR] Path traversal bloqueado: %s", path)
            return os.path.join(self.directory, "__forbidden__")
        return super().translate_path(path)

    def log_message(self, format, *args):
        logger.debug("[HTTP] %s", format % args)


class _HTTPServerNoReuse(HTTPServer):
    # HTTPServer activa SO_REUSEADDR por defecto y en Windows eso permitiría
    # un segundo bind al mismo puerto; sin él, un puerto ocupado falla y el
    # bucle de abajo reintenta con otro.
    allow_reuse_address = False


class LocalHTTPServer:
    """Servidor HTTP local para web/ y avatars/ en un puerto aleatorio."""

    def __init__(self, web_dir, avatars_dir, port=None):
        self.web_dir = web_dir
        self.avatars_dir = avatars_dir
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Inicia el servidor reintentando con otro puerto si está ocupado."""
        handler = functools.partial(
            AvatarHTTPRequestHandler,
            web_dir=self.web_dir,
            avatars_dir=self.avatars_dir,
        )
        last_error = None
        for _ in range(PORT_MAX_ATTEMPTS):
            self.port = self.port or random.randint(PORT_MIN, PORT_MAX)
            try:
                self.server = _HTTPServerNoReuse(("localhost", self.port), handler)
                break
            except OSError as e:
                last_error = e
                logger.warning("⚠️ [AVATAR] Puerto %d ocupado (%s), reintentando",
                               self.port, e)
                self.port = None
        else:
            logger.error("❌ [AVATAR] Sin puerto libre tras %d intentos: %s",
                         PORT_MAX_ATTEMPTS, last_error)
            raise RuntimeError(
                f"No se encontró puerto libre en {PORT_MIN}-{PORT_MAX} "
                f"tras {PORT_MAX_ATTEMPTS} intentos") from last_error
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info("🌐 [AVATAR] Servidor HTTP en http://localhost:%d", self.port)

    def stop(self):
        """Detiene el servidor."""
        if self.server:
            self.server.shutdown()

    def get_url(self, path="index.html"):
        """Retorna la URL completa para un archivo."""
        return f"http://localhost:{self.port}/{path}"


class AvatarBridge(QObject):
    """Objeto expuesto al JS vía QWebChannel como `bridge`.

    El JS llama bridge.jsEvent(JSON.stringify({type: ...})). Se mantienen las
    señales legacy que main.py consume hoy: js_event(str) con el payload
    histórico ("TTS_ENDED") y system_ready().
    """

    js_event = Signal(str)
    system_ready = Signal()

    @Slot(str)
    def jsEvent(self, payload: str):
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("⚠️ [AVATAR] Evento JS no es JSON válido: %s", payload)
            return
        event_type = event.get("type")
        if event_type == "ready":
            logger.info("✅ [AVATAR] Sistema del avatar listo (QWebChannel)")
            self.system_ready.emit()
        elif event_type == "tts_ended":
            self.js_event.emit("TTS_ENDED")
        else:
            logger.debug("[AVATAR] Evento JS sin handler: %s", event)
            self.js_event.emit(payload)


class AvatarWindow(QWidget):
    js_event = Signal(str)
    system_ready = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().geometry())

        # Establecer icono de la ventana
        icon_path = os.path.join(BASE_DIR, "app.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Configuración centralizada (sin lecturas sueltas de config.json)
        self.config = get_config()
        self.active_avatar = self.config.get("active_avatar", "Lindsay.vrm")
        self.avatar_name = self.config.get("avatar_name", "Lindsay")
        self.hotkey = self.config.get("hotkey", "Key.alt_r")

        if not os.path.exists(os.path.join(AVATARS_DIR, self.active_avatar)):
            logger.error("❌ [AVATAR] Avatar no encontrado: %s",
                         os.path.join(AVATARS_DIR, self.active_avatar))

        # Convertir hotkey a formato legible
        hotkey_display = str(self.hotkey).replace("Key.", "").replace("_", " ").upper()
        if hotkey_display == "ALT R":
            hotkey_display = "Alt+R"
        self.hotkey_display = hotkey_display

        # Servidor HTTP local (sirve web/ y avatars/; ya no hay copia a web/)
        self.http_server = LocalHTTPServer(WEB_DIR, AVATARS_DIR)
        self.http_server.start()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.webview = QWebEngineView()
        self.profile = QWebEngineProfile("AvatarProfile", self.webview)
        self.page = CustomWebPage(self.profile, self.webview)
        self.page.setBackgroundColor(Qt.transparent)
        self.webview.setPage(self.page)

        # QWebChannel: eventos JS → Python (sustituye al hack de console.log
        # y al polling de window.systemReady)
        self.bridge = AvatarBridge(self.page)
        self.bridge.js_event.connect(self.js_event.emit)
        self.bridge.system_ready.connect(self.system_ready.emit)
        self.channel = QWebChannel(self.page)
        self.channel.registerObject("bridge", self.bridge)
        self.page.setWebChannel(self.channel)

        settings = self.webview.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)

        # El avatar se pasa por query string para que el JS lo tenga disponible
        # antes de evaluar el módulo (runJavaScript llegaría tarde).
        html_url = self.http_server.get_url(
            "index.html?avatar=" + quote(str(self.active_avatar)))
        self.webview.setUrl(QUrl(html_url))
        layout.addWidget(self.webview)

        # Inyectar el nombre del avatar y hotkey después de cargar
        # json.dumps produce un literal de string JS seguro (escapa comillas, backslashes, etc.)
        self.webview.loadFinished.connect(lambda: self._run_js(
            f"window.setAvatarName({json.dumps(self.avatar_name)}); "
            f"window.setHotkey({json.dumps(self.hotkey_display)});"))

    # ------------------------------------------------------------- helpers

    def _run_js(self, snippet: str) -> None:
        """Ejecuta JS en la página; las excepciones se loguean, no propagan."""
        try:
            self.webview.page().runJavaScript(snippet)
        except Exception as e:
            logger.error("❌ [AVATAR] Error ejecutando JS: %s", e, exc_info=True)

    # --------------------------------------------------------- API Python → JS

    def toggle_recording_ui(self, is_recording):
        self._run_js(f"window.toggleRecordingUI({'true' if is_recording else 'false'});")

    def hide_recording_ui(self):
        self._run_js("window.hideRecordingUI();")

    def update_transcription(self, text):
        self._run_js(f"window.updateTranscription({json.dumps(text)});")

    def update_live_transcription(self, text):
        self._run_js(f"window.updateLiveTranscription({json.dumps(text)});")

    def transition_to_chat_mode(self, user_name, user_text):
        self._run_js(
            f"window.transitionToChatMode({json.dumps(user_name)}, {json.dumps(user_text)});")

    def show_assistant_response(self, response, is_fallback=False):
        self._run_js(
            f"window.showAssistantResponse({json.dumps(response)}, {str(is_fallback).lower()});")

    def update_assistant_response(self, text):
        """Actualización incremental del chat: reemplaza el contenido con el
        texto completo acumulado hasta ahora (sin animaciones ni resets)."""
        self._run_js(f"window.updateAssistantResponse({json.dumps(text)});")

    def hide_chat(self):
        self._run_js("window.hideChat();")

    def flash_error(self):
        self._run_js("window.flashError();")

    def set_avatar_state(self, state: str):
        """Cambia el estado del avatar (idle, thinking, speaking, hidden)."""
        self._run_js(f"window.setAvatarState({json.dumps(state)});")

    def hide_ready_notification(self):
        """Oculta la notificación de 'Sistema Listo'."""
        self._run_js("window.hideReadyNotification();")

    def update_volume_meter(self, level: float):
        """Actualiza el medidor de volumen del micrófono (0..1)."""
        self._run_js(f"window.updateVolumeMeter({float(level)});")

    def point_at(self, x: int, y: int):
        """Gesto de apuntar del avatar (apuntarFalso en el JS)."""
        self._run_js(f"window.apuntarFalso({int(x)}, {int(y)});")

    def reset_transcription_ui(self):
        """Vuelve la caja de chat a modo transcripción para un turno nuevo."""
        self._run_js("window.resetToTranscriptionMode();")

    def on_text_to_speak(self, text, duration, timeline):
        """Slot del TTS: texto, duración (s) y timeline de visemas al frontend."""
        self._run_js(
            f"window.setVisemeTimeline({json.dumps(timeline)}); "
            f"window.startSpeaking({json.dumps(text)}, {float(duration)});")

    # ------------------------------------------------------------- Qt events

    def closeEvent(self, event):
        """Cerrar sin limpieza"""
        super().closeEvent(event)
