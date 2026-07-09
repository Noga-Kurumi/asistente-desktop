import os
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile, QWebEnginePage

class CustomWebPage(QWebEnginePage):
    console_message_signal = Signal(str)
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        self.console_message_signal.emit(message)
        super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)

class AvatarWindow(QWidget):
    js_event = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().geometry())

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

        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'index.html'))
        self.webview.setUrl(QUrl.fromLocalFile(html_path))
        layout.addWidget(self.webview)

    def reproducir_base64(self, b64_audio):
        # Manda el string crudo al reproductor de JS
        self.webview.page().runJavaScript(f"window.reproducirAudioBase64('{b64_audio}');")

    def toggle_recording_ui(self, is_recording):
        self.webview.page().runJavaScript(f"window.toggleRecordingUI({'true' if is_recording else 'false'});")

    def update_transcription(self, text):
        texto_limpio = text.replace("'", "\\'").replace('\n', ' ')
        self.webview.page().runJavaScript(f"window.updateTranscription('{texto_limpio}');")