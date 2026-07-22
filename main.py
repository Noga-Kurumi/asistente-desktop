"""Orquestación principal del asistente de voz.

AssistantApp conecta los módulos (input, audio, LLM, TTS) con la ventana del
avatar mediante una máquina de estados explícita (modules/state_machine):

    IDLE → RECORDING → TRANSCRIBING → THINKING → SPEAKING → IDLE
    ERROR es transitorio: se entra desde cualquier estado y se vuelve a IDLE.

Reglas de la casa:
- setup_logging() se llama UNA vez aquí (launch.py también la llama antes; es
  idempotente). Nada de basicConfig ni prints.
- La configuración se lee exclusivamente vía modules.config_manager.
- Todo slot va protegido con @_slot_guard: las excepciones se loguean y no
  tumban el loop de Qt.
"""

import json
import logging
import os
import random
import sys
from functools import wraps

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from modules import model_manager
from modules.api_brain import AssistantBrain
from modules.audio_core import AssistantAudioCore
from modules.config_manager import get_config
from modules.input_handler import VoiceInputManager
from modules.log_setup import setup_logging
from modules.state_machine import AssistantStateMachine, State
from modules.tts_core import AssistantTTS

import setup
from avatar_window import AvatarWindow

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, "app.ico")
FRASES_PATH = os.path.join(BASE_DIR, "frases.json")
HIDE_DELAY_MS = 3000
DEFAULT_VOICE = "ef_dora"

def _slot_guard(fn):
    """Decorador de slots: loguea cualquier excepción en vez de propagarla."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error("❌ [MAIN] Excepción en slot %s: %s", fn.__name__, e,
                         exc_info=True)
    return wrapper


class _DownloadSignals(QObject):
    """Puente para que el worker de descarga avise al hilo de la UI."""

    finished = Signal(list)  # rutas descargadas por model_manager.ensure_models


class _ModelDownloadWorker(QRunnable):
    """Descarga los modelos que falten en el QThreadPool global."""

    def __init__(self, config_snapshot: dict, signals: _DownloadSignals):
        super().__init__()
        self._config = config_snapshot
        self._signals = signals

    def run(self) -> None:
        try:
            downloaded = model_manager.ensure_models(self._config)
        except Exception as e:
            logger.error("❌ [MAIN] Error descargando modelos: %s", e, exc_info=True)
            downloaded = []
        self._signals.finished.emit(downloaded)


class AssistantApp(QObject):
    """Orquestador del asistente: dueño de los módulos y del estado del turno."""

    def __init__(self, app: QApplication):
        # parent=app: Qt retiene el objeto aunque no haya referencias Python.
        super().__init__(app)
        self.app = app
        self.config = get_config()
        self.frases_data = self._load_frases()
        self.sm = AssistantStateMachine(self)

        # Estado del turno (antes flags nonlocal de run_app).
        self.last_transcribed_text: str | None = None
        self.is_immediate_phrase = False
        self.chat_visible = False
        self.username = self.config.get("username", "") or "Usuario"

        self._last_audio = None          # audio del turno, para reintentar
        self._download_in_progress = False
        self._download_signals = _DownloadSignals(self)

        # Timer único de ocultamiento (antes se creaba uno nuevo cada vez).
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._hide_ui_and_avatar)

        # Módulos.
        self.audio_core = AssistantAudioCore()
        self.api_brain = AssistantBrain()
        self.tts_core = AssistantTTS()

        self.avatar = AvatarWindow()
        self.avatar.show()
        self.tts_core.set_avatar_widget(self.avatar)

        self.input_manager = VoiceInputManager(hotkey=self._parse_hotkey())

        self._setup_tray()
        self._connect_signals()

        # Avisos de modelos ausentes (DESPUÉS de conectar las señales).
        self.audio_core.notify_model_missing()
        self._check_tts_models()

        logger.info("✅ [MAIN] AssistantApp inicializada")

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _load_frases() -> dict:
        try:
            with open(FRASES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("❌ [MAIN] No se pudo leer frases.json: %s", e, exc_info=True)
            return {"fallbacks": {}}

    def _parse_hotkey(self):
        from pynput.keyboard import Key, KeyCode

        hotkey_str = str(self.config.get("hotkey", "Key.alt_r"))
        try:
            if hotkey_str.startswith("Key."):
                return getattr(Key, hotkey_str.split(".", 1)[1])
            if len(hotkey_str) != 1:
                # KeyCode.from_char NO valida: con "f2" crearía un KeyCode
                # basura que nunca coincide con una tecla real.
                raise ValueError("se esperaba 'Key.<nombre>' o un solo carácter")
            return KeyCode.from_char(hotkey_str)
        except (AttributeError, ValueError) as e:
            logger.warning("⚠️ [MAIN] Hotkey inválida '%s' (%s); usando Key.alt_r",
                           hotkey_str, e)
            return Key.alt_r

    def _setup_tray(self) -> None:
        self.tray_icon = QSystemTrayIcon(QIcon(ICON_PATH), self.app)
        menu = QMenu()
        config_action = QAction("Configuraciones", self.app)
        config_action.triggered.connect(self._open_config_window)
        quit_action = QAction("Cerrar Asistente", self.app)
        quit_action.triggered.connect(self.app.quit)
        menu.addAction(config_action)
        menu.addAction(quit_action)
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()
        self.app.setWindowIcon(QIcon(ICON_PATH))

    @_slot_guard
    def _open_config_window(self):
        setup.run_setup_window(from_system_tray=True)

    def _connect_signals(self) -> None:
        # Frontend → main
        self.avatar.js_event.connect(self.on_js_event)
        self.avatar.system_ready.connect(self.on_system_ready)

        # Input → main / audio
        self.input_manager.recording_started.connect(self.on_recording_started)
        self.input_manager.recording_canceled.connect(self.on_recording_canceled)
        self.input_manager.audio_ready.connect(self.on_valid_audio_input)
        self.input_manager.audio_live_ready.connect(self.audio_core.process_live_input)
        self.input_manager.volume_level.connect(self.on_volume_level)
        # ERR_MIC y futuros errores de input comparten los fallbacks hablados.
        self.input_manager.error_occurred.connect(self.on_error)

        # Audio → main
        self.audio_core.live_text_ready.connect(self.on_live_text)
        self.audio_core.text_transcribed.connect(self.on_text_transcribed)
        self.audio_core.model_missing.connect(self.on_model_missing)

        # Cerebro → main
        self.api_brain.text_chunk_ready.connect(self.on_llm_text_ready)
        self.api_brain.point_action_ready.connect(self.on_point_action)
        self.api_brain.error_occurred.connect(self.on_error)

        # TTS → main / avatar
        self.tts_core.speech_started.connect(self.on_speech_started)
        self.tts_core.speech_ended.connect(self.on_speech_ended)
        # text_to_speak emite (texto, duracion, timeline_visemas).
        self.tts_core.text_to_speak.connect(self.avatar.on_text_to_speak)

        # Descarga de modelos → main
        self._download_signals.finished.connect(self.on_models_downloaded)

    def _check_tts_models(self) -> None:
        """Si faltan los modelos Kokoro, dispara la misma descarga que whisper."""
        kokoro_ok = (os.path.exists(self.tts_core.model_path)
                     or os.path.exists(self.tts_core.model_quant_path))
        if not kokoro_ok or not os.path.exists(self.tts_core.voices_path):
            self.on_model_missing(self.tts_core.voices_path)

    # --------------------------------------------------------------- helpers

    def _set_avatar_state(self, state: str) -> None:
        self.avatar.set_avatar_state(state)

    def _voz_activa(self) -> str:
        return self.config.get("active_voice", DEFAULT_VOICE) or DEFAULT_VOICE

    def _fallback_message(self, error_code: str) -> str:
        fallbacks = self.frases_data.get("fallbacks", {})
        fallback_data = fallbacks.get(
            error_code, ["Algo reventó en el backend y no sé qué es. Revisá la consola."])
        if isinstance(fallback_data, list) and fallback_data:
            return random.choice(fallback_data)
        if isinstance(fallback_data, str):
            return fallback_data
        return "Error desconocido."

    def _schedule_hide_timer(self) -> None:
        logger.info("⏱️ [MAIN] Ocultamiento programado en %d ms", HIDE_DELAY_MS)
        self.hide_timer.start(HIDE_DELAY_MS)

    @_slot_guard
    def _hide_ui_and_avatar(self):
        logger.info("🔽 [MAIN] Ocultando caja de chat y avatar")
        self.chat_visible = False
        self.avatar.hide_chat()
        self._set_avatar_state("hidden")

    # ------------------------------------------------------------------ slots

    @_slot_guard
    def on_system_ready(self):
        self.input_manager.set_locked(False)

    @_slot_guard
    def on_js_event(self, msg: str):
        if msg == "TTS_ENDED":
            logger.info("🔓 [MAIN] TTS terminado (JS), desbloqueando input")
            self.input_manager.set_locked(False)

    @_slot_guard
    def on_recording_started(self):
        logger.info("🎙️ [MAIN] Grabación iniciada")
        self.sm.transition(State.RECORDING)
        self.input_manager.set_locked(True)

        # Estado del turno nuevo.
        self.last_transcribed_text = None
        self.is_immediate_phrase = False
        self.chat_visible = False
        self._last_audio = None
        self.hide_timer.stop()

        # Captura de pantalla en el momento exacto de la grabación.
        self.api_brain.capture_and_store_screen()

        # Volver de modo chat a modo transcripción.
        self.avatar.reset_transcription_ui()
        self.avatar.toggle_recording_ui(True)
        self.avatar.hide_ready_notification()

    @_slot_guard
    def on_recording_canceled(self):
        logger.info("❌ [MAIN] Grabación cancelada")
        self.audio_core.stop_live_transcription()
        self.sm.reset()

        if self.chat_visible:
            # Con respuesta a la vista, solo reiniciar el ocultamiento.
            self._schedule_hide_timer()
        else:
            self._hide_ui_and_avatar()
        self.input_manager.set_locked(False)

    @_slot_guard
    def on_valid_audio_input(self, audio_array):
        logger.info("🎯 [MAIN] Audio válido: %d samples", len(audio_array))
        self._last_audio = audio_array  # por si hay que reintentar tras descargar el modelo
        self.avatar.toggle_recording_ui(False)  # estado "Procesando"
        self.sm.transition(State.TRANSCRIBING)
        # Bloqueante: al volver, el streaming terminó (adiós QTimer.singleShot(100)).
        self.audio_core.stop_live_transcription()
        self.audio_core.process_voice_input(audio_array)

    @_slot_guard
    def on_live_text(self, text: str):
        self.avatar.update_live_transcription(text)

    @_slot_guard
    def on_volume_level(self, level: float):
        self.avatar.update_volume_meter(level)

    @_slot_guard
    def on_text_transcribed(self, text: str):
        text = (text or "").strip()
        if not text:
            if self._download_in_progress:
                # Sin modelo no hay transcripción: se reintenta al terminar la
                # descarga (ver on_models_downloaded).
                logger.info("⏳ [MAIN] Transcripción aplazada: descarga de modelo en curso")
                return
            logger.warning("⚠️ [MAIN] Transcripción vacía, cancelando turno")
            self.sm.reset()
            self._hide_ui_and_avatar()
            self.input_manager.set_locked(False)
            return

        logger.info("✅ [MAIN] Texto transcribido: '%s'", text)
        self.last_transcribed_text = text
        self._last_audio = None
        self.avatar.update_transcription(text)
        self.sm.transition(State.THINKING)
        self._set_avatar_state("thinking")

        # Frase inmediata solo si hay API key (sin ella vendrá ERR_NO_API).
        api_key = str(self.config.get("api_key", "") or "").strip()
        pool = self.frases_data.get("inmediatos", []) + self.frases_data.get("largos", [])
        if api_key and pool:
            frase = random.choice(pool)
            logger.info("🎵 [MAIN] Frase inmediata: '%s'", frase)
            self.is_immediate_phrase = True
            self.tts_core.process_text_async(frase, self._voz_activa())

        self.api_brain.submit_query(text)

    @_slot_guard
    def on_error(self, error_code: str):
        logger.error("❌ [MAIN] Error recibido: %s", error_code)
        msg = self._fallback_message(error_code)

        self.avatar.flash_error()

        # Bug corregido: aunque la transcripción no produjera texto
        # (last_transcribed_text es None), el error también se muestra en la UI.
        user_text = self.last_transcribed_text or "(sin transcripción de audio)"
        self.chat_visible = True
        self.avatar.transition_to_chat_mode(self.username, user_text)
        self.avatar.show_assistant_response(msg, is_fallback=True)

        # ERROR es transitorio; desde IDLE (p.ej. ERR_MIC sin grabación) no hay
        # transición y el estado simplemente se queda en IDLE.
        if self.sm.can_transition(State.ERROR):
            self.sm.transition(State.ERROR)

        self.tts_core.process_text_async(msg, self._voz_activa())

    @_slot_guard
    def on_llm_text_ready(self, text: str):
        # A partir de aquí lo que suena es la respuesta real, no frase inmediata.
        self.is_immediate_phrase = False

        user_text = self.last_transcribed_text or "(sin transcripción de audio)"
        self.chat_visible = True
        self.avatar.transition_to_chat_mode(self.username, user_text)
        self.avatar.show_assistant_response(text)

        self.tts_core.process_text_async(text, self._voz_activa())

    @_slot_guard
    def on_point_action(self, x: int, y: int):
        self.avatar.point_at(x, y)

    @_slot_guard
    def on_speech_started(self):
        if self.sm.can_transition(State.SPEAKING):
            self.sm.transition(State.SPEAKING)
        self._set_avatar_state("speaking")

    @_slot_guard
    def on_speech_ended(self):
        self._set_avatar_state("idle")

        if self.is_immediate_phrase:
            # Era la frase de cortesía: seguimos esperando la respuesta real.
            logger.info("⏭️ [MAIN] Frase inmediata terminada, esperando al modelo")
            self.is_immediate_phrase = False
            return

        self.input_manager.set_locked(False)
        if self.sm.state in (State.SPEAKING, State.ERROR):
            self.sm.transition(State.IDLE)
        self._schedule_hide_timer()

    # -------------------------------------------------------- descarga modelos

    @_slot_guard
    def on_model_missing(self, path: str):
        if self._download_in_progress:
            logger.info("⏳ [MAIN] Descarga de modelos ya en curso (faltaba %s)", path)
            return
        self._download_in_progress = True
        logger.warning("⚠️ [MAIN] Modelo ausente (%s); descargando en segundo plano", path)
        # Aviso hablado: la descarga puede tardar varios minutos.
        self.tts_core.process_text_async(
            "Me falta el modelo de voz. Lo estoy descargando, dame unos minutos.",
            self._voz_activa())
        worker = _ModelDownloadWorker(self.config.as_dict(), self._download_signals)
        QThreadPool.globalInstance().start(worker)

    @_slot_guard
    def on_models_downloaded(self, downloaded: list):
        self._download_in_progress = False
        logger.info("✅ [MAIN] ensure_models terminó; descargados: %s", downloaded)

        # Refrescar disponibilidad del modelo whisper (se evaluó en __init__).
        whisper = self.audio_core.whisper
        whisper.model_available = os.path.exists(whisper.model_path)

        if not whisper.model_available:
            logger.error("❌ [MAIN] La descarga no restauró el modelo whisper")
            if self.sm.state is State.TRANSCRIBING:
                self.sm.reset()
                self._hide_ui_and_avatar()
                self.input_manager.set_locked(False)
            self.on_error("ERR_NETWORK")
            return

        self.tts_core.process_text_async(
            "Listo, modelo descargado. Ya puedes hablarme.", self._voz_activa())

        # Reintento: si el turno quedó esperando la transcripción, relanzarla.
        if self._last_audio is not None and self.sm.state is State.TRANSCRIBING:
            logger.info("🔁 [MAIN] Reintentando transcripción tras la descarga")
            audio, self._last_audio = self._last_audio, None
            self.audio_core.process_voice_input(audio)


def run_app():
    setup_logging()  # idempotente: launch.py ya pudo llamarla antes

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Sin referencia Python ni parent Qt, el GC recolecta AssistantApp en
    # cualquier momento y Qt desconecta todos sus slots (hotkey y tray mueren).
    assistant = AssistantApp(app)
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
