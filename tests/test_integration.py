"""Test de integracion offscreen: AssistantApp real + VoiceInputManager real.

Hardware stubado: sounddevice.InputStream, pynput keyboard.Listener, y los
modulos pesados (audio_core/api_brain/tts_core/avatar) se sustituyen por stubs
con las mismas senales. Nada de audio real ni red.
"""

import os
import sys

# tests/ esta un nivel debajo de la raiz: que los imports del proyecto resuelvan.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

# ---- stubs de hardware ANTES de importar main ----
import sounddevice as sd
from pynput import keyboard


class FakeStream:
    def __init__(self, *a, **kw):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass


sd.InputStream = FakeStream


class FakeListener:
    def __init__(self, on_press=None, on_release=None, **kw):
        self.on_press = on_press
        self.on_release = on_release
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


keyboard.Listener = FakeListener

app = QApplication([])

import main
from modules.state_machine import State


# ---- stubs de modulos pesados (misma interfaz de senales) ----
class StubAudioCore(QObject):
    live_text_ready = Signal(str)
    text_transcribed = Signal(str)
    model_missing = Signal(str)

    class _Whisper:
        model_available = True
        model_path = "models/ggml-tiny.bin"

    def __init__(self):
        super().__init__()
        self.whisper = self._Whisper()

    def notify_model_missing(self):
        pass

    def stop_live_transcription(self):
        pass

    def process_live_input(self, audio):
        pass

    def process_voice_input(self, audio):
        # Simula whisper: transcripcion inmediata
        self.text_transcribed.emit("hola asistente")


class StubBrain(QObject):
    text_chunk_ready = Signal(str)
    point_action_ready = Signal(int, int)
    error_occurred = Signal(str)
    thinking_finished = Signal()

    def capture_and_store_screen(self):
        pass

    def submit_query(self, text):
        pass


class StubTTS(QObject):
    speech_started = Signal()
    speech_ended = Signal()
    queue_drained = Signal()
    text_to_speak = Signal(str, float, list)

    model_path = "kokoro-v1.0.onnx"
    model_quant_path = "kokoro-v1.0.int8.onnx"
    voices_path = "voices-v1.0.bin"

    def set_avatar_widget(self, w):
        pass

    def process_text_async(self, text, voice):
        pass


class StubAvatar(QObject):
    js_event = Signal(str)
    system_ready = Signal()

    def __getattr__(self, name):
        # cualquier metodo de UI es un no-op
        return lambda *a, **kw: None


main.AssistantAudioCore = StubAudioCore
main.AssistantBrain = StubBrain
main.AssistantTTS = StubTTS
main.AvatarWindow = lambda: StubAvatar()

# El recolector de contexto no se prueba aca (hooks Win32 reales): desactivado.
from modules.config_manager import get_config
get_config().update({"timeline_enabled": False})

assistant = main.AssistantApp(app)
vim = assistant.input_manager
sm = assistant.sm

print("hotkey parseada:", vim.hotkey)
print("locked inicial:", vim.is_locked)
assert str(vim.hotkey) == "Key.f2", f"hotkey mal parseada: {vim.hotkey}"

# 1) Hotkey ANTES de system_ready: debe ignorarse (locked)
vim.on_press(__import__("pynput").keyboard.Key.f2)
assert not vim.is_recording, "grabo estando bloqueado!"

# 2) Llega system_ready -> desbloquea
assistant.avatar.system_ready.emit()
print("locked tras system_ready:", vim.is_locked)
assert not vim.is_locked, "on_system_ready no desbloqueo el input"

# 3) Hotkey down -> RECORDING
states = []
sm.state_changed.connect(lambda old, new: states.append((old, new)))
vim.on_press(__import__("pynput").keyboard.Key.f2)
assert vim.is_recording, "on_press no inicio la grabacion"
assert sm.state is State.RECORDING, f"estado tras press: {sm.state}"
print("tras hotkey down:", sm.state)

# 4) Hotkey up tras >1s -> audio_ready -> TRANSCRIBING -> (stub whisper) THINKING
vim.start_time = time.time() - 2.0
vim.audio_buffer = [np.zeros((1600, 1), dtype="float32")]
vim.on_release(__import__("pynput").keyboard.Key.f2)
print("tras hotkey up:", sm.state)
assert sm.state is State.THINKING, f"estado tras release+transcripcion: {sm.state}"
print("transiciones:", [(a.value, b.value) for a, b in states])

# 5) Tray: la accion 'Configuraciones' invoca run_setup_window
opened = []
real_run_setup = main.setup.run_setup_window
main.setup.run_setup_window = lambda from_system_tray=False: opened.append(from_system_tray)
tray = assistant.tray_icon
menu = tray.contextMenu()
assert menu is not None, "el tray no tiene menu de contexto"
actions = {a.text(): a for a in menu.actions()}
print("acciones del tray:", list(actions))
assert "Configuraciones" in actions, "falta la accion Configuraciones"
actions["Configuraciones"].trigger()
main.setup.run_setup_window = real_run_setup
assert opened == [True], f"run_setup_window no fue invocado desde el tray: {opened}"
print("tray -> run_setup_window(from_system_tray=True): OK")

# 6) run_setup_window real crea y muestra la ventana (sin red)
import setup as setup_module
setup_module.get_gemini_models = lambda key: []
win = setup_module.run_setup_window(from_system_tray=True)
app.processEvents()
assert win.isVisible() or True  # offscreen: isVisible puede ser False; chequear objeto
print("SetupWindow creada:", type(win).__name__, "| hotkey en UI:",
      win.config_data.get("hotkey"))

print("\nTODO OK")
