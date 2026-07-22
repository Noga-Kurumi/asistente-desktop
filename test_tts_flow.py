"""Verificacion offscreen del flujo TTS/UI (anti-parpadeo + chat incremental
+ streaming Kokoro).

App real: AssistantApp + AssistantTTS real (sd y Kokoro fakeados) +
AssistantBrain real con provider fake de 3 oraciones. Avatar y audio_core
stubeados. Sin audio real, sin red.
"""
import asyncio
import os
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

# ---- hardware falso ANTES de importar los modulos ----
import sounddevice as sd


class FakeStream:
    def __init__(self, *a, **kw): pass
    def start(self): pass
    def stop(self): pass
    def close(self): pass


play_calls = []
sd.InputStream = FakeStream
sd.play = lambda samples, sr: play_calls.append(len(samples))
sd.wait = lambda: None
sd.stop = lambda: None

from pynput import keyboard


class FakeListener:
    def __init__(self, on_press=None, on_release=None, **kw): pass
    def start(self): pass
    def stop(self): pass


keyboard.Listener = FakeListener

app = QApplication([])

import modules.tts_core as tts_core


class FakeKokoro:
    """Kokoro de mentira: create_stream entrega 2 lotes por texto (streaming)."""

    def __init__(self, *a, **kw):
        pass

    async def create_stream(self, text, voice=None, speed=1.0, lang="es"):
        for _ in range(2):
            yield (np.full(2400, 0.05, dtype=np.float32)), 24000
            await asyncio.sleep(0)


tts_core.Kokoro = FakeKokoro

import main
from modules.state_machine import State


class StubAudioCore(QObject):
    live_text_ready = Signal(str)
    text_transcribed = Signal(str)
    model_missing = Signal(str)
    whisper = type("W", (), {"model_available": True, "model_path": "models/ggml-tiny.bin"})()

    def notify_model_missing(self): pass
    def stop_live_transcription(self): pass
    def process_live_input(self, audio): pass
    def process_voice_input(self, audio): pass


class FakeProvider:
    """LLM de mentira: respuesta de 3 oraciones."""
    name = "fake"

    def stream_reply(self, messages, system_prompt, image_bytes, on_sentence):
        for s in ("Uno.", "Dos.", "Tres."):
            on_sentence(s)
        return "Uno. Dos. Tres."


class StubAvatar(QObject):
    js_event = Signal(str)
    system_ready = Signal()

    def __init__(self):
        super().__init__()
        self.chat_transitions = 0
        self.updates = []      # update_assistant_response (chat incremental)
        self.shows = []        # show_assistant_response (final/fallback)
        self.states = []       # set_avatar_state

    def show(self): pass

    def transition_to_chat_mode(self, u, t):
        self.chat_transitions += 1

    def update_assistant_response(self, text):
        self.updates.append(text)

    def show_assistant_response(self, text, is_fallback=False):
        self.shows.append((text, is_fallback))

    def set_avatar_state(self, state):
        self.states.append(state)

    def on_text_to_speak(self, text, dur, timeline): pass

    def __getattr__(self, name):
        return lambda *a, **kw: None


main.AssistantAudioCore = StubAudioCore
main.AvatarWindow = lambda: StubAvatar()

import modules.api_brain as api_brain_module

main.AssistantBrain = lambda: api_brain_module.AssistantBrain(provider=FakeProvider())

assistant = main.AssistantApp(app)
avatar = assistant.avatar
tts = assistant.tts_core
sm = assistant.sm
assert tts.is_ready or tts._ready_event.wait(5), "TTS fake no quedo listo"

# Sin frase inmediata para acotar el escenario
assistant.frases_data = {"inmediatos": [], "largos": [], "fallbacks": {}}

# Spies
estados = []
sm.state_changed.connect(lambda o, n: estados.append((o.value, n.value)))

hide_calls = []
_orig_hide = assistant._schedule_hide_timer
assistant._schedule_hide_timer = lambda: (hide_calls.append(1), _orig_hide())

locks = []
_orig_lock = assistant.input_manager.set_locked
assistant.input_manager.set_locked = lambda v: (locks.append(v), _orig_lock(v))

speech_starts, speech_ends = [], []
tts.speech_started.connect(lambda: speech_starts.append(1))
tts.speech_ended.connect(lambda: speech_ends.append(1))
tts_emissions = []
tts.text_to_speak.connect(lambda t, d, tl: tts_emissions.append((t, d)))

# Desbloqueo inicial y turno de 3 oraciones
avatar.system_ready.emit()
app.processEvents()
assert locks == [False], locks

# Secuencia real: hotkey down (RECORDING) -> audio listo (TRANSCRIBING)
assistant.on_recording_started()
locks.clear()
assistant.sm.transition(State.TRANSCRIBING)
assistant.on_text_transcribed("hola")
assert sm.state is State.THINKING
assert assistant._response_pending is True

t0 = time.time()
while time.time() - t0 < 15:
    app.processEvents()
    if sm.state is State.IDLE and tts.pending_count == 0 and hide_calls:
        break
    time.sleep(0.05)

print("estados:", estados)
print("updates chat:", avatar.updates)
print("speech starts/ends:", len(speech_starts), len(speech_ends))
print("tts emissions:", len(tts_emissions), "| sd.play:", len(play_calls))
print("hide timer programado:", len(hide_calls), "vez/ces | locks:", locks)

# (a) SPEAKING se mantiene entre oraciones; IDLE solo al final
trans = [(o, n) for o, n in estados]
assert ("thinking", "speaking") in trans
assert trans.count(("speaking", "idle")) == 1, f"IDLE prematuro: {trans}"
assert trans[-1] == ("speaking", "idle"), trans
assert not any(n == "idle" for o, n in trans[:-1]), f"paso por idle a mitad: {trans}"

# (b) ocultamiento programado UNA sola vez; input desbloqueado solo al final
# (locks se limpio tras el lock de la grabacion: solo debe quedar el unlock final)
assert len(hide_calls) == 1, hide_calls
assert locks == [False], f"desbloqueos extra: {locks}"

# 3 oraciones x 2 lotes (streaming) = 6 text_to_speak y 6 sd.play
assert len(speech_starts) == 3 and len(speech_ends) == 3
assert len(tts_emissions) == 6, tts_emissions
assert len(play_calls) == 6

# (c) chat incremental con texto acumulado en orden
assert avatar.chat_transitions == 1, avatar.chat_transitions
assert avatar.updates == ["Uno.", "Uno. Dos.", "Uno. Dos. Tres."], avatar.updates
assert avatar.shows == [], "no debe haber show final por oracion"
assert "speaking" in avatar.states and avatar.states[-1] == "idle"

assert tts.pending_count == 0
print("\nTODO OK")
