"""Verificacion del pipeline productor-consumidor del TTS (anti-pausas).

Con Kokoro y sd fake (reproduccion controlada con eventos) demuestra:
1. OVERLAP: la sintesis de la oracion 2 OCURRE MIENTRAS suena la 1.
2. Orden de reproduccion, pending_count en puntos clave, sin drained intermedio.
3. queue_drained solo cuando no queda NADA por sonar.
4. clear_queue interrumpe lo que suena, descarta lo sintetizado pendiente de
   sonar y resetea el contador; speech_ended del texto interrumpido se emite.
"""

import os
import sys

# tests/ esta un nivel debajo de la raiz: que los imports del proyecto resuelvan.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import os
import threading
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import sounddevice as sd


class FakeSD:
    """sd fake: cada play queda bloqueado en wait() hasta que el test lo suelta."""

    def __init__(self):
        self.plays = []          # un Event por reproduccion
        self.current = None
        self.stops = 0

    def play(self, samples, sr):
        ev = threading.Event()
        self.plays.append(ev)
        self.current = ev

    def wait(self):
        if self.current is not None:
            self.current.wait(10)

    def stop(self):
        self.stops += 1
        if self.current is not None:
            self.current.set()


fake_sd = FakeSD()
sd.play = fake_sd.play
sd.wait = fake_sd.wait
sd.stop = fake_sd.stop

import modules.tts_core as tts_core

synth_started = {}


class FakeKokoro:
    def __init__(self, *a, **kw):
        pass

    async def create_stream(self, text, voice=None, speed=1.0, lang="es"):
        synth_started[text] = time.time()
        await asyncio.sleep(0.3)  # la sintesis tarda un tiempo no trivial
        yield np.full(2400, 0.05, dtype=np.float32), 24000


tts_core.Kokoro = FakeKokoro

app = QApplication([])
tts = tts_core.AssistantTTS()
assert tts._ready_event.wait(5), "motor fake no quedo listo"

started, ended, drained, spoken = [], [], [], []
tts.speech_started.connect(lambda: started.append(1), Qt.DirectConnection)
tts.speech_ended.connect(lambda: ended.append(1), Qt.DirectConnection)
tts.queue_drained.connect(lambda: drained.append(1), Qt.DirectConnection)
tts.text_to_speak.connect(lambda t, d, tl: spoken.append(t), Qt.DirectConnection)


def espera(cond, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


# ---- 1. OVERLAP productor-consumidor ----
tts.process_text_async("t1", "voz")
tts.process_text_async("t2", "voz")
assert tts.pending_count == 2, tts.pending_count

assert espera(lambda: len(fake_sd.plays) == 1), "t1 nunca empezo a sonar"
# t1 esta sonando (bloqueado en sd.wait). La sintesis de t2 debe ocurrir YA.
time.sleep(0.6)  # > retardo de sintesis
assert "t2" in synth_started, \
    "la sintesis de t2 NO solapo con la reproduccion de t1 (sigue en serie)"
print("1. overlap OK: sintesis de t2 empezo mientras sonaba t1")

# ---- 2. orden, pending_count, sin drained intermedio ----
fake_sd.plays[0].set()  # termina de sonar t1
assert espera(lambda: len(fake_sd.plays) == 2), "t2 nunca empezo a sonar"
assert ended == [1] and started == [1, 1]
assert not drained, f"drained prematuro: {drained}"
assert tts.pending_count == 1, tts.pending_count
assert spoken == ["t1", "t2"], spoken
print("2. orden t1->t2 OK, pending_count 2->1, sin drained intermedio")

# ---- 3. queue_drained solo al final ----
fake_sd.plays[1].set()
assert espera(lambda: len(drained) == 1), "queue_drained no llego"
assert tts.pending_count == 0
assert ended == [1, 1]
print("3. queue_drained solo al final; pending_count 0")

# ---- 4. clear_queue: interrumpe, descarta, resetea ----
started.clear(); ended.clear(); drained.clear(); spoken.clear()
tts.process_text_async("t3", "voz")
tts.process_text_async("t4", "voz")
assert espera(lambda: len(fake_sd.plays) == 3), "t3 nunca empezo a sonar"
time.sleep(0.6)  # t4 queda sintetizado en la cola de audio, esperando sonar
assert "t4" in synth_started
tts.clear_queue()
assert tts.pending_count == 0, tts.pending_count
assert espera(lambda: len(ended) == 1), "speech_ended del texto interrumpido no llego"
time.sleep(0.3)
assert len(fake_sd.plays) == 3, "t4 no debio sonar tras clear_queue"
assert fake_sd.stops >= 1, "clear_queue no llamo sd.stop()"
assert not drained, "clear_queue no debe emitir queue_drained"
print("4. clear_queue OK: t3 interrumpido (ended x1), t4 descartado, pending 0")

print("\nTODO OK")
