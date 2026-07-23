"""Test vital (barato): lógica pura del recolector de reuniones.

Conversión int16→float32, downmix estéreo→mono, RMS gate, remuestreo lineal y
_flush_segment (descarte por silencio / encolado / flush corto / cola llena).
SIN audio real ni whisper: la captura y la detección en Discord son prueba
manual del usuario.
"""
import queue
from types import SimpleNamespace

import numpy as np

from modules.collectors.meeting_audio import (
    MeetingAudioCollector,
    int16_bytes_to_float32,
    resample_linear,
    rms,
    SAMPLE_RATE,
)

# ---- int16_bytes_to_float32 ----
pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()
audio = int16_bytes_to_float32(pcm)
assert audio.dtype == np.float32
assert abs(audio[0]) < 1e-6 and abs(audio[1] - 0.5) < 1e-3
assert abs(audio[3] - 1.0) < 1e-3 and abs(audio[4] + 1.0) < 1e-3
# estéreo: L=1.0, R=-1.0 -> mono 0.0
stereo = np.array([32767, -32768] * 100, dtype=np.int16).tobytes()
mono = int16_bytes_to_float32(stereo, channels=2)
assert len(mono) == 100 and np.abs(mono).max() < 1e-3
print("int16->float32 + downmix estéreo OK")

# ---- rms ----
assert rms(np.array([], dtype=np.float32)) == 0.0
assert rms(np.zeros(1000, dtype=np.float32)) == 0.0
assert abs(rms(np.full(1000, 0.5, dtype=np.float32)) - 0.5) < 1e-6
print("rms OK")

# ---- resample_linear ----
x = np.zeros(4800, dtype=np.float32)
assert resample_linear(x, SAMPLE_RATE) is x  # mismo rate: pasa tal cual
y = resample_linear(np.sin(np.linspace(0, 10, 48000)).astype(np.float32), 48000)
assert len(y) == 16000, len(y)
assert abs(float(y.max()) - float(np.sin(np.linspace(0, 10, 48000)).max())) < 0.05
print("resample_linear OK (48k->16k, forma y longitud)")

# ---- _flush_segment (RMS gate y encolado) ----
cfg = SimpleNamespace(get=lambda k, d=None: {
    "meeting_poll_seconds": 5, "meeting_segment_seconds": 8,
    "meeting_rms_threshold": 0.01, "meeting_source_apps": ["discord"],
}.get(k, d))
collector = MeetingAudioCollector(db=None, config=cfg)

# silencio -> descartado (no entra a la cola de transcripción)
collector._flush_segment("audio_in", [np.zeros(SAMPLE_RATE, dtype=np.float32)])
assert collector._segments.empty(), "el silencio NO debe llegar a transcribir"
# señal fuerte -> encolada
fuerte = [np.full(SAMPLE_RATE, 0.2, dtype=np.float32)]
collector._flush_segment("audio_out", fuerte)
assert collector._segments.qsize() == 1
source, audio, seconds = collector._segments.get_nowait()
assert source == "audio_out" and abs(seconds - 1.0) < 1e-6
# flush final corto -> descartado
collector._flush_segment("audio_in", [np.full(SAMPLE_RATE // 2, 0.5, dtype=np.float32)],
                         final=True)
assert collector._segments.empty()
# cola llena -> descarta sin reventar
for _ in range(8):
    collector._segments.put_nowait(("audio_in", np.zeros(8, dtype=np.float32), 1.0))
collector._flush_segment("audio_in", [np.full(SAMPLE_RATE, 0.5, dtype=np.float32)])
assert collector._segments.qsize() == 8
print("_flush_segment OK (RMS gate, encolado, flush corto, cola llena)")

# ---- detección en vivo (sanity barato): Discord corre pero SIN llamada ----
assert collector._detect_meeting() is False, \
    "Discord está inactivo (State 0): no debe detectar meeting"
print("detección en vivo OK (Discord sin llamada -> False)")

print("\nTODO OK")
