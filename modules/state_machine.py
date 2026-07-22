"""Máquina de estados explícita del asistente.

Estados del pipeline de voz:
    IDLE → RECORDING → TRANSCRIBING → THINKING → SPEAKING → IDLE
ERROR es transitorio: se entra desde cualquier estado y se vuelve a IDLE.

Thread-safe: transition() puede llamarse desde cualquier hilo; la señal
state_changed (queued en Qt) llega siempre al hilo de la UI.

Integración con main.py (lo aplica el agente que reescriba main.py):
    - Crear UNA instancia y conectar state_changed a la UI del avatar:
          sm = AssistantStateMachine()
          sm.state_changed.connect(lambda old, new: ...)
    - Reemplazar los flags nonlocal (is_chat_mode, is_immediate_phrase, ...)
      por transiciones explícitas en los handlers de señales:
          recording_started   → sm.transition(State.RECORDING)
          audio_ready         → sm.transition(State.TRANSCRIBING)
          text_transcribed OK → sm.transition(State.THINKING)
          speech_started      → sm.transition(State.SPEAKING)
          speech_ended        → sm.transition(State.IDLE)
          error_occurred      → sm.transition(State.ERROR) y luego IDLE
    - Las transiciones inválidas se loguean y se ignoran (devuelve False);
      si el flujo necesita una transición nueva, añadirla a TRANSITIONS.
"""

import enum
import logging
import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


# Transiciones válidas: origen → destinos permitidos.
TRANSITIONS = {
    State.IDLE: {State.RECORDING},
    State.RECORDING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.THINKING, State.IDLE, State.ERROR},
    State.THINKING: {State.SPEAKING, State.IDLE, State.ERROR},
    State.SPEAKING: {State.IDLE, State.SPEAKING, State.ERROR},  # SPEAKING→SPEAKING: frases encadenadas
    State.ERROR: {State.IDLE},
}


class AssistantStateMachine(QObject):
    """Máquina de estados del asistente. Thread-safe, con señal Qt."""

    state_changed = Signal(object, object)  # (State anterior, State nuevo)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._state = State.IDLE

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def can_transition(self, new_state: State) -> bool:
        """True si la transición desde el estado actual es válida."""
        with self._lock:
            return new_state in TRANSITIONS.get(self._state, set())

    def transition(self, new_state: State) -> bool:
        """Intenta transicionar. Devuelve True si la transición fue válida.

        Las transiciones inválidas se loguean como warning y se ignoran: la
        máquina nunca queda en un estado inconsistente.
        """
        with self._lock:
            old = self._state
            if new_state not in TRANSITIONS.get(old, set()):
                logger.warning(
                    "⚠️ [STATE] Transición inválida ignorada: %s → %s",
                    old.value, new_state.value,
                )
                return False
            self._state = new_state
        logger.info("🔄 [STATE] %s → %s", old.value, new_state.value)
        self.state_changed.emit(old, new_state)
        return True

    def reset(self) -> None:
        """Fuerza el retorno a IDLE (uso excepcional: cancelaciones, cierre)."""
        with self._lock:
            old = self._state
            if old is State.IDLE:
                return
            self._state = State.IDLE
        logger.info("🔄 [STATE] %s → idle (reset forzado)", old.value)
        self.state_changed.emit(old, State.IDLE)
