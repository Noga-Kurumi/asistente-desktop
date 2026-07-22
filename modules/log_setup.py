"""Configuración centralizada de logging para todo el asistente.

Llamar a setup_logging() UNA SOLA VEZ desde el entry point (launch.py/main.py)
antes de crear cualquier otro módulo. Es idempotente: llamadas posteriores se
ignoran. Todos los módulos deben usar logging.getLogger(__name__) y NUNCA
logging.basicConfig() (eso reconfiguraría el root logger a espaldas de este
módulo).
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Union

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "asistente.log")

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_initialized = False


def setup_logging(level: Union[int, str] = logging.INFO) -> None:
    """Inicializa el root logger: consola + archivo rotativo en logs/.

    Args:
        level: Nivel de logging (constante de logging o nombre, p.ej. "DEBUG").

    Idempotente: solo la primera llamada tiene efecto. Debe llamarse una única
    vez desde el entry point de la aplicación.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(LOG_DIR, exist_ok=True)

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
