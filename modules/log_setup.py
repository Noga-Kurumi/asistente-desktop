"""Configuración centralizada de logging para todo el asistente.

Llamar a setup_logging() una sola vez al inicio (en launch/main).
Todos los módulos deben usar logging.getLogger(__name__) y NO basicConfig.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "asistente.log")

_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_initialized = False


def setup_logging():
    """Inicializa el logging root: consola + archivo rotativo. Idempotente."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1024 * 1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
