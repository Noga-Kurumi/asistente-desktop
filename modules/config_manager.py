"""Gestor único de configuración del asistente.

Única fuente de verdad de configuración. Todos los módulos leen/escriben la
configuración a través de get_config(), evitando lecturas de JSON dispersas.

Carga en cascada (cada capa sobrescribe a la anterior):
    DEFAULTS  ←  config.json  ←  config.local.json

config.local.json está gitignored y es donde viven los secretos (api_key).
save() reparte automáticamente: las claves sensibles van a config.local.json
y el resto a config.json, siempre con escritura atómica (.tmp + os.replace).
"""

import os
import json
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
LOCAL_CONFIG_FILE = os.path.join(BASE_DIR, "config.local.json")

# Claves que NUNCA se escriben en config.json (van a config.local.json).
SENSITIVE_KEYS = frozenset({"api_key"})

# Campos que deben tener valor para que el asistente funcione.
REQUIRED_KEYS = ("api_key",)

DEFAULTS: Dict[str, Any] = {
    "username": "",
    "api_key": "",
    "api_provider": "gemini",
    "avatar_name": "Lindsay",
    "active_avatar": "Lindsay.vrm",
    "active_voice": "ef_dora",
    "launch_with_system": False,
    "hotkey": "Key.alt_r",
    "custom_instructions": "",
    "audio_device": None,
    "whisper_model": "tiny",
    "whisper_quantization": "q5_1",
    "whisper_threads": 4,
    "gemini_model": "gemini-2.0-flash",
}


def _read_json(path: str) -> Dict[str, Any]:
    """Lee un JSON de configuración; devuelve {} si no existe o está corrupto."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("⚠️ [CONFIG] %s no contiene un objeto JSON, se ignora", path)
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error("❌ [CONFIG] Error leyendo %s: %s. Se usan defaults para esas claves.", path, e)
        return {}


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Escritura atómica: escribe a .tmp y renombra con os.replace."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


class ConfigManager:
    """Singleton thread-safe de configuración con carga en cascada."""

    _instance: Optional["ConfigManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self, config_file: str = CONFIG_FILE, local_file: str = LOCAL_CONFIG_FILE):
        self._config_file = config_file
        self._local_file = local_file
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = dict(DEFAULTS)
        self.reload()

    @classmethod
    def instance(cls) -> "ConfigManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def reload(self) -> None:
        """Recarga la cascada DEFAULTS ← config.json ← config.local.json."""
        with self._lock:
            data = dict(DEFAULTS)
            data.update(_read_json(self._config_file))
            data.update(_read_json(self._local_file))
            self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def update(self, values: Dict[str, Any]) -> None:
        """Actualiza claves en memoria. Persistir llamando a save()."""
        if not isinstance(values, dict):
            raise TypeError("update() espera un dict de clave → valor")
        with self._lock:
            self._data.update(values)

    def save(self) -> None:
        """Persiste la configuración de forma atómica y thread-safe.

        Las claves sensibles (SENSITIVE_KEYS) se escriben solo en
        config.local.json; el resto en config.json.
        """
        with self._lock:
            public = {k: v for k, v in self._data.items() if k not in SENSITIVE_KEYS}
            private = {k: v for k, v in self._data.items() if k in SENSITIVE_KEYS}
            try:
                _write_json_atomic(self._config_file, public)
                _write_json_atomic(self._local_file, private)
            except OSError as e:
                logger.error("❌ [CONFIG] Error guardando configuración: %s", e, exc_info=True)
                raise

    def validate(self) -> List[str]:
        """Devuelve la lista de campos obligatorios sin valor (vacía si todo OK)."""
        with self._lock:
            return [k for k in REQUIRED_KEYS if not str(self._data.get(k) or "").strip()]


def get_config() -> ConfigManager:
    """Atajo al singleton de configuración."""
    return ConfigManager.instance()
