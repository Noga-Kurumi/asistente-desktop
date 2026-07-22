"""Gestor único de configuración del asistente.

Singleton: todos los módulos leen/escriben config.json a través de get_config(),
evitando las N copias de lógica de lectura dispersas por el proyecto.
"""

import os
import json
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

DEFAULTS = {
    "username": "Usuario",
    "api_key": "",
    "api_provider": "",
    "avatar_name": "Lindsay",
    "active_avatar": "",
    "active_voice": "",
    "launch_with_system": False,
    "hotkey": "Key.alt_r",
    "custom_instructions": "",
    "audio_device": None,
    "whisper_model": "tiny",
    "whisper_quantization": "none",
    "gemini_model": "",
    "anthropic_model": "claude-3-5-haiku-20241022",
}


class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._data = dict(DEFAULTS)
        self.reload()

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def reload(self):
        """Recarga config.json desde disco, manteniendo defaults para claves faltantes."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._data.update(data)
            except Exception:
                pass

    def get(self, key, default=None):
        return self._data.get(key, default)

    def as_dict(self):
        return dict(self._data)

    def update(self, **kwargs):
        self._data.update(kwargs)

    def save(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=4)


def get_config():
    """Atajo al singleton."""
    return ConfigManager.instance()
