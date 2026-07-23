"""Recolectores pasivos de contexto (Fase B): window_change, clipboard, screen_ocr.

El coordinador (ContextCollector) los instancia sobre una TimelineDB
compartida. Todos son hilos daemon con start()/stop() y errores atrapados:
ningún recolector debe tumbar la app.
"""

from modules.collectors.coordinator import ContextCollector

__all__ = ["ContextCollector"]
