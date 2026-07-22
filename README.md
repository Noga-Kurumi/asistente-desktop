# asistente-desktop

Asistente de voz para Windows con avatar 3D (VRM): mantén pulsada la tecla de
grabación, habla, y el asistente transcribe tu voz (Whisper), consulta a Gemini
y te responde en voz alta (Kokoro TTS) con lip sync.

## Requisitos

- Windows 10/11
- Python 3.13
- Una API key de Google Gemini (se configura en el primer arranque)

## Arranque

```
python launch.py
```

El primer arranque crea el entorno virtual (`.venv`), instala las dependencias
y abre la ventana de configuración. Los modelos de voz (Whisper y Kokoro) se
descargan solos la primera vez que hacen falta.

## Estructura

- `launch.py` — lanzador: venv, dependencias, validación de config.
- `main.py` — orquestación del pipeline de voz (máquina de estados).
- `setup.py` — ventana de configuración (API key, voz, hotkey, modelos).
- `modules/` — audio, LLM, TTS, configuración, logging y descarga de modelos.
- `avatars/`, `web/` — avatar VRM y su visor.

## Créditos

El avatar incluido se redistribuye con acreditación; ver `NOTICE`.
