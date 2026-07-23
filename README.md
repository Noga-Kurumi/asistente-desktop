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

## Timeline de contexto

El asistente registra un **timeline local de tu actividad** (SQLite + FTS5 en
`data/timeline.db`, solo texto, con retención configurable) mediante
recolectores pasivos en `modules/collectors/`:

- **Ventana activa** — cada cambio de foco (app + título).
- **Portapapeles** — texto copiado (deduplicado).
- **OCR de pantalla** — texto visible, solo cuando la pantalla cambia (OCR
  nativo de Windows, paquete `winsdk`).
- **Audio de reuniones** — cuando Discord está en canal de voz (sesión de
  audio activa vía `pycaw`), transcribe tu mic (`audio_in`) y el audio de la
  llamada por WASAPI Loopback (`audio_out`) con PyAudioWPatch + whisper.cpp.
  El audio se descarta: solo queda el texto.

El LLM accede al timeline mediante **2 tools MCP**
(`search_timeline_by_keywords` y `get_timeline_by_time_range`, servidor en
`modules/timeline_mcp_server.py`, también usable standalone por stdio), con
function calling de Gemini. Podés preguntarle "¿qué estuve haciendo esta
mañana?" o "¿qué copié recién?".

Config en `config.json` (sin UI por ahora): `timeline_enabled`,
`timeline_retention_hours`, `ocr_interval_seconds`, `ocr_max_chars`,
`meeting_*` y `mcp_timeline_enabled`. Dependencias nuevas: `pywin32`,
`psutil`, `winsdk`, `PyAudioWPatch`, `pycaw`, `mcp`.

## Empaquetado (ejecutable Windows)

Se usa PyInstaller en modo **onedir** (carpeta portable, arranque rápido con
QtWebEngine). Los modelos grandes NO van dentro: se descargan junto al exe la
primera vez, igual que en el repo.

```
pyinstaller --noconfirm --clean asistente.spec
```

Genera `dist/asistente/` con `asistente.exe` (windowed, sin consola; los logs
quedan en `logs/asistente.log` junto al exe). Toda la carpeta es portable:
config, logs, timeline y modelos viven al lado del exe. Para depurar el build,
cambiar `console=False` a `True` en `asistente.spec`.

## Créditos

El avatar incluido se redistribuye con acreditación; ver `NOTICE`.
