"""Verificacion offscreen de los dos seguimientos:
1. _parse_hotkey: "f2" invalido -> warning + fallback Key.alt_r (no KeyCode basura).
2. SetupWindow: get_gemini_models asincrono (ventana inmediata, combo se puebla
   cuando llega la respuesta, error logueado y combo con 'No se encontraron modelos').
"""

import os
import sys

# tests/ esta un nivel debajo de la raiz: que los imports del proyecto resuelvan.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

app = QApplication([])

# ---------- 1. _parse_hotkey ----------
import main
from pynput.keyboard import Key, KeyCode

parse = main.AssistantApp._parse_hotkey
assert parse(SimpleNamespace(config={"hotkey": "Key.f2"})) == Key.f2
assert parse(SimpleNamespace(config={"hotkey": "Key.alt_r"})) == Key.alt_r
assert parse(SimpleNamespace(config={"hotkey": "a"})) == KeyCode.from_char("a")
hk = parse(SimpleNamespace(config={"hotkey": "f2"}))
assert hk == Key.alt_r, f"'f2' debio caer al fallback, salio {hk}"
assert hk != Key.f2
print("1. _parse_hotkey: 'f2' -> fallback alt_r con warning; casos validos OK")

# ---------- 2. get_gemini_models asincrono ----------
import setup as setup_module

MODELOS_FAKE = [
    {"name": "gemini-2.0-flash", "display_name": "Gemini 2.0 Flash"},
    {"name": "gemini-3.5-flash", "display_name": "Gemini 3.5 Flash"},
]


def fake_models_lenta(api_key):
    time.sleep(1.0)  # si fuera sincrona, el constructor se trabaria 1s
    return list(MODELOS_FAKE)


setup_module.get_gemini_models = fake_models_lenta

t0 = time.time()
win = setup_module.SetupWindow(from_system_tray=True)
dt = time.time() - t0
assert dt < 1.0, f"SetupWindow tardo {dt:.2f}s: la red sigue siendo sincrona"
print(f"2. SetupWindow construida en {dt*1000:.0f} ms (no bloquea)")
assert win.combo_model.itemText(0) == "Cargando modelos...", win.combo_model.itemText(0)

# Esperar a que el worker entregue los modelos
for _ in range(50):
    app.processEvents()
    if win.combo_model.itemText(0) != "Cargando modelos...":
        break
    time.sleep(0.1)

textos = [win.combo_model.itemText(i) for i in range(win.combo_model.count())]
print("   combo tras carga:", textos, "| seleccionado:", win.combo_model.currentText())
assert win.combo_model.count() == 2
assert win.combo_model.itemData(0) == "gemini-2.0-flash"
# Si el gemini_model de la config esta en la lista debe quedar seleccionado;
# si no, cae al primero.
from modules.config_manager import get_config
_cfg_model = get_config().get("gemini_model", "")
_esperado = _cfg_model if _cfg_model in [m["name"] for m in MODELOS_FAKE] \
    else MODELOS_FAKE[0]["name"]
assert win.combo_model.currentData() == _esperado, \
    f"esperado {_esperado}, salio {win.combo_model.currentData()}"

# --- caso error: get_gemini_models revienta -> loguea y combo queda con fallback ---
def fake_models_rota(api_key):
    raise RuntimeError("red caida")

setup_module.get_gemini_models = fake_models_rota
win.input_api.setText(win.input_api.text() + " ")  # re-dispara on_api_key_changed
for _ in range(50):
    app.processEvents()
    if win.combo_model.itemText(0) not in ("Cargando modelos...",):
        break
    time.sleep(0.1)
assert win.combo_model.itemText(0) == "No se encontraron modelos", win.combo_model.itemText(0)
print("2. error de red -> combo 'No se encontraron modelos' (excepcion logueada)")

win.close()
print("\nTODO OK")
