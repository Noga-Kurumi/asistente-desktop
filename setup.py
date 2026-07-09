import os
import sys
import json
import time
from pynput import keyboard
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QLineEdit, QTextEdit, QCheckBox, QPushButton, 
                                QComboBox, QMessageBox)
from PySide6.QtCore import Qt, QObject, Signal

CONFIG_FILE = "config.json"

class KeyCatcher(QObject):
    key_caught = Signal(str)

    def __init__(self):
        super().__init__()
        self.listener = None

    def start_listening(self):
        self.listener = keyboard.Listener(on_press=self.on_catch)
        self.listener.start()

    def on_catch(self, key):
        if hasattr(key, 'name'):
            key_val = f"Key.{key.name}"
        elif hasattr(key, 'char') and key.char is not None:
            key_val = key.char
        else:
            key_val = str(key).replace("'", "")
        self.key_caught.emit(key_val)
        return False

class SetupWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración del Asistente")
        self.resize(400, 520)
        self.config_data = self.load_config()
        
        self.catcher = KeyCatcher()
        self.catcher.key_caught.connect(self.on_hotkey_caught)
        self.init_ui()

    def load_config(self):
        default_config = {
            "username": "Noga", 
            "api_key": "", 
            "active_avatar": "", 
            "active_voice": "", 
            "use_gpu": True, 
            "launch_with_system": False,
            "hotkey": "Key.alt_r",
            "custom_instructions": ""
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    default_config.update(data)
                    return default_config
            except Exception:
                pass
        return default_config

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Nombre de Usuario:"))
        self.input_user = QLineEdit(self.config_data.get("username", ""))
        layout.addWidget(self.input_user)

        layout.addWidget(QLabel("Anthropic API Key:"))
        self.input_api = QLineEdit(self.config_data.get("api_key", ""))
        self.input_api.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.input_api)

        layout.addWidget(QLabel("Tecla de Grabación (Hold-to-Talk):"))
        self.btn_hotkey = QPushButton(f"Capturar Tecla: [ {self.config_data.get('hotkey')} ]")
        self.btn_hotkey.clicked.connect(self.start_key_capture)
        layout.addWidget(self.btn_hotkey)

        layout.addWidget(QLabel("Instrucciones Personalizadas (System Prompt):"))
        self.input_custom = QTextEdit()
        self.input_custom.setPlainText(self.config_data.get("custom_instructions", ""))
        self.input_custom.setMaximumHeight(80)
        layout.addWidget(self.input_custom)

        layout.addWidget(QLabel("Avatar 3D (.vrm):"))
        self.combo_avatar = QComboBox()
        self.populate_combo(self.combo_avatar, "avatars", ".vrm", self.config_data.get("active_avatar", ""))
        layout.addWidget(self.combo_avatar)

        layout.addWidget(QLabel("Voz Kokoro (desde voices.json):"))
        self.combo_voice = QComboBox()
        self.populate_voices_combo(self.combo_voice, self.config_data.get("active_voice", ""))
        layout.addWidget(self.combo_voice)

        self.check_gpu = QCheckBox("Usar Aceleración por GPU (CUDA)")
        self.check_gpu.setChecked(self.config_data.get("use_gpu", True))
        layout.addWidget(self.check_gpu)

        self.check_startup = QCheckBox("Iniciar con Windows")
        self.check_startup.setChecked(self.config_data.get("launch_with_system", False))
        layout.addWidget(self.check_startup)

        btn_save = QPushButton("Guardar y Probar Sistema")
        btn_save.clicked.connect(self.save_config)
        layout.addSpacing(10)
        layout.addWidget(btn_save)

        self.setLayout(layout)

    def populate_combo(self, combo, folder, extension, current_val):
        """Función original para poblar los VRM desde la carpeta."""
        combo.addItem("Ninguno seleccionado")
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(extension)]
            for file in files:
                combo.addItem(file)
                if file == current_val:
                    combo.setCurrentText(file)

    def populate_voices_combo(self, combo, current_val):
        voces_js = [
            "es_gs", "es_am", "es_sz", "es_ug",
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", 
            "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", 
            "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", 
            "am_puck", "am_santa", "bf_emma", "bf_isabella", "bm_george", "bm_lewis", 
            "bf_alice", "bf_lily", "bm_daniel", "bm_fable"
        ]
        
        # Si está vacío o hay basura vieja, forzamos un default
        if not current_val or current_val not in voces_js:
            current_val = "af_bella"

        for v in voces_js:
            combo.addItem(v)
            
        combo.setCurrentText(current_val)

    def start_key_capture(self):
        self.btn_hotkey.setText("Escuchando... Presioná una tecla")
        self.btn_hotkey.setEnabled(False)
        self.btn_hotkey.setStyleSheet("background-color: #ffaa00; color: black; font-weight: bold;")
        self.catcher.start_listening()

    def on_hotkey_caught(self, key_str):
        self.config_data["hotkey"] = key_str
        self.btn_hotkey.setText(f"Capturar Tecla: [ {key_str} ]")
        self.btn_hotkey.setEnabled(True)
        self.btn_hotkey.setStyleSheet("")

    def save_config(self):
        self.config_data["username"] = self.input_user.text()
        self.config_data["api_key"] = self.input_api.text()
        self.config_data["custom_instructions"] = self.input_custom.toPlainText().strip()
        
        avatar_val = self.combo_avatar.currentText()
        self.config_data["active_avatar"] = avatar_val if avatar_val != "Ninguno seleccionado" else ""
        
        voice_val = self.combo_voice.currentText()
        self.config_data["active_voice"] = voice_val if voice_val != "Ninguno seleccionado" else ""
        
        self.config_data["use_gpu"] = self.check_gpu.isChecked()
        self.config_data["launch_with_system"] = self.check_startup.isChecked()

        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config_data, f, indent=4)

        self.handle_startup_os(self.config_data["launch_with_system"])

        QMessageBox.information(self, "Guardado", "Configuración actualizada.")
        self.close()

    def handle_startup_os(self, enable):
        if os.name == 'nt':
            import winreg as reg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            try:
                key = reg.OpenKey(reg.HKEY_CURRENT_USER, key_path, 0, reg.KEY_ALL_ACCESS)
                if enable:
                    python_exe = sys.executable
                    script_path = os.path.abspath("launch.py")
                    reg.SetValueEx(key, "AsistenteHaiku", 0, reg.REG_SZ, f'"{python_exe}" "{script_path}"')
                else:
                    try:
                        reg.DeleteValue(key, "AsistenteHaiku")
                    except FileNotFoundError:
                        pass
                reg.CloseKey(key)
            except Exception as e:
                print(f"Error gestionando startup: {e}")

_setup_window_instance = None

def run_setup_window():
    global _setup_window_instance
    app = QApplication.instance()
    needs_exec = False
    if not app:
        app = QApplication(sys.argv)
        needs_exec = True
        
    _setup_window_instance = SetupWindow()
    _setup_window_instance.show()
    if needs_exec:
        app.exec()

if __name__ == "__main__":
    run_setup_window()