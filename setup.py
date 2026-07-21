import os
import sys
import json
import re
import numpy as np
import sounddevice as sd
from pynput import keyboard
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QLineEdit, QTextEdit, QCheckBox, QPushButton, 
                                QComboBox, QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QObject, Signal, QTimer
from PySide6.QtGui import QIcon

# Importar función para obtener voces disponibles
from modules.tts_core import get_available_voices
from modules.api_brain import AssistantBrain

# Usar ruta absoluta para config.json
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

# Regex para detectar proveedores de API
REGEX_ANTHROPIC = r"^sk-ant-(?:api\d{2}|oat\d{2}|[A-Za-z0-9_-]+)-[A-Za-z0-9_-]{40,}$"
REGEX_GEMINI = r"^(?:AIzaSy[A-Za-z0-9_-]{33}|AQ\.[A-Za-z0-9_-]+)$"

def detect_provider(api_key):
    """Detecta el proveedor de API basado en el formato de la clave."""
    if not api_key or not api_key.strip():
        return None
    if re.match(REGEX_ANTHROPIC, api_key.strip()):
        return "anthropic"
    if re.match(REGEX_GEMINI, api_key.strip()):
        return "gemini"
    return None

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
    def __init__(self, from_system_tray=False):
        super().__init__()
        self.setWindowTitle("Configuración del Asistente")
        
        # Usar ruta absoluta para el icono
        icon_path = os.path.join(os.path.dirname(__file__), "app.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.resize(400, 520)
        self.config_data = self.load_config()
        self.is_first_run = not os.path.exists(CONFIG_FILE)
        self.from_system_tray = from_system_tray  # True si se abrió desde system tray, False si fue por falta de datos
        
        self.catcher = KeyCatcher()
        self.catcher.key_caught.connect(self.on_hotkey_caught)
        
        # Cargar icono para los popups
        self.app_icon = None
        icon_path = os.path.join(os.path.dirname(__file__), "app.ico")
        if os.path.exists(icon_path):
            self.app_icon = QIcon(icon_path)
        
        # Variables para test de micrófono
        self.test_stream = None
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.update_volume_meter)
        self.current_volume = 0
        
        self.init_ui()
    
    def closeEvent(self, event):
        """Limpiar recursos al cerrar la ventana"""
        self.stop_mic_test()
        event.accept()

    def load_config(self):
        default_config = {
            "username": "Noga", 
            "api_key": "", 
            "active_avatar": "", 
            "active_voice": "", 
            "launch_with_system": False,
            "hotkey": "Key.alt_r",
            "custom_instructions": "",
            "audio_device": None,
            "whisper_model": "tiny",
            "whisper_quantization": "none",
            "gemini_model": ""
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

        layout.addWidget(QLabel("Nombre del Avatar:"))
        self.input_avatar_name = QLineEdit(self.config_data.get("avatar_name", "Kurumi"))
        layout.addWidget(self.input_avatar_name)

        layout.addWidget(QLabel("API Key:"))
        self.input_api = QLineEdit(self.config_data.get("api_key", ""))
        self.input_api.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_api.textChanged.connect(self.on_api_key_changed)
        layout.addWidget(self.input_api)
        
        # Label para mostrar proveedor detectado
        self.label_provider = QLabel("No hay clave")
        self.label_provider.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.label_provider)
        
        # Dropdown de modelos (inicialmente vacío)
        layout.addWidget(QLabel("Modelo:"))
        self.combo_model = QComboBox()
        self.combo_model.addItem("Configura API key primero", "")
        layout.addWidget(self.combo_model)
        
        # Inicializar detección de proveedor
        self.on_api_key_changed(self.input_api.text())

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

        layout.addWidget(QLabel("Modelo Whisper:"))
        self.combo_whisper_model = QComboBox()
        self.combo_whisper_model.addItem("Tiny (Rápido, menos preciso)", "tiny")
        self.combo_whisper_model.addItem("Small (Más preciso, más lento)", "small")
        
        # Seleccionar modelo actual
        current_model = self.config_data.get("whisper_model", "tiny")
        index = self.combo_whisper_model.findData(current_model)
        if index >= 0:
            self.combo_whisper_model.setCurrentIndex(index)
        layout.addWidget(self.combo_whisper_model)

        layout.addWidget(QLabel("Cuantización Whisper:"))
        self.combo_whisper_quant = QComboBox()
        self.combo_whisper_quant.addItem("Sin cuantización (FP16)", "none")
        self.combo_whisper_quant.addItem("Q5_1 (Balance tamaño/calidad)", "q5_1")
        self.combo_whisper_quant.addItem("Q8_0 (Mejor calidad quantizada)", "q8_0")
        
        # Seleccionar cuantización actual
        current_quant = self.config_data.get("whisper_quantization", "none")
        index = self.combo_whisper_quant.findData(current_quant)
        if index >= 0:
            self.combo_whisper_quant.setCurrentIndex(index)
        layout.addWidget(self.combo_whisper_quant)

        layout.addWidget(QLabel("Dispositivo de Audio (Micrófono):"))
        self.combo_audio = QComboBox()
        self.populate_audio_devices()
        layout.addWidget(self.combo_audio)

        # Test de micrófono
        audio_test_layout = QHBoxLayout()
        self.btn_test_mic = QPushButton("Testear Micrófono")
        self.btn_test_mic.clicked.connect(self.toggle_mic_test)
        self.btn_test_mic.setCheckable(True)
        audio_test_layout.addWidget(self.btn_test_mic)
        
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        self.volume_bar.setTextVisible(False)
        self.volume_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        audio_test_layout.addWidget(self.volume_bar)
        
        layout.addLayout(audio_test_layout)

        self.check_startup = QCheckBox("Iniciar con Windows")
        self.check_startup.setChecked(self.config_data.get("launch_with_system", False))
        layout.addWidget(self.check_startup)

        btn_save = QPushButton("Guardar Cambios")
        btn_save.clicked.connect(self.save_config)
        layout.addSpacing(10)
        layout.addWidget(btn_save)

        self.setLayout(layout)

    def populate_combo(self, combo, folder, extension, current_val):
        """Función original para poblar los VRM desde la carpeta."""
        combo.addItem("Ninguno seleccionado")
        # Usar ruta absoluta para la carpeta
        folder_path = os.path.join(os.path.dirname(__file__), folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
            for file in files:
                combo.addItem(file)
                if file == current_val:
                    combo.setCurrentText(file)

    def populate_voices_combo(self, combo, current_val):
        """Poblar el combo con voces dinámicas desde voices-v1.0.bin"""
        # Obtener voces disponibles del archivo binario
        voices = get_available_voices()
        
        if not voices:
            # Fallback si no se pueden cargar las voces
            combo.addItem("Error cargando voces")
            return
        
        # Si está vacío o hay basura vieja, forzamos un default (af_bella - Inglés US)
        if not current_val or current_val not in [v[0] for v in voices]:
            current_val = "af_bella"
        
        # Agregar voces al combo con formato "nombre - idioma"
        for voice_id, voice_name in voices:
            combo.addItem(voice_name, voice_id)  # voice_name como texto, voice_id como userData
        
        # Seleccionar la voz actual por userData
        index = combo.findData(current_val)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            # Fallback a af_bella si no se encuentra
            index = combo.findData("af_bella")
            if index >= 0:
                combo.setCurrentIndex(index)

    def populate_gemini_models(self):
        """Poblar el combo con modelos Gemini disponibles"""
        try:
            print(f"🔍 [SETUP] Iniciando populate_gemini_models")
            
            # Verificar si hay API key de Gemini
            api_key = self.config_data.get("api_key", "")
            print(f"🔍 [SETUP] API key presente: {bool(api_key)}")
            
            if not api_key:
                self.combo_model.addItem("Configura API key primero", "")
                return
            
            # Verificar si el proveedor es Gemini
            provider = detect_provider(api_key)
            print(f"🔍 [SETUP] Proveedor detectado: {provider}")
            
            if provider != "gemini":
                self.combo_model.addItem("Solo disponible con Gemini", "")
                return
            
            # Llamar directamente al método estático con la API key
            print(f"🔍 [SETUP] Llamando a get_gemini_models con API key")
            brain = AssistantBrain()
            models = brain.get_gemini_models(api_key=api_key)
            print(f"🔍 [SETUP] Modelos obtenidos: {len(models)}")
            
            if not models:
                self.combo_model.addItem("No se encontraron modelos", "")
                return
            
            # Agregar modelos al combo con display_name
            for model in models:
                self.combo_model.addItem(model['display_name'], model['name'])
            
            # Seleccionar modelo actual
            current_model = self.config_data.get("gemini_model", "")
            if current_model:
                index = self.combo_model.findData(current_model)
                if index >= 0:
                    self.combo_model.setCurrentIndex(index)
            elif self.combo_model.count() > 0:
                # Seleccionar el primero por defecto
                self.combo_model.setCurrentIndex(0)
                
        except Exception as e:
            print(f"❌ [SETUP] Error en populate_gemini_models: {e}")
            import traceback
            traceback.print_exc()
            self.combo_model.addItem("Error cargando modelos", "")

    def populate_audio_devices(self):
        """Poblar el combo con dispositivos de audio de entrada"""
        try:
            devices = sd.query_devices()
            current_device = self.config_data.get("audio_device")
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    device_name = f"[{i}] {dev['name']}"
                    self.combo_audio.addItem(device_name, i)
                    
                    # Seleccionar dispositivo actual si coincide
                    if current_device is not None and i == current_device:
                        self.combo_audio.setCurrentIndex(self.combo_audio.count() - 1)
        except Exception as e:
            self.combo_audio.addItem("Error cargando dispositivos")

    def toggle_mic_test(self):
        """Iniciar/detener el test de micrófono"""
        if self.btn_test_mic.isChecked():
            # Iniciar test
            device_index = self.combo_audio.currentData()
            if device_index is None:
                self.btn_test_mic.setChecked(False)
                QMessageBox.warning(self, "Error", "Selecciona un dispositivo de audio primero")
                return
            
            try:
                self.test_stream = sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    dtype='float32',
                    callback=self.audio_callback,
                    device=device_index
                )
                self.test_stream.start()
                self.test_timer.start(50)  # Actualizar cada 50ms
                self.btn_test_mic.setText("Detener Test")
            except Exception as e:
                self.btn_test_mic.setChecked(False)
                QMessageBox.warning(self, "Error", f"No se pudo iniciar el test: {e}")
        else:
            # Detener test
            self.stop_mic_test()

    def stop_mic_test(self):
        """Detener el test de micrófono"""
        if self.test_stream:
            try:
                self.test_stream.stop()
                self.test_stream.close()
            except:
                pass
            self.test_stream = None
        
        self.test_timer.stop()
        self.volume_bar.setValue(0)
        self.btn_test_mic.setText("Testear Micrófono")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback para capturar audio del micrófono"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Calcular nivel de volumen (RMS)
        rms = np.sqrt(np.mean(indata ** 2))
        # Normalizar a 0-100 para la barra de progreso
        level = min(rms * 1000, 100)
        
        # Guardar el nivel para actualizar en el hilo principal
        self.current_volume = level

    def update_volume_meter(self):
        """Actualizar la barra de volumen en el hilo principal"""
        if hasattr(self, 'current_volume'):
            self.volume_bar.setValue(int(self.current_volume))

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

    def on_api_key_changed(self, text):
        """Detecta el proveedor de API y actualiza el label."""
        provider = detect_provider(text)
        if provider == "anthropic":
            self.label_provider.setText("✓ Anthropic detectado")
            self.label_provider.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            self.combo_model.clear()
            self.combo_model.addItem("Anthropic usa modelo fijo", "")
        elif provider == "gemini":
            self.label_provider.setText("✓ Gemini detectado")
            self.label_provider.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            # Poblar modelos de Gemini
            self.populate_gemini_models()
        elif text.strip():
            self.label_provider.setText("✗ Clave inválida")
            self.label_provider.setStyleSheet("color: #F44336; font-size: 11px;")
            self.combo_model.clear()
            self.combo_model.addItem("Clave inválida", "")
        else:
            self.label_provider.setText("No hay clave")
            self.label_provider.setStyleSheet("color: #888; font-size: 11px;")
            self.combo_model.clear()
            self.combo_model.addItem("Configura API key primero", "")

    def save_config(self):
        # Detener test de micrófono si está activo
        if self.btn_test_mic.isChecked():
            self.stop_mic_test()
        
        # Guardar configuración anterior para detectar cambios
        old_config = self.config_data.copy() if os.path.exists(CONFIG_FILE) else {}
        
        self.config_data["username"] = self.input_user.text()
        self.config_data["avatar_name"] = self.input_avatar_name.text()
        self.config_data["api_key"] = self.input_api.text()
        self.config_data["api_provider"] = detect_provider(self.input_api.text()) or ""
        self.config_data["custom_instructions"] = self.input_custom.toPlainText().strip()
        
        avatar_val = self.combo_avatar.currentText()
        if avatar_val == "Ninguno seleccionado":
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Debes seleccionar un avatar para continuar.")
            msg.setIcon(QMessageBox.Icon.Warning)
            if self.app_icon:
                msg.setWindowIcon(self.app_icon)
            msg.exec()
            return
        self.config_data["active_avatar"] = avatar_val
        
        voice_val = self.combo_voice.currentData()  # Obtener el ID de la voz (userData)
        if voice_val is None:
            voice_val = self.combo_voice.currentText()
        self.config_data["active_voice"] = voice_val if voice_val and voice_val != "Error cargando voces" else ""
        
        # Guardar dispositivo de audio
        audio_device = self.combo_audio.currentData()
        self.config_data["audio_device"] = audio_device
        
        # Guardar configuración de whisper
        self.config_data["whisper_model"] = self.combo_whisper_model.currentData()
        self.config_data["whisper_quantization"] = self.combo_whisper_quant.currentData()
        
        # Guardar modelo seleccionado
        selected_model = self.combo_model.currentData()
        if selected_model and selected_model not in ["", "Configura API key primero", "Anthropic usa modelo fijo", "Solo disponible con Gemini", "No se encontraron modelos", "Error cargando modelos", "Clave inválida"]:
            self.config_data["gemini_model"] = selected_model
        else:
            # Limpiar el modelo si no es válido
            self.config_data["gemini_model"] = ""
        
        self.config_data["launch_with_system"] = self.check_startup.isChecked()

        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config_data, f, indent=4)

        self.handle_startup_os(self.config_data["launch_with_system"])

        # Detectar si hubo cambios
        config_changed = old_config != self.config_data
        
        # Verificar campos obligatorios (api_key NO es obligatorio)
        required_fields = ["username", "avatar_name", "active_avatar", "active_voice"]
        missing_fields = [field for field in required_fields if not self.config_data.get(field) or self.config_data.get(field) == ""]
        
        if missing_fields:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText(f"Faltan campos obligatorios: {', '.join(missing_fields)}")
            msg.setIcon(QMessageBox.Icon.Warning)
            if self.app_icon:
                msg.setWindowIcon(self.app_icon)
            msg.exec()
            return
        
        # Advertir si no hay API key pero permitir guardar
        if not self.config_data.get("api_key") or self.config_data.get("api_key") == "":
            msg = QMessageBox(self)
            msg.setWindowTitle("Advertencia")
            msg.setText("No has configurado una API Key. El software no responderá a consultas sin una clave válida.")
            msg.setIcon(QMessageBox.Icon.Warning)
            if self.app_icon:
                msg.setWindowIcon(self.app_icon)
            msg.exec()
        
        # Comportamiento según origen
        if not self.from_system_tray:
            # Se abrió por falta de datos - solo cerrar sin mensajes
            self.close()
        else:
            # Se abrió desde system tray
            if config_changed:
                msg = QMessageBox(self)
                msg.setWindowTitle("Reinicio Requerido")
                msg.setText("La configuración ha cambiado. Debes reiniciar el software manualmente para aplicar los cambios.")
                msg.setIcon(QMessageBox.Icon.Information)
                if self.app_icon:
                    msg.setWindowIcon(self.app_icon)
                msg.exec()
            # Siempre cerrar
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

def run_setup_window(from_system_tray=False):
    global _setup_window_instance
    app = QApplication.instance()
    needs_exec = False
    if not app:
        app = QApplication(sys.argv)
        needs_exec = True
        
    _setup_window_instance = SetupWindow(from_system_tray=from_system_tray)
    _setup_window_instance.show()
    if needs_exec:
        app.exec()
    
    return _setup_window_instance

if __name__ == "__main__":
    run_setup_window()