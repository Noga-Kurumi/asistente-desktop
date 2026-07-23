"""Lanzador del asistente de voz.

Secuencia de arranque:
1. Si no estamos en el venv (.venv), lo crea si hace falta y se re-ejecuta dentro.
2. Ya en el venv: setup_logging() (una vez, idempotente).
3. Verifica que los imports clave funcionan; solo si falta algo instala
   requirements.txt (nada de reinstalar/actualizar pip en cada arranque).
4. Valida la configuración (config.validate()); si falta algo, abre setup.py.
5. Arranca main.run_app(). Cualquier error se loguea y se muestra al usuario
   con un mensaje claro (QMessageBox si se puede, consola si no).
"""

import logging
import os
import subprocess
import sys
import venv

# Constantes de rutas (ancladas al directorio del script, no al CWD).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
REQUIREMENTS = os.path.join(BASE_DIR, "requirements.txt")

# (módulo importable, paquete pip) que la app necesita para arrancar.
REQUIRED_IMPORTS = [
    ("PySide6", "PySide6"),
    ("sounddevice", "sounddevice"),
    ("numpy", "numpy"),
    ("onnxruntime", "onnxruntime"),
    ("google.genai", "google-genai"),
    ("pywhispercpp", "pywhispercpp"),
    ("mss", "mss"),
    ("pynput", "pynput"),
    ("requests", "requests"),
    # Contexto timeline: recolectores (A/B), meeting audio (C) y MCP (D).
    ("win32gui", "pywin32"),
    ("psutil", "psutil"),
    ("winsdk", "winsdk"),
    ("pyaudiowpatch", "PyAudioWPatch"),
    ("pycaw", "pycaw"),
    ("mcp", "mcp"),
]

logger = logging.getLogger(__name__)


def is_in_venv():
    """Verifica si estamos corriendo dentro del entorno virtual."""
    return sys.prefix != sys.base_prefix


def get_venv_python():
    """Devuelve la ruta exacta del Python dentro del venv dependiendo del SO."""
    if os.name == 'nt':  # Windows
        return os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    return os.path.join(VENV_DIR, 'bin', 'python')


def setup_venv():
    """Crea el venv e instala dependencias. SOLO se llama si el venv no existe."""
    print(f"Creando entorno virtual en {VENV_DIR} (primera ejecución)...")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)

    python_executable = get_venv_python()

    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([python_executable, "-m", "pip", "install", "-r", REQUIREMENTS, "--prefer-binary"])


def relaunch_in_venv():
    """Corta la ejecución actual y la reinicia usando el Python del venv."""
    python_executable = get_venv_python()
    os.execv(python_executable, [python_executable] + sys.argv)


def missing_imports():
    """Devuelve los paquetes pip cuyo módulo no se puede importar."""
    import importlib

    missing = []
    for module_name, package_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    return missing


def install_requirements():
    """Instala requirements.txt con el pip del venv actual."""
    logger.info("📦 [LAUNCH] Instalando dependencias desde requirements.txt...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS, "--prefer-binary"])


def notify_user(title, message):
    """Muestra un mensaje al usuario: QMessageBox si hay Qt, consola si no."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        QApplication.instance() or QApplication(sys.argv)
        QMessageBox.warning(None, title, message)
    except Exception:
        print(f"\n*** {title} ***\n{message}\n")


def main():
    # Modo congelado (PyInstaller): todo viene empaquetado; no hay venv que
    # crear ni dependencias que instalar.
    if getattr(sys, "frozen", False):
        from modules.log_setup import setup_logging
        setup_logging()
        logger.info("✅ [LAUNCH] Modo ejecutable (PyInstaller), dependencias empaquetadas")
        try:
            import main as main_module
            main_module.run_app()
        except Exception as e:
            logger.error("❌ [LAUNCH] Error fatal al arrancar: %s", e, exc_info=True)
            notify_user("Error al arrancar el asistente",
                        f"{e}\n\nRevisá logs/asistente.log para más detalle.")
            sys.exit(1)
        return

    # 1. Si NO estamos en el venv, tenemos que entrar.
    if not is_in_venv():
        if not os.path.exists(get_venv_python()):
            try:
                setup_venv()
            except subprocess.CalledProcessError as e:
                print(f"ERROR: no se pudo preparar el entorno virtual: {e}")
                sys.exit(1)
        relaunch_in_venv()
        return  # os.execv reemplaza el proceso; el return es buena práctica

    # --- A PARTIR DE ACÁ, YA ESTAMOS 100% ADENTRO DEL VENV ---

    # 2. Logging centralizado, una sola vez (idempotente).
    from modules.log_setup import setup_logging
    setup_logging()

    # 3. Dependencias: solo se instala si realmente falta algo.
    missing = missing_imports()
    if missing:
        logger.warning("⚠️ [LAUNCH] Dependencias faltantes: %s", ", ".join(missing))
        try:
            install_requirements()
        except subprocess.CalledProcessError as e:
            logger.error("❌ [LAUNCH] pip install falló: %s", e, exc_info=True)
            notify_user("Error de dependencias",
                        "No se pudieron instalar las dependencias.\n"
                        f"Revisá logs/asistente.log para más detalle.\n\n{e}")
            sys.exit(1)
        missing = missing_imports()
        if missing:
            logger.error("❌ [LAUNCH] Siguen faltando tras instalar: %s", missing)
            notify_user("Error de dependencias",
                        "Estos paquetes no se pudieron importar tras instalar:\n"
                        + ", ".join(missing))
            sys.exit(1)
        logger.info("✅ [LAUNCH] Dependencias instaladas correctamente")
    else:
        logger.info("✅ [LAUNCH] Dependencias OK, no hace falta instalar nada")

    # 4. Configuración: si falta algo obligatorio, abrir la ventana de setup.
    from modules.config_manager import get_config
    config = get_config()
    missing_fields = config.validate()
    if missing_fields:
        logger.warning("⚠️ [LAUNCH] Config incompleta (%s); abriendo setup",
                       ", ".join(missing_fields))
        import setup as setup_module
        setup_module.run_setup_window()
        config.reload()
        missing_fields = config.validate()
        if missing_fields:
            logger.warning("⚠️ [LAUNCH] Config sigue incompleta (%s); se arranca igual "
                           "(el asistente avisará con fallbacks hablados)",
                           ", ".join(missing_fields))
            notify_user("Configuración incompleta",
                        "Faltan campos obligatorios: " + ", ".join(missing_fields) +
                        ".\nEl asistente arrancará, pero no responderá consultas "
                        "hasta que completes la configuración.")

    # 5. Arranque de la aplicación.
    try:
        import main as main_module
        main_module.run_app()
    except Exception as e:
        logger.error("❌ [LAUNCH] Error fatal al arrancar: %s", e, exc_info=True)
        notify_user("Error al arrancar el asistente",
                    f"{e}\n\nRevisá logs/asistente.log para más detalle.")
        sys.exit(1)


if __name__ == "__main__":
    main()
