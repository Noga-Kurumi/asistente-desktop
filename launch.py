import os
import sys
import json
import subprocess
import venv

# Constantes de rutas
VENV_DIR = ".venv"
CONFIG_FILE = "config.json"
REQUIREMENTS = "requirements.txt"

def is_in_venv():
    """Verifica si estamos corriendo dentro del entorno virtual."""
    return sys.prefix != sys.base_prefix

def create_and_relaunch():
    """Crea el venv, instala dependencias y se vuelve a ejecutar a sí mismo."""
    print("🚀 [BOOT] Creando entorno virtual aislado...")
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(VENV_DIR)

    # Determinar el ejecutable de Python del venv
    if os.name == 'nt': # Windows
        python_executable = os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    else: # Mac/Linux
        python_executable = os.path.join(VENV_DIR, 'bin', 'python')

    print("📦 [BOOT] Actualizando pip y herramientas base...")
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    print("📦 [BOOT] Instalando dependencias (esto puede tardar un toque)...")
    # El flag --prefer-binary evita que intente compilar librerías complejas como 'av' desde cero
    subprocess.check_call([python_executable, "-m", "pip", "install", "-r", REQUIREMENTS, "--prefer-binary"])
    
    print("✅ [BOOT] Entorno listo. Relanzando...")
    # Relanzar este mismo script pero desde el venv
    os.execv(python_executable, [python_executable] + sys.argv)

def check_config():
    """Verifica si el config.json tiene la API Key. Si no, levanta setup.py."""
    if not os.path.exists(CONFIG_FILE):
        return False
        
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            if not config.get("api_key") or config.get("api_key") == "":
                return False
    except Exception:
        return False
        
    return True

def main():
    if not is_in_venv():
        create_and_relaunch()
        return

    print("🔍 [BOOT] Chequeando configuraciones...")
    if not check_config():
        print("⚠️ [BOOT] Faltan datos. Abriendo panel de configuración (podés cerrarlo y seguir igual)...")
        import setup
        setup.run_setup_window()

    print("🔥 [BOOT] Todo en verde. Levantando el orquestador principal...")
    import main # Llamamos al archivo central que vamos a crear ahora
    main.run_app()

if __name__ == "__main__":
    main()