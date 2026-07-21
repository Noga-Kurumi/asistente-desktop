import os
import sys
import json
import subprocess
import venv

# Constantes de rutas (ancladas al directorio del script, no al CWD)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
REQUIREMENTS = os.path.join(BASE_DIR, "requirements.txt")

def is_in_venv():
    """Verifica si estamos corriendo dentro del entorno virtual."""
    return sys.prefix != sys.base_prefix

def get_venv_python():
    """Devuelve la ruta exacta del Python dentro del venv dependiendo del SO."""
    if os.name == 'nt': # Windows
        return os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    return os.path.join(VENV_DIR, 'bin', 'python')

def setup_venv():
    """Crea el venv e instala dependencias. SOLO se llama si el venv no existe."""
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)

    python_executable = get_venv_python()

    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([python_executable, "-m", "pip", "install", "-r", REQUIREMENTS, "--prefer-binary"])

def relaunch_in_venv():
    """Corta la ejecución actual y la reinicia usando el Python del venv."""
    python_executable = get_venv_python()
    os.execv(python_executable, [python_executable] + sys.argv)

def check_config():
    """Verifica si el config.json tiene todos los campos obligatorios."""
    if not os.path.exists(CONFIG_FILE):
        return False
        
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Verificar campos obligatorios (api_key NO es obligatorio)
            required_fields = ["username", "avatar_name", "active_avatar", "active_voice"]
            for field in required_fields:
                if not config.get(field) or config.get(field) == "":
                    return False
    except Exception:
        return False
        
    return True

def main():
    # 1. Si NO estamos en el venv, tenemos que entrar.
    if not is_in_venv():
        # Si el ejecutable no existe, armamos todo el setup primero
        if not os.path.exists(get_venv_python()):
            setup_venv()
        
        # Una vez que aseguramos que el venv existe, nos metemos ahí
        relaunch_in_venv()
        return # Técnicamente os.execv frena todo, pero el return es buena práctica

    # --- A PARTIR DE ACÁ, YA ESTAMOS 100% ADENTRO DEL VENV ---
    
    if not check_config():
        import setup
        setup.run_setup_window()

    import main # Llamamos al archivo central
    main.run_app()

if __name__ == "__main__":
    main()