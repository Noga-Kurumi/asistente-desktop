"""Descarga de modelos on-first-run.

Modelos gestionados:
- Whisper ggml: models/ggml-{modelo}[-{cuantiz}].bin desde el repo
  ggerganov/whisper.cpp de HuggingFace.
- Kokoro TTS: kokoro-v1.0.int8.onnx (o kokoro-v1.0.onnx si tts_model="fp32")
  y voices-v1.0.bin en la raíz del repo.
  NOTA DE FUENTE (verificada 2026-07-22 contra la API de HF): el repo
  onnx-community/Kokoro-82M-ONNX NO publica esos nombres de archivo (tiene
  onnx/model_*.onnx y voces sueltas en voices/*.bin, sin un voices-v1.0.bin
  consolidado). Los binarios oficiales con esos nombres viven en el release
  model-files-v1.0 de thewh1teagle/kokoro-onnx en GitHub, que es lo que
  documenta el paquete kokoro-onnx. Se usa GitHub como fuente primaria y el
  modelo cuantizado de HF (onnx/model_quantized.onnx, equivalente int8) como
  fallback para el .onnx.

Descargas con requests (ya es dependencia), escritura atómica (.part +
os.replace) y reanudación con Range si quedó un .part de un intento anterior.
"""

import logging
import os
from typing import Any, List, Mapping, Optional

import requests

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

WHISPER_URL_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
KOKORO_GH_RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
KOKORO_HF_BASE = "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main"

_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_LOG_EVERY_FRACTION = 0.10  # progreso al log cada ~10%


def _whisper_filename(model: str, quantization: Optional[str]) -> str:
    if quantization in ("none", "", None):
        return f"ggml-{model}.bin"
    return f"ggml-{model}-{quantization}.bin"


def download_file(url: str, dest_path: str, timeout: int = 30) -> bool:
    """Descarga url a dest_path de forma atómica, con reanudación simple.

    Si existe dest_path.part se reanuda con Range; al terminar se renombra a
    dest_path. Devuelve True si el archivo quedó completo en dest_path.
    """
    part_path = dest_path + ".part"
    resume_from = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
            # 416: el .part ya estaba completo (o el servidor no acepta Range).
            if resp.status_code == 416 and resume_from:
                os.replace(part_path, dest_path)
                return True
            if resp.status_code not in (200, 206):
                logger.error("❌ [MODELS] HTTP %s descargando %s", resp.status_code, url)
                return False
            if resume_from and resp.status_code == 200:
                # Servidor ignoró el Range: empezar de cero.
                resume_from = 0

            total = resp.headers.get("Content-Length")
            total = int(total) + resume_from if total else None
            mode = "ab" if resume_from else "wb"

            downloaded = resume_from
            next_log = _LOG_EVERY_FRACTION
            with open(part_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        frac = downloaded / total
                        if frac >= next_log:
                            logger.info("⬇️ [MODELS] %s: %d%% de %.1f MB",
                                        os.path.basename(dest_path),
                                        int(frac * 100), total / 1e6)
                            next_log += _LOG_EVERY_FRACTION

        os.replace(part_path, dest_path)
        logger.info("✅ [MODELS] Descargado: %s", dest_path)
        return True
    except requests.RequestException as e:
        logger.error("❌ [MODELS] Error de red descargando %s: %s", url, e)
        return False
    except OSError as e:
        logger.error("❌ [MODELS] Error de disco escribiendo %s: %s", dest_path, e)
        return False


def _download_first_available(urls: List[str], dest_path: str) -> bool:
    for url in urls:
        if download_file(url, dest_path):
            return True
        logger.warning("⚠️ [MODELS] Falló la fuente %s, probando la siguiente", url)
    return False


def ensure_models(config: Mapping[str, Any]) -> List[str]:
    """Garantiza que los modelos necesarios existen en disco; descarga los que falten.

    Args:
        config: Mapping con whisper_model y whisper_quantization.

    Returns:
        Lista de rutas de archivos que faltaban y se descargaron con éxito.
        Los fallos se loguean (no lanzan excepción) y simplemente no aparecen
        en la lista.
    """
    downloaded: List[str] = []
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Whisper ggml ---
    model = str(config.get("whisper_model", "tiny"))
    quant = config.get("whisper_quantization", "q5_1")
    whisper_name = _whisper_filename(model, quant)
    whisper_path = os.path.join(MODELS_DIR, whisper_name)
    if not os.path.exists(whisper_path):
        logger.info("⬇️ [MODELS] Falta el modelo whisper %s, descargando...", whisper_name)
        url = f"{WHISPER_URL_BASE}/{whisper_name}"
        if download_file(url, whisper_path):
            downloaded.append(whisper_path)
        else:
            logger.error("❌ [MODELS] No se pudo descargar %s", whisper_name)

    # --- Kokoro ONNX: se descarga la variante pedida por tts_model ---
    # (si ya hay ALGUNA variante en disco no se descarga nada: tts_core cae
    # a la disponible con warning).
    kokoro_full = os.path.join(BASE_DIR, "kokoro-v1.0.onnx")
    kokoro_int8 = os.path.join(BASE_DIR, "kokoro-v1.0.int8.onnx")
    if not os.path.exists(kokoro_full) and not os.path.exists(kokoro_int8):
        prefer = str(config.get("tts_model", "int8"))
        if prefer == "fp32":
            wanted, urls = kokoro_full, [
                f"{KOKORO_GH_RELEASE}/kokoro-v1.0.onnx",
                f"{KOKORO_HF_BASE}/onnx/model.onnx",  # fp32 en HF
            ]
        else:
            wanted, urls = kokoro_int8, [
                f"{KOKORO_GH_RELEASE}/kokoro-v1.0.int8.onnx",
                f"{KOKORO_HF_BASE}/onnx/model_quantized.onnx",  # equivalente int8 en HF
            ]
        logger.info("⬇️ [MODELS] Falta el modelo Kokoro (%s), descargando...", prefer)
        if _download_first_available(urls, wanted):
            downloaded.append(wanted)
        else:
            logger.error("❌ [MODELS] No se pudo descargar el modelo Kokoro")

    # --- Voces Kokoro ---
    voices_path = os.path.join(BASE_DIR, "voices-v1.0.bin")
    if not os.path.exists(voices_path):
        logger.info("⬇️ [MODELS] Faltan las voces Kokoro, descargando...")
        url = f"{KOKORO_GH_RELEASE}/voices-v1.0.bin"
        if download_file(url, voices_path):
            downloaded.append(voices_path)
        else:
            logger.error("❌ [MODELS] No se pudieron descargar las voces Kokoro")

    if downloaded:
        logger.info("✅ [MODELS] Descargas completadas: %d archivo(s)", len(downloaded))
    else:
        logger.info("✅ [MODELS] Todos los modelos ya estaban en disco")
    return downloaded
