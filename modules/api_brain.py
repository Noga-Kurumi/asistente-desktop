import os
import re
import base64
import io
import logging
import requests
import threading
from mss import mss
from PIL import Image
from anthropic import Anthropic
from google import genai
from google.genai import types
from PySide6.QtCore import QObject, Signal

from modules.config_manager import get_config

logger = logging.getLogger(__name__)


def get_gemini_models(api_key):
    """Obtiene la lista de modelos Gemini disponibles que soportan generateContent.

    Función de módulo: no requiere instanciar AssistantBrain (ni cliente
    Anthropic, ni chequeo de Screenpipe) solo para listar modelos.

    Args:
        api_key: API key de Gemini a usar.

    Returns:
        Lista de dicts {'name', 'display_name'} ordenada por nombre, o [] si falla.
    """
    logger.info("🔍 [BRAIN] get_gemini_models() llamado")

    if not api_key:
        logger.info("🔍 [BRAIN] No hay API key, retornando []")
        return []

    try:
        temp_client = genai.Client(api_key=api_key)

        logger.info("🔍 [BRAIN] Obteniendo modelos Gemini...")
        models = temp_client.models.list()
        available_models = []

        models_list = list(models)
        logger.info(f"🔍 [BRAIN] Total de modelos encontrados: {len(models_list)}")

        for model in models_list:
            # Extraer solo el nombre del modelo (quitar 'models/' prefijo)
            model_name = model.name
            if model_name.startswith("models/"):
                model_name = model_name[7:]

            name_lower = model_name.lower()

            # El nuevo SDK usa 'supported_actions' en lugar de 'supported_generation_methods'
            actions = model.supported_actions or []
            actions_lower = [action.lower() for action in actions]

            # 1. ¿Genera texto, imágenes y corre MCP? (Se llama 'generatecontent')
            if "generatecontent" not in actions_lower:
                logger.debug(f"⏭️ [BRAIN] Modelo excluido (no soporta generateContent): {model_name}")
                continue

            # 2. Solo la familia Gemini
            if "gemini" not in name_lower:
                logger.debug(f"⏭️ [BRAIN] Modelo excluido (no es Gemini): {model_name}")
                continue

            # 3. Excluir experimentales o versiones inestables de testing
            if "-exp" in name_lower or "preview" in name_lower or "experimental" in name_lower:
                logger.debug(f"⏭️ [BRAIN] Modelo excluido (experimental/preview): {model_name}")
                continue

            # 4. Excluir especializaciones de imagen pura o audio nativo
            if "-image" in name_lower or "-audio" in name_lower:
                logger.debug(f"⏭️ [BRAIN] Modelo excluido (especialización imagen/audio): {model_name}")
                continue

            # 5. Mantener solo los Flash (Los reyes del free-tier)
            if "flash" not in name_lower:
                logger.debug(f"⏭️ [BRAIN] Modelo excluido (no es flash): {model_name}")
                continue

            display_name = model.display_name if hasattr(model, 'display_name') else model_name
            available_models.append({'name': model_name, 'display_name': display_name})
            logger.debug(f"✅ [BRAIN] Modelo apto: {model_name} ({display_name})")

        available_models.sort(key=lambda x: x['name'])
        logger.info(f"🔍 [BRAIN] Modelos disponibles después del filtro: {len(available_models)}")
        return available_models
    except Exception as e:
        logger.error(f"❌ [BRAIN] Error obteniendo modelos Gemini: {e}", exc_info=True)
        return []


class AssistantBrain(QObject):
    # Señales para comunicarse con la UI y el Audio Core
    text_chunk_ready = Signal(str)
    point_action_ready = Signal(int, int)
    thinking_started = Signal()
    thinking_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.api_provider = self.config.get("api_provider", "anthropic")

        # Buffer de memoria a corto plazo (Ahora lo usamos para los DOS proveedores)
        self.message_history = []
        self.MAX_HISTORY = 10

        # Inicializar cliente según proveedor
        if self.api_provider == "gemini":
            self.client = genai.Client(api_key=self.config.get("api_key", ""))
            self.gemini_model = self.config.get("gemini_model", "gemini-2.0-flash")
            self.anthropic_model = None
        else:
            self.client = Anthropic(api_key=self.config.get("api_key", ""))
            self.gemini_model = None
            self.anthropic_model = self.config.get("anthropic_model", "claude-3-5-haiku-20241022")

        # Verificar disponibilidad de Screenpipe
        self.screenpipe_available = self._check_screenpipe_availability()

        # Captura de pantalla guardada (tomada al iniciar grabación)
        self.captured_image = None

        custom_inst = self.config.get("custom_instructions", "")
        avatar_name = self.config.get("avatar_name", "Lindsay")

        self.system_prompt = f"""
        You are a virtual desktop assistant named {avatar_name} for a senior developer named {self.config.get('username', 'Usuario')}.
        You are currently viewing the user's main screen.

        STRICT RULES:
        1. PERSONA: Speak entirely in Neutral Spanish (Español Neutro). DO NOT use any regional slang, idioms, or colloquialisms. Maintain a direct, natural, and professional tone.
        2. IDENTITY: Your name is {avatar_name}. You may introduce yourself as {avatar_name} and refer to yourself by this name when appropriate.
        3. FORMATTING: Do NOT use complex Markdown (no bolding, no code blocks) because your response will be read aloud by a TTS engine. Speak naturally.

        CUSTOM USER INSTRUCTIONS:
        {custom_inst if custom_inst else "None."}
        """

    def _check_screenpipe_availability(self):
        """Verifica si Screenpipe está disponible al inicio y devuelve el resultado."""
        try:
            response = requests.get(
                "http://localhost:3030/search",
                params={"q": "test", "limit": 1, "content_type": "ocr"},
                timeout=1.0
            )
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _get_primary_monitor(sct):
        """Elige el monitor principal: el que empieza en (0,0). Fallback: el primero real."""
        real_monitors = sct.monitors[1:]  # monitors[0] es el bounding box combinado
        for mon in real_monitors:
            if mon["left"] == 0 and mon["top"] == 0:
                return mon
        return real_monitors[0]

    def capture_screen_bytes(self):
        """Captura el monitor principal y devuelve JPEG comprimido en bytes."""
        try:
            with mss() as sct:
                monitor = self._get_primary_monitor(sct)
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

                # Compresión a 720p (suficiente para el OCR interno del LLM)
                target_width = 1280
                if img.size[0] > target_width:
                    ratio = target_width / float(img.size[0])
                    target_height = int(float(img.size[1]) * ratio)
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=60)
                return buffer.getvalue()
        except Exception as e:
            logger.error(f"❌ [VISIÓN] Error al capturar pantalla: {e}")
            return None

    def capture_and_store_screen(self):
        """Captura la pantalla actual (bytes JPEG) y la guarda para el próximo query."""
        self.captured_image = self.capture_screen_bytes()

    def search_screenpipe_local(self, query, limit=5):
        """Hace un request rápido al daemon local de Screenpipe."""
        try:
            response = requests.get(
                "http://localhost:3030/search",
                params={"q": query, "limit": limit, "content_type": "ocr"},
                timeout=2.0
            )
            if response.status_code == 200:
                data = response.json()
                results = [item['content']['text'] for item in data.get('data', []) if 'content' in item]
                return "\n".join(results) if results else "No encontré nada relevante en los logs recientes."
            return "Error: Screenpipe respondió con código diferente a 200."
        except Exception:
            return "Error de conexión con Screenpipe local."

    @staticmethod
    def _strip_gemini_images(content):
        """Reemplaza las imágenes de un turno de Gemini por un placeholder de texto.

        Evita reenviar screenshots viejos en cada llamada (tokens al pedo y
        contexto confuso). Solo la imagen del turno actual debe viajar.
        """
        text_parts = [
            types.Part(text="[captura de pantalla anterior]")
            if getattr(part, "inline_data", None) is not None else part
            for part in content.parts
        ]
        content.parts = text_parts

    def _trim_history_gemini(self):
        """Poda el historial específico para el formato de Gemini."""
        if len(self.message_history) > self.MAX_HISTORY:
            # Mantener siempre el primer mensaje si existe (para contexto inicial)
            if len(self.message_history) > 1:
                self.message_history = [self.message_history[0]] + self.message_history[-(self.MAX_HISTORY-1):]
            else:
                self.message_history = self.message_history[-self.MAX_HISTORY:]

        # Limpiar imágenes de turnos anteriores: solo la última (la del turno
        # actual) viaja a la API
        for msg in self.message_history[:-1]:
            if msg.role == "user":
                self._strip_gemini_images(msg)

    def _trim_history_anthropic(self):
        """Tu lógica original de podado para Anthropic."""
        if len(self.message_history) > self.MAX_HISTORY:
            self.message_history = self.message_history[-self.MAX_HISTORY:]

        for i in range(len(self.message_history) - 1):
            msg = self.message_history[i]
            if msg["role"] == "user" and isinstance(msg["content"], list):
                text_only = [c for c in msg["content"] if c.get("type") == "text" or c.get("type") == "tool_result"]
                msg["content"] = text_only

    def process_query_async(self, user_prompt):
        api_key = self.config.get("api_key", "").strip()
        if not api_key:
            self.error_occurred.emit("ERR_NO_API")
            return

        threading.Thread(target=self._run_llm_chain, args=(user_prompt,), daemon=True).start()

    def _run_llm_chain(self, user_prompt):
        if self.api_provider == "gemini":
            self._run_gemini_chain(user_prompt)
        else:
            self._run_anthropic_chain(user_prompt)

    def _run_gemini_chain(self, user_prompt):
            self.thinking_started.emit()
            img_bytes = self.captured_image

            try:
                # 1. Armamos el payload del turno actual
                current_turn_parts = []
                if img_bytes:
                    current_turn_parts.append(
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    )
                current_turn_parts.append(types.Part(text=user_prompt))

                # 2. Agregamos el input del usuario a la memoria
                self.message_history.append(types.Content(role="user", parts=current_turn_parts))
                self._trim_history_gemini()

                # 3. Llamada con el NUEVO SDK usando "contents" (historial completo)
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=self.message_history, # Le pasamos toda la charla previa
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        max_output_tokens=1000,
                        temperature=0.7
                    )
                )

                full_response_text = response.text

                # 4. Guardamos la respuesta del asistente en memoria
                if full_response_text:
                    self.message_history.append(types.Content(role="model", parts=[types.Part(text=full_response_text)]))

                # 5. Lógica de Pointing
                point_pattern = re.compile(r'<PointTo,\s*(\d+),\s*(\d+)>')
                point_match = point_pattern.search(full_response_text)
                if point_match:
                    x, y = int(point_match.group(1)), int(point_match.group(2))
                    self.point_action_ready.emit(x, y)
                    full_response_text = point_pattern.sub('', full_response_text)

                if full_response_text.strip():
                    self.text_chunk_ready.emit(full_response_text.strip())

            except Exception as e:
                logger.error(f"❌ [BRAIN] Error en Gemini: {e}", exc_info=True)
                self.error_occurred.emit("ERR_GENERIC")

            finally:
                self.thinking_finished.emit()

    def _run_anthropic_chain(self, user_prompt):
        self.thinking_started.emit()
        img_bytes = self.captured_image

        # Armamos el input actual
        content_array = []
        if img_bytes:
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            content_array.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            })
        content_array.append({"type": "text", "text": user_prompt})

        # Agregamos al historial y lo podamos
        self.message_history.append({"role": "user", "content": content_array})
        self._trim_history_anthropic()

        tools = [{
            "name": "search_screenpipe",
            "description": "Busca en los logs locales de OCR de la pantalla del usuario. Úsalo si el usuario pregunta por algo que sucedió en el pasado o que ya no está visible.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La palabra clave a buscar"}
                },
                "required": ["query"]
            }
        }]

        try:
            response = self.client.messages.create(
                model=self.anthropic_model,
                max_tokens=500,
                system=self.system_prompt,
                tools=tools,
                messages=self.message_history
            )

            if response.stop_reason == "tool_use":
                tool_use = next(block for block in response.content if block.type == "tool_use")
                query = tool_use.input.get("query")

                tool_result = self.search_screenpipe_local(query)

                # Guardamos el uso de la herramienta y el resultado en la memoria
                self.message_history.append({"role": "assistant", "content": response.content})
                self.message_history.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result
                    }]
                })

            stream = self.client.messages.create(
                model=self.anthropic_model,
                max_tokens=500,
                system=self.system_prompt,
                messages=self.message_history,
                stream=True
            )

            # Extraemos el string completo de la IA mientras la UI lo lee en vivo
            full_response_text = self._process_stream(stream)

            # Guardamos la respuesta final en la memoria para el próximo turno
            if full_response_text:
                self.message_history.append({"role": "assistant", "content": full_response_text})

        except requests.exceptions.ConnectionError:
            self.error_occurred.emit("ERR_NETWORK")
        except Exception as e:
            logger.error(f"❌ [BRAIN] Crash en la API: {e}", exc_info=True)
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                self.error_occurred.emit("ERR_AUTH")
            elif "credit" in str(e).lower() or "balance" in str(e).lower():
                self.error_occurred.emit("ERR_QUOTA")
            else:
                self.error_occurred.emit("ERR_GENERIC")

        finally:
            self.thinking_finished.emit()

    def _process_stream(self, stream):
        buffer_text = ""
        full_llm_output = "" # Acumulador para la memoria a corto plazo
        sentence_end_pattern = re.compile(r'([.?!])\s')
        point_pattern = re.compile(r'<PointTo,\s*(\d+),\s*(\d+)>')

        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                chunk = event.delta.text
                buffer_text += chunk
                full_llm_output += chunk # Guardamos el bloque tal cual sale

                match = sentence_end_pattern.search(buffer_text)
                if match:
                    cut_index = match.end()
                    sentence = buffer_text[:cut_index]
                    buffer_text = buffer_text[cut_index:]

                    point_match = point_pattern.search(sentence)
                    if point_match:
                        x, y = int(point_match.group(1)), int(point_match.group(2))
                        self.point_action_ready.emit(x, y)
                        sentence = point_pattern.sub('', sentence)

                    clean_sentence = sentence.strip()
                    if clean_sentence:
                        self.text_chunk_ready.emit(clean_sentence)

        if buffer_text.strip():
            final_sentence = buffer_text.strip()
            point_match = point_pattern.search(final_sentence)
            if point_match:
                x, y = int(point_match.group(1)), int(point_match.group(2))
                self.point_action_ready.emit(x, y)
                final_sentence = point_pattern.sub('', final_sentence).strip()

            if final_sentence:
                self.text_chunk_ready.emit(final_sentence)

        return full_llm_output
