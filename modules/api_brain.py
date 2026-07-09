import os
import re
import json
import base64
import io
import requests
import threading
from mss import mss
from PIL import Image
from anthropic import Anthropic
from PySide6.QtCore import QObject, Signal

CONFIG_FILE = "config.json"

class AssistantBrain(QObject):
    # Señales para comunicarse con la UI y el Audio Core
    text_chunk_ready = Signal(str)      
    point_action_ready = Signal(int, int) 
    thinking_started = Signal()
    thinking_finished = Signal()
    error_occurred = Signal(str)        

    def __init__(self):
        super().__init__()
        self.config = self._load_config()
        self.client = Anthropic(api_key=self.config.get("api_key", ""))
        
        # Buffer de memoria a corto plazo
        self.message_history = []
        self.MAX_HISTORY = 10 # Retiene los últimos 10 mensajes (5 idas y 5 vueltas)
        
        custom_inst = self.config.get("custom_instructions", "")

        self.system_prompt = f"""
        You are a virtual desktop assistant for a senior developer named {self.config.get('username', 'Usuario')}. 
        You are currently viewing the user's main screen.
        
        STRICT RULES:
        1. PERSONA: Speak entirely in Neutral Spanish (Español Neutro). DO NOT use any regional slang, idioms, or colloquialisms. Maintain a direct, natural, and professional tone.
        2. POINTING ACTION: If the user asks about something visual on the screen and you can identify its location, you MUST include the tag <PointTo, X, Y> in your response (where X and Y are the approximate pixel coordinates). Example: "El error está acá <PointTo, 450, 800> en la línea 42."
        3. FORMATTING: Do NOT use complex Markdown (no bolding, no code blocks) because your response will be read aloud by a TTS engine. Speak naturally.
        4. TOOL USE: If asked about recent past events or errors no longer on screen, use the 'search_screenpipe' tool to query the OCR logs.

        CUSTOM USER INSTRUCTIONS:
        {custom_inst if custom_inst else "None."}
        """

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def capture_screen_compressed(self):
        """Captura el monitor principal y lo comprime agresivamente para ahorrar tokens/costos."""
        print("📸 [VISIÓN] Capturando pantalla principal...")
        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                
                # Compresión a 720p (Suficiente para el OCR interno de Claude)
                target_width = 1280
                if img.size[0] > target_width:
                    ratio = target_width / float(img.size[0])
                    target_height = int(float(img.size[1]) * ratio)
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                # Calidad 60 para ultra-compresión de JPEG
                img.save(buffer, format="JPEG", quality=60)
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return img_b64
        except Exception as e:
            print(f"❌ [VISIÓN] Error al capturar: {e}")
            return None

    def search_screenpipe_local(self, query, limit=5):
        """Hace un request rápido al daemon local de Screenpipe."""
        print(f"🕵️‍♂️ [SCREENPIPE] Buscando en el pasado: '{query}'")
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
        except Exception as e:
            print(f"⚠️ [SCREENPIPE] Timeout o apagado: {e}")
            return "Error de conexión con Screenpipe local."

    def _trim_history(self):
        """
        Mantiene el límite de historial y ELIMINA las imágenes de turnos anteriores
        para no saturar el payload de la API ni disparar los costos.
        """
        # 1. Cortar si nos pasamos del límite
        if len(self.message_history) > self.MAX_HISTORY:
            self.message_history = self.message_history[-self.MAX_HISTORY:]
            
        # 2. Remover imágenes viejas (solo dejamos la del último turno)
        # Iteramos hasta el anteúltimo elemento
        for i in range(len(self.message_history) - 1):
            msg = self.message_history[i]
            if msg["role"] == "user" and isinstance(msg["content"], list):
                # Filtramos y nos quedamos solo con los bloques de texto
                text_only = [c for c in msg["content"] if c.get("type") == "text" or c.get("type") == "tool_result"]
                msg["content"] = text_only

    def process_query_async(self, user_prompt):
        api_key = self.config.get("api_key", "").strip()
        if not api_key:
            self.error_occurred.emit("ERR_NO_API")
            return
            
        threading.Thread(target=self._run_llm_chain, args=(user_prompt,), daemon=True).start()

    def _run_llm_chain(self, user_prompt):
        self.thinking_started.emit()
        img_b64 = self.capture_screen_compressed()
        
        # Armamos el input actual
        content_array = []
        if img_b64:
            content_array.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            })
        content_array.append({"type": "text", "text": user_prompt})

        # Agregamos al historial y lo podamos
        self.message_history.append({"role": "user", "content": content_array})
        self._trim_history()

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
            print("🚀 [BRAIN] Mandando request a Haiku 4.5...")
            
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022", # Usando Haiku 3.5 (el más rápido y barato actual)
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
                    model="claude-3-5-haiku-20241022",
                    max_tokens=500,
                    system=self.system_prompt,
                    messages=self.message_history,
                    stream=True
                )
            else:
                stream = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",
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
            print(f"❌ [BRAIN] Crash en la API: {e}")
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
                        print(f"🎯 [BRAIN] Coordenadas detectadas: X={x}, Y={y}")
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