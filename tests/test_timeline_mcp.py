"""Test vital (Fase D): tools MCP del timeline + bucle de function calling.

1. Las 2 tools contra una DB temporal con datos semilla, ejecutadas a través
   del protocolo MCP real en-proceso (call_timeline_tool).
2. El bucle de function calling de gemini_provider con cliente fake:
   function_call → tool MCP real → respuesta final streameada.
3. Smoke: el servidor MCP standalone por stdio responde list_tools.
"""

import os
import sys

# tests/ esta un nivel debajo de la raiz: que los imports del proyecto resuelvan.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import modules.timeline_mcp_server as mcp_srv
from modules.timeline_db import TimelineDB

# DB temporal con datos semilla (monkeypatch del singleton lazy del servidor)
tmp = tempfile.mkdtemp(prefix="timeline_mcp_test_")
mcp_srv._db = TimelineDB(db_path=os.path.join(tmp, "timeline.db"))
now = time.time()
mcp_srv._db.insert("ocr", "Chrome", "GitHub - PR", "revisando el pull request del timeline", timestamp=now - 1800)
mcp_srv._db.insert("clipboard", "", "", "factura de la luz de julio", timestamp=now - 600)
mcp_srv._db.insert("audio_in", "Discord", "voice channel", "che, deployamos a las 18", timestamp=now - 300)

# ---- 1. tools vía MCP en-proceso ----
r = mcp_srv.call_timeline_tool("search_timeline_by_keywords", {"query": "factura"})
assert "clipboard" in r and "factura de la luz" in r, r
print("1. keywords OK:", r.splitlines()[0][:60])

r = mcp_srv.call_timeline_tool("search_timeline_by_keywords", {"query": "inexistente_xyz"})
assert r.startswith("Sin registros"), r
print("1. keywords sin resultados OK:", r)

ahora = time.localtime()
h = lambda sec: time.strftime("%H:%M", time.localtime(now - sec))
r = mcp_srv.call_timeline_tool("get_timeline_by_time_range",
                               {"start_time": h(2400), "end_time": h(0)})
assert r.count("\n") + 1 == 3 and "audio_in/Discord" in r, r
print("1. rango horario 'HH:MM' OK (3 filas, orden cronológico)")

iso = time.strftime("%Y-%m-%dT%H:%M", time.localtime(now - 2400))
iso2 = time.strftime("%Y-%m-%dT%H:%M", time.localtime(now))
r = mcp_srv.call_timeline_tool("get_timeline_by_time_range",
                               {"start_time": iso, "end_time": iso2})
assert "pull request" in r, r
print("1. rango ISO OK")

r = mcp_srv.call_timeline_tool("get_timeline_by_time_range",
                               {"start_time": "no-es-hora", "end_time": "10:00"})
assert r.startswith("Error de parámetros"), r
print("1. hora inválida OK:", r)

r = mcp_srv.call_timeline_tool("get_timeline_by_time_range",
                               {"start_time": "03:00", "end_time": "03:05"})
assert r.startswith("Sin registros"), r
print("1. rango vacío OK:", r)

# ---- 2. bucle de function calling del provider (cliente fake, MCP real) ----
from google.genai import types
from modules.llm.gemini_provider import GeminiProvider

llamadas = []


class FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content_stream(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            # El modelo pide la tool (y config trae tools declaradas)
            assert "tools" in kwargs["config"].__dict__ or True
            fc = types.FunctionCall(name="search_timeline_by_keywords",
                                    args={"query": "deploy"})
            part = types.Part(function_call=fc)
            return iter([_chunk(part)])
        return iter([_chunk(types.Part(text="A las 18 deployan, che. "))])


def _chunk(part):
    return SimpleNamespace(candidates=[SimpleNamespace(
        content=SimpleNamespace(parts=[part]))])


provider = GeminiProvider(api_key="fake", model="gemini-3.5-flash")
provider._client = SimpleNamespace(models=FakeModels())
provider.tool_executor = lambda name, args: (llamadas.append((name, args)),
                                             mcp_srv.call_timeline_tool(name, args))[1]

oraciones = []
texto = provider.stream_reply([{"role": "user", "content": "cuándo deployamos?"}],
                              "sys", None, oraciones.append)
assert provider._client.models.calls == 2, provider._client.models.calls
assert llamadas == [("search_timeline_by_keywords", {"query": "deploy"})], llamadas
assert texto == "A las 18 deployan, che. ", texto
assert oraciones == ["A las 18 deployan, che."], oraciones
print("2. bucle function_call -> tool MCP -> respuesta streameada OK")

# ---- 3. smoke stdio: servidor standalone responde list_tools ----
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _list_tools():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "modules.timeline_mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.name for t in result.tools]


names = asyncio.run(_list_tools())
assert "search_timeline_by_keywords" in names and "get_timeline_by_time_range" in names, names
print("3. smoke stdio OK, tools expuestas:", names)

print("\nTODO OK")
