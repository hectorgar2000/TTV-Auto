"""Metadata agent: generates YouTube title, description and tags."""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from config import Config
    from state import VideoState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Eres un experto en SEO y optimización de vídeos de YouTube en español.
El canal se centra en lore de universos de fantasía (Tolkien, etc.).
La audiencia es hispanohablante de España.

Genera metadatos optimizados para el vídeo:

1. **Título**: máx. 60 caracteres, gancho emocional, incluir keyword principal.
   Usa fórmulas tipo: "La VERDADERA historia de X", "Lo que NO sabías de X",
   "El MISTERIO de X explicado".

2. **Descripción**: 150-300 palabras. Empieza con un párrafo gancho (aparece
   en búsquedas), luego resumen del contenido, timestamps si aplica, y CTA
   para suscribirse. Incluye hashtags relevantes al final.

3. **Tags**: 15-25 tags en español, mezcla de:
   - Genéricos del nicho (lore, fantasía, Tolkien)
   - Específicos del tema (nombres, lugares, eventos)
   - Long-tail (preguntas que la gente buscaría)

Responde EXCLUSIVAMENTE con JSON válido:
{
  "title": "...",
  "description": "...",
  "tags": ["tag1", "tag2", ...]
}
"""

MAX_RETRIES = 2


def create_metadata_node(config: Config):
    llm = ChatAnthropic(
        model=config.llm_model,
        temperature=0.7,
        max_tokens=2048,
    )

    def metadata_node(state: VideoState) -> dict:
        logger.info("📝 Generando metadatos YouTube...")
        topic = state.get("topic", "")
        script = state.get("script", "")

        script_preview = script[:2000] if script else "[sin guion]"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Tema del vídeo: **{topic}**\n\n"
                    f"Guion (extracto):\n{script_preview}\n\n"
                    f"Genera título, descripción y tags en JSON."
                )
            ),
        ]

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = llm.invoke(messages)
                parsed = _parse_metadata(response.content)
                logger.info("✅ Metadatos: '%s'", parsed["title"][:50])
                return {"metadata": parsed}
            except Exception as e:
                last_error = e
                logger.warning("Intento %d/%d: %s", attempt + 1, MAX_RETRIES + 1, e)

        logger.error("❌ Generación de metadatos fallida")
        return {
            "metadata": {"title": topic, "description": "", "tags": []},
            "error_log": [f"metadata_agent: {last_error}"],
        }

    return metadata_node


def _parse_metadata(content: str) -> dict:
    cleaned = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        brace = cleaned.find("{")
        if brace > 0:
            cleaned = cleaned[brace:]
    data = json.loads(cleaned)
    return {
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "tags": data.get("tags", []),
    }
