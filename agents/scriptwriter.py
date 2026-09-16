"""Scriptwriter agent: writes the narration script split into scenes."""
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
Eres un guionista experto en vídeos de YouTube sobre lore y fantasía.
Escribes narraciones épicas, envolventes y accesibles en español de España.

Tu estilo:
- Narración en tercera persona, tono de cronista/bardo.
- Frases claras, ritmo pausado (para narración en voz en off).
- Transiciones suaves entre escenas ("Pero la oscuridad no descansaba...").
- Ganchos al inicio para retener al espectador.
- Cierre memorable que invite a ver más vídeos.

INSTRUCCIONES:
1. Escribe un guion de 1.100 a 1.900 palabras (8-15 minutos a ritmo pausado).
2. Divide el guion en 6-12 escenas numeradas.
3. Cada escena debe tener entre 80 y 200 palabras de narración.
4. Estima la duración en segundos de cada escena (aprox. 1 palabra = 0.55s).

Responde EXCLUSIVAMENTE con un JSON válido con esta estructura:
{
  "script": "Texto completo del guion (todas las escenas juntas)",
  "scenes": [
    {
      "index": 0,
      "narration_text": "Texto de narración de esta escena...",
      "target_duration": 65.0
    }
  ]
}
"""

MAX_RETRIES = 2


def create_scriptwriter_node(config: Config):
    llm = ChatAnthropic(
        model=config.llm_model,
        temperature=0.8,
        max_tokens=8192,
    )

    def scriptwriter_node(state: VideoState) -> dict:
        logger.info("✍️ Escribiendo guion...")
        topic = state.get("topic", "")
        research = state.get("research_notes", "")
        fact_notes = state.get("fact_check_notes", "")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Tema: **{topic}**\n\n"
                    f"Investigación verificada:\n{research}\n\n"
                    f"Notas de verificación:\n{fact_notes}\n\n"
                    f"Escribe el guion completo en JSON."
                )
            ),
        ]

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = llm.invoke(messages)
                parsed = _parse_script_response(response.content)
                scene_count = len(parsed["scenes"])
                word_count = len(parsed["script"].split())
                logger.info(
                    "✅ Guion: %d escenas, %d palabras", scene_count, word_count
                )
                return {
                    "script": parsed["script"],
                    "scenes": parsed["scenes"],
                }
            except Exception as e:
                last_error = e
                logger.warning("Intento %d/%d: %s", attempt + 1, MAX_RETRIES + 1, e)

        logger.error("❌ Escritura de guion fallida")
        return {
            "script": "",
            "scenes": [],
            "error_log": [f"scriptwriter_agent: {last_error}"],
        }

    return scriptwriter_node


def _parse_script_response(content: str) -> dict:
    """Parse JSON from the LLM response, handling markdown fences and preamble text."""
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

    if "script" not in data or "scenes" not in data:
        raise ValueError("Response JSON missing 'script' or 'scenes' keys")

    for i, scene in enumerate(data["scenes"]):
        scene["index"] = i
        if "target_duration" not in scene:
            words = len(scene.get("narration_text", "").split())
            scene["target_duration"] = words * 0.55

    return data
