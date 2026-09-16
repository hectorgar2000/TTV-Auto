"""Storyboard agent: generates image prompts and Wikimedia search terms per scene."""
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
Eres un director artístico para vídeos de YouTube sobre lore de fantasía.
Tu trabajo es, para cada escena del guion, generar:

1. Un prompt de imagen EN INGLÉS para Stable Diffusion, estilo:
   "epic fantasy illustration, [descripción detallada], dramatic lighting,
    high detail, cinematic composition, 16:9 aspect ratio"
   Sé específico con vestimenta, paisajes, colores y atmósfera.

2. Términos de búsqueda para Wikimedia Commons (EN INGLÉS) por si existe
   una ilustración, mapa o grabado de dominio público. Esto es útil para
   mapas de mundos ficticios, portadas de libros en dominio público, o
   arte fan con licencia libre. Si la escena es muy específica del lore
   y no es probable encontrar imagen real, deja la lista vacía.

Responde EXCLUSIVAMENTE con un JSON válido:
{
  "scenes": [
    {
      "index": 0,
      "image_prompt_en": "epic fantasy illustration of...",
      "wikimedia_search_terms": ["term1", "term2"]
    }
  ]
}

Genera una entrada por cada escena del guion, manteniendo el mismo índice.
"""

MAX_RETRIES = 2


def create_storyboard_node(config: Config):
    llm = ChatAnthropic(
        model=config.llm_model,
        temperature=0.7,
        max_tokens=4096,
    )

    def storyboard_node(state: VideoState) -> dict:
        logger.info("🎨 Generando storyboard...")
        scenes = state.get("scenes", [])
        topic = state.get("topic", "")

        if not scenes:
            logger.warning("Sin escenas, saltando storyboard")
            return {"error_log": ["storyboard_agent: no scenes to process"]}

        scenes_summary = "\n\n".join(
            f"Escena {s['index']}:\n{s.get('narration_text', '')}"
            for s in scenes
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Tema: **{topic}**\n\n"
                    f"Escenas del guion ({len(scenes)} escenas):\n\n"
                    f"{scenes_summary}\n\n"
                    f"Genera los prompts de imagen y términos de búsqueda "
                    f"para cada escena. JSON:"
                )
            ),
        ]

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = llm.invoke(messages)
                parsed = _parse_storyboard(response.content, len(scenes))
                updated_scenes = _merge_storyboard(scenes, parsed)
                logger.info("✅ Storyboard: %d escenas procesadas", len(updated_scenes))
                return {"scenes": updated_scenes}
            except Exception as e:
                last_error = e
                logger.warning("Intento %d/%d: %s", attempt + 1, MAX_RETRIES + 1, e)

        logger.error("❌ Storyboard fallido")
        return {"error_log": [f"storyboard_agent: {last_error}"]}

    return storyboard_node


def _parse_storyboard(content: str, expected_count: int) -> list[dict]:
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

    storyboard_scenes = data.get("scenes", [])
    if len(storyboard_scenes) < expected_count:
        logger.warning(
            "Storyboard devolvió %d escenas, esperaba %d",
            len(storyboard_scenes),
            expected_count,
        )
    return storyboard_scenes


def _merge_storyboard(scenes: list[dict], storyboard: list[dict]) -> list[dict]:
    """Merge storyboard prompts into existing scenes by index."""
    result = [dict(s) for s in scenes]
    for sb in storyboard:
        idx = sb.get("index")
        if idx is not None and 0 <= idx < len(result):
            result[idx]["image_prompt_en"] = sb.get("image_prompt_en", "")
            result[idx]["wikimedia_search_terms"] = sb.get(
                "wikimedia_search_terms", []
            )
    return result
