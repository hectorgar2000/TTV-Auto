"""Scriptwriter agent: writes the narration script split into scenes."""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from config import Config
    from state import VideoState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Eres un guionista experto en vídeos de YouTube sobre lore y fantasía.
Escribes narraciones épicas en español de España, diseñadas para ENGANCHAR.

ESTILO DE ESCRITURA:
- Frases CORTAS y PUNZANTES. Máximo 10-12 palabras por frase.
- Ritmo rápido. Nada de párrafos largos ni explicaciones farragosas.
- Ganchos constantes: preguntas retóricas, cliffhangers entre escenas.
- Callbacks: referencia algo del principio al final para cerrar el círculo.
- Abre con un gancho brutal que haga imposible irse.
- Cierra con una frase memorable que deje huella.
- Transiciones con tensión: "Pero eso fue solo el principio."
- Usa pausas dramáticas (puntos suspensivos, frases de una palabra).

ESTRUCTURA:
- Empieza con un HOOK de 15-20 segundos que enganche de inmediato.
- Cada escena tiene un mini-cliffhanger o revelación.
- Alterna ritmo: momento tenso → pausa reflexiva → acción.

INSTRUCCIONES:
1. Escribe un guion de 1.100 a 1.900 palabras (8-15 minutos).
2. Divide en 8-14 escenas.
3. Cada escena: 60 a 180 palabras de narración.
4. Estima duración en segundos (1 palabra ≈ 0.55s).
5. Para cada escena, incluye un "impact_text": una frase corta e impactante
   (1-5 palabras, EN MAYÚSCULAS) que se mostrará en grande sobre la imagen
   en el momento clave. Ejemplos: "LA TRAICIÓN", "NO HUBO PIEDAD",
   "EL ÚLTIMO ALIENTO". Si la escena no tiene momento de impacto, deja "".

Responde EXCLUSIVAMENTE con JSON válido:
{
  "script": "Texto completo del guion",
  "scenes": [
    {
      "index": 0,
      "narration_text": "Texto de narración...",
      "impact_text": "FRASE IMPACTANTE",
      "target_duration": 65.0
    }
  ]
}
"""

MAX_RETRIES = 2


def create_scriptwriter_node(config: Config):
    llm = ChatOllama(
        model=config.llm_model,
        temperature=0.8,
        base_url=config.ollama_base_url,
        num_predict=8192,
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
        if "impact_text" not in scene:
            scene["impact_text"] = ""

    return data
