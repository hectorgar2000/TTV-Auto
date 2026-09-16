"""Fact-check agent: reviews research for lore accuracy."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from config import Config
    from state import VideoState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Eres un verificador experto en lore de universos de fantasía.
Tu trabajo es revisar las notas de investigación y:

1. Identificar afirmaciones que puedan ser incorrectas o no canónicas.
2. Señalar si algo es de una adaptación (película/serie/juego) vs. la obra original.
3. Marcar datos sin fuente sólida con [SIN FUENTE].
4. Eliminar especulaciones presentadas como hechos.
5. Verificar coherencia interna (fechas, nombres, eventos).

Produce un informe en español con:
- ✅ Hechos verificados (fiables para el guion)
- ⚠️ Hechos dudosos (reformular o eliminar)
- ❌ Errores detectados (no usar en el guion)
- Notas de investigación corregidas y limpias al final.
"""

MAX_RETRIES = 2


def create_fact_check_node(config: Config):
    llm = ChatAnthropic(
        model=config.llm_model,
        temperature=0.3,
        max_tokens=4096,
    )

    def fact_check_node(state: VideoState) -> dict:
        logger.info("🔎 Verificando investigación...")
        research = state.get("research_notes", "")
        topic = state.get("topic", "")

        if not research:
            logger.warning("Sin notas de investigación, saltando verificación")
            return {"fact_check_notes": "[SKIP] No hay investigación que verificar."}

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Tema del vídeo: **{topic}**\n\n"
                    f"Notas de investigación a verificar:\n\n{research}"
                )
            ),
        ]

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = llm.invoke(messages)
                logger.info("✅ Verificación completada")
                return {"fact_check_notes": response.content}
            except Exception as e:
                last_error = e
                logger.warning("Intento %d/%d: %s", attempt + 1, MAX_RETRIES + 1, e)

        logger.error("❌ Verificación fallida")
        return {
            "fact_check_notes": f"[ERROR] Verificación fallida: {last_error}",
            "error_log": [f"fact_check_agent: {last_error}"],
        }

    return fact_check_node
