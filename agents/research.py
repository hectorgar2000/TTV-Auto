"""Research agent: investigates the topic using web search."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from config import Config
    from state import VideoState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Eres un investigador experto en universos de fantasía, lore y worldbuilding.
Tu trabajo es investigar a fondo el tema proporcionado, recopilando:
- Hechos canónicos del universo (eventos, fechas, personajes)
- Contexto narrativo (qué pasó antes y después)
- Detalles visuales útiles para ilustrar la historia (lugares, vestimenta, objetos)
- Fuentes consultadas (wikis oficiales, libros, etc.)

IMPORTANTE:
- Escribe siempre en español (España).
- Distingue entre lore canónico (libros originales) y adaptaciones (películas, juegos).
- Si encuentras información contradictoria entre fuentes, señálalo.
- Produce notas detalladas y organizadas por secciones.
- Al final, lista todas las fuentes consultadas.
"""

MAX_RETRIES = 2


def create_research_node(config: Config):
    llm = ChatAnthropic(
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=4096,
    )
    search_tool = DuckDuckGoSearchResults(
        max_results=5,
        output_format="list",
    )
    agent = create_react_agent(llm, [search_tool], prompt=SYSTEM_PROMPT)

    def research_node(state: VideoState) -> dict:
        topic = state.get("topic", "")
        logger.info("🔍 Investigando: %s", topic)

        user_msg = (
            f"Investiga a fondo el siguiente tema para un vídeo de YouTube "
            f"sobre lore de fantasía:\n\n**{topic}**\n\n"
            f"Busca en wikis especializadas (ej: tolkiengateway.net, "
            f"lotr.fandom.com) y fuentes fiables. "
            f"Produce notas de investigación completas en español."
        )

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = agent.invoke(
                    {"messages": [HumanMessage(content=user_msg)]}
                )
                final_content = result["messages"][-1].content
                sources = _extract_sources(final_content)
                logger.info(
                    "✅ Investigación completada (%d fuentes)", len(sources)
                )
                return {
                    "research_notes": final_content,
                    "sources": sources,
                }
            except Exception as e:
                last_error = e
                logger.warning(
                    "Intento %d/%d fallido: %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    e,
                )

        logger.error("❌ Investigación fallida tras %d intentos", MAX_RETRIES + 1)
        return {
            "research_notes": f"[ERROR] Investigación fallida: {last_error}",
            "sources": [],
            "error_log": [f"research_agent: {last_error}"],
        }

    return research_node


def _extract_sources(text: str) -> list[str]:
    """Extract URLs and source references from research text."""
    import re

    urls = re.findall(r"https?://[^\s\)]+", text)
    return list(dict.fromkeys(urls))
