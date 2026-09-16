"""CLI entry point for the video production pipeline."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from config import Config
from state import VideoState

_FFMPEG_DLL_DIR = Path(__file__).resolve().parent / "lib" / "ffmpeg"
if _FFMPEG_DLL_DIR.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_FFMPEG_DLL_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Genera vídeos de YouTube sobre lore de fantasía.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            '  python main.py --tema "La caída de Gondolin"\n'
            '  python main.py --tema "La historia de Isildur" --review\n'
            '  python main.py --tema "Los Balrogs de Morgoth" --config config.yaml\n'
        ),
    )
    parser.add_argument(
        "--tema",
        required=True,
        help="Tema del vídeo (ej: 'La caída de Gondolin')",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Pausar antes del montaje para revisión manual (human-in-the-loop)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta a config.yaml (opcional)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar un pipeline interrumpido para el mismo tema",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Logging detallado (DEBUG)",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)
    logger = logging.getLogger("main")

    config = Config.load(args.config)
    logger.info("Configuración cargada: LLM=%s, TTS=%s, SD=%s",
                config.llm_model, config.tts_engine, config.sd_model)

    if not _check_ollama(config):
        logger.error(
            "Ollama no está accesible en %s. "
            "Asegúrate de que Ollama está instalado y ejecutándose:\n"
            "  1. Descarga: https://ollama.com/download\n"
            "  2. Ejecuta: ollama serve\n"
            "  3. Descarga modelo: ollama pull %s",
            config.ollama_base_url,
            config.llm_model,
        )
        sys.exit(1)

    from graph import build_graph

    logger.info("Construyendo grafo...")
    compiled_graph, checkpointer = build_graph(
        config, interrupt_before_assembly=args.review
    )

    thread_id = _slugify(args.tema)
    graph_config = {"configurable": {"thread_id": thread_id}}

    initial_state: VideoState = {
        "topic": args.tema,
        "review_requested": args.review,
        "sources": [],
        "scenes": [],
        "error_log": [],
    }

    logger.info("=" * 60)
    logger.info("🎬 Iniciando pipeline: '%s'", args.tema)
    logger.info("   Thread ID: %s", thread_id)
    logger.info("   Review mode: %s", args.review)
    logger.info("=" * 60)

    start_time = time.time()

    try:
        if args.resume:
            logger.info("Reanudando pipeline desde checkpoint...")
            state = compiled_graph.get_state(graph_config)
            if state and state.values:
                result = compiled_graph.invoke(None, config=graph_config)
            else:
                logger.warning("No hay checkpoint previo, iniciando desde cero")
                result = compiled_graph.invoke(initial_state, config=graph_config)
        else:
            result = compiled_graph.invoke(initial_state, config=graph_config)

    except KeyboardInterrupt:
        logger.info("\n⏸️ Pipeline interrumpido. Usa --resume para continuar.")
        sys.exit(0)

    elapsed = time.time() - start_time

    if result is None:
        if args.review:
            logger.info(
                "\n⏸️ Pipeline pausado para revisión (human-in-the-loop).\n"
                "Revisa las imágenes y el guion en output/%s/\n"
                "Cuando estés listo, ejecuta de nuevo con --resume:\n"
                '  python main.py --tema "%s" --resume',
                thread_id,
                args.tema,
            )
            sys.exit(0)
        logger.error("Pipeline devolvió None sin review mode")
        sys.exit(1)

    _print_summary(result, elapsed, logger)


def _print_summary(result: dict, elapsed: float, logger: logging.Logger):
    video_path = result.get("video_path", "")
    metadata = result.get("metadata", {})
    review = result.get("review_notes", "")
    errors = result.get("error_log", [])

    logger.info("\n" + "=" * 60)
    logger.info("🎬 PIPELINE COMPLETADO en %.1f minutos", elapsed / 60)
    logger.info("=" * 60)

    if video_path:
        logger.info("📹 Vídeo: %s", video_path)
    if metadata:
        logger.info("📌 Título: %s", metadata.get("title", ""))
        logger.info("📝 Tags: %s", ", ".join(metadata.get("tags", [])[:10]))
    if review:
        logger.info("\n%s", review)
    if errors:
        logger.warning("\n⚠️ Errores registrados:")
        for err in errors:
            logger.warning("  - %s", err)


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _check_ollama(config) -> bool:
    """Check that Ollama is running and the model is available."""
    import urllib.request
    import urllib.error
    try:
        url = f"{config.ollama_base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            base_name = config.llm_model.split(":")[0]
            found = any(base_name in m for m in models)
            if not found:
                logging.getLogger("main").warning(
                    "Modelo '%s' no encontrado en Ollama. "
                    "Modelos disponibles: %s. "
                    "Ejecuta: ollama pull %s",
                    config.llm_model,
                    ", ".join(models) or "(ninguno)",
                    config.llm_model,
                )
            return True
    except (urllib.error.URLError, OSError):
        return False


def _slugify(text: str) -> str:
    import re
    import unicodedata
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")


if __name__ == "__main__":
    main()
