"""LangGraph StateGraph: orchestrates the full video production pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from state import VideoState

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


def build_graph(config: Config, interrupt_before_assembly: bool = False):
    """Build and compile the video production StateGraph."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    from agents.fact_check import create_fact_check_node
    from agents.metadata import create_metadata_node
    from agents.research import create_research_node
    from agents.scriptwriter import create_scriptwriter_node
    from agents.storyboard import create_storyboard_node

    research_node = create_research_node(config)
    fact_check_node = create_fact_check_node(config)
    scriptwriter_node = create_scriptwriter_node(config)
    storyboard_node = create_storyboard_node(config)
    metadata_node = create_metadata_node(config)

    graph = StateGraph(VideoState)

    # --- Sequential LLM agent nodes ---
    graph.add_node("research", _timed("research", research_node))
    graph.add_node("fact_check", _timed("fact_check", fact_check_node))
    graph.add_node("scriptwriter", _timed("scriptwriter", scriptwriter_node))
    graph.add_node("storyboard", _timed("storyboard", storyboard_node))

    # --- Parallel per-scene nodes ---
    graph.add_node("generate_voice", _timed("generate_voice", _make_voice_node(config)))
    graph.add_node("generate_image", _timed("generate_image", _make_image_node(config)))

    # --- Sequential post-processing ---
    graph.add_node("generate_subtitles", _timed("subtitles", _make_subtitle_node()))
    graph.add_node("video_assembly", _timed("video_assembly", _make_assembly_node(config)))
    graph.add_node("metadata", _timed("metadata", metadata_node))
    graph.add_node("qa", _timed("qa", _make_qa_node()))

    # --- Edges: sequential pipeline ---
    graph.set_entry_point("research")
    graph.add_edge("research", "fact_check")
    graph.add_edge("fact_check", "scriptwriter")
    graph.add_edge("scriptwriter", "storyboard")

    # --- Fan-out: storyboard -> voice per scene ---
    graph.add_conditional_edges(
        "storyboard",
        _dispatch_voice,
        ["generate_voice"],
    )

    # --- Fan-out: after all voice -> image per scene ---
    graph.add_conditional_edges(
        "generate_voice",
        _dispatch_image,
        ["generate_image"],
    )

    # --- After all images -> subtitles (sequential, fast) ---
    graph.add_edge("generate_image", "generate_subtitles")

    # --- Subtitles -> assembly -> metadata -> qa ---
    graph.add_edge("generate_subtitles", "video_assembly")
    graph.add_edge("video_assembly", "metadata")
    graph.add_edge("metadata", "qa")
    graph.add_edge("qa", END)

    # --- Compile with checkpointer ---
    interrupt = ["video_assembly"] if interrupt_before_assembly else []

    import sqlite3
    conn = sqlite3.connect(str(config.db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt,
    )

    return compiled, checkpointer


# --- Send dispatchers ---

def _send_payload(state: VideoState, scene_index: int) -> dict:
    """Build a minimal Send payload — only fields the per-scene nodes need."""
    return {
        "current_scene_index": scene_index,
        "topic": state.get("topic", ""),
        "scenes": state.get("scenes", []),
        "error_log": [],
        "sources": [],
    }


def _dispatch_voice(state: VideoState) -> list[Send]:
    """Fan out: one Send per scene for voice generation."""
    scenes = state.get("scenes", [])
    sends = []
    for i, scene in enumerate(scenes):
        if not scene.get("audio_path"):
            sends.append(Send("generate_voice", _send_payload(state, i)))
    if not sends:
        sends.append(Send("generate_voice", _send_payload(state, 0)))
    return sends


def _dispatch_image(state: VideoState) -> list[Send]:
    """Fan out: one Send per scene for image generation."""
    scenes = state.get("scenes", [])
    sends = []
    for i, scene in enumerate(scenes):
        if not scene.get("image_path"):
            sends.append(Send("generate_image", _send_payload(state, i)))
    if not sends:
        sends.append(Send("generate_image", _send_payload(state, 0)))
    return sends


# --- Per-scene processing nodes ---

def _make_voice_node(config: Config):
    def voice_node(state: VideoState) -> dict:
        from tools.voice import generate_voice, get_audio_duration

        idx = state.get("current_scene_index", 0)
        scenes = state.get("scenes", [])
        if idx >= len(scenes):
            return {}

        scene = scenes[idx]
        text = scene.get("narration_text", "")
        if not text:
            return {"error_log": [f"scene {idx}: no narration text"]}

        topic_slug = _slugify(state.get("topic", "video"))
        out_dir = config.output_dir / topic_slug / "audio"
        out_path = out_dir / f"scene_{idx:03d}.wav"

        try:
            audio_path = generate_voice(text, out_path, config)
            duration = get_audio_duration(audio_path)
            return {
                "scenes": [{
                    "index": idx,
                    "audio_path": str(audio_path),
                    "duration_real": duration,
                }],
            }
        except Exception as e:
            logger.error("Error generando voz escena %d: %s", idx, e)
            return {"error_log": [f"voice scene {idx}: {e}"]}

    return voice_node


def _make_image_node(config: Config):
    def image_node(state: VideoState) -> dict:
        from tools.images import create_placeholder_image, generate_image

        idx = state.get("current_scene_index", 0)
        scenes = state.get("scenes", [])
        if idx >= len(scenes):
            return {}

        scene = scenes[idx]
        prompt = scene.get("image_prompt_en", "")
        wiki_terms = scene.get("wikimedia_search_terms", [])
        wiki_url = scene.get("wikimedia_image_url", "")

        topic_slug = _slugify(state.get("topic", "video"))
        out_dir = config.output_dir / topic_slug / "images"
        out_path = out_dir / f"scene_{idx:03d}.png"

        try:
            image_path = generate_image(
                prompt_en=prompt,
                wikimedia_terms=wiki_terms,
                output_path=out_path,
                config=config,
                wikimedia_url=wiki_url,
            )
            return {"scenes": [{"index": idx, "image_path": str(image_path)}]}
        except Exception as e:
            logger.error("Error generando imagen escena %d: %s", idx, e)
            try:
                placeholder = create_placeholder_image(out_path, prompt[:60])
                return {
                    "scenes": [{"index": idx, "image_path": str(placeholder)}],
                    "error_log": [f"image scene {idx}: used placeholder ({e})"],
                }
            except Exception:
                return {"error_log": [f"image scene {idx}: {e}"]}

    return image_node


def _make_subtitle_node():
    def subtitle_node(state: VideoState) -> dict:
        from tools.subtitles import generate_subtitles_for_scene

        scenes = state.get("scenes", [])
        offset = 0.0
        updated = []

        for scene in scenes:
            idx = scene.get("index", 0)
            text = scene.get("narration_text", "")
            duration = scene.get("duration_real", 0)

            if not text or not duration:
                offset += duration
                continue

            entries = generate_subtitles_for_scene(text, duration, offset)
            updated.append({
                "index": idx,
                "subtitle_entries": entries,
            })
            offset += duration

        logger.info("✅ Subtítulos generados para %d escenas", len(updated))
        return {"scenes": updated}

    return subtitle_node


def _make_assembly_node(config: Config):
    def assembly_node(state: VideoState) -> dict:
        from tools.subtitles import generate_srt_file
        from tools.video import assemble_video

        scenes = state.get("scenes", [])
        topic_slug = _slugify(state.get("topic", "video"))
        out_dir = config.output_dir / topic_slug

        all_subs = []
        for scene in scenes:
            all_subs.extend(scene.get("subtitle_entries", []))

        if all_subs:
            srt_path = out_dir / "subtitles.srt"
            generate_srt_file(all_subs, srt_path)

        video_path = out_dir / "video_final.mp4"
        try:
            result = assemble_video(scenes, video_path, config)
            return {"video_path": str(result)}
        except Exception as e:
            logger.error("Error montando vídeo: %s", e)
            return {"error_log": [f"video_assembly: {e}"]}

    return assembly_node


def _make_qa_node():
    def qa_node(state: VideoState) -> dict:
        from tools.voice import get_audio_duration

        scenes = state.get("scenes", [])
        errors = []
        total_duration = 0.0

        for scene in scenes:
            idx = scene.get("index", 0)
            if not scene.get("audio_path"):
                errors.append(f"Escena {idx}: sin audio")
            if not scene.get("image_path"):
                errors.append(f"Escena {idx}: sin imagen")
            if not scene.get("subtitle_entries"):
                errors.append(f"Escena {idx}: sin subtítulos")
            total_duration += scene.get("duration_real", 0)

        duration_min = total_duration / 60
        video_path = state.get("video_path", "")

        if duration_min < 8:
            errors.append(
                f"⚠️ Duración ({duration_min:.1f} min) menor a 8 minutos"
            )
        elif duration_min > 15:
            errors.append(
                f"⚠️ Duración ({duration_min:.1f} min) mayor a 15 minutos"
            )

        status = "✅ APROBADO" if not errors else "⚠️ CON OBSERVACIONES"
        review = (
            f"=== QA Report ===\n"
            f"Estado: {status}\n"
            f"Escenas: {len(scenes)}\n"
            f"Duración total: {duration_min:.1f} minutos\n"
            f"Vídeo: {video_path}\n"
        )
        if errors:
            review += "\nObservaciones:\n" + "\n".join(f"  - {e}" for e in errors)

        logger.info("\n%s", review)
        return {"review_notes": review}

    return qa_node


# --- Helpers ---

def _timed(name: str, fn):
    """Wrap a node function with timing logs."""
    def wrapper(state):
        logger.info("▶️  Nodo '%s' iniciado", name)
        start = time.time()
        try:
            result = fn(state)
            elapsed = time.time() - start
            logger.info("⏱️  Nodo '%s' completado en %.1fs", name, elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error("💥 Nodo '%s' falló en %.1fs: %s", name, elapsed, e)
            raise
    return wrapper


def _slugify(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")
