"""Subtitle generation: builds SRT entries from script + audio durations."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state import SubtitleEntry

logger = logging.getLogger(__name__)

MAX_CHARS_PER_LINE = 35
MAX_SUBTITLE_DURATION = 4.5  # seconds


def generate_subtitles_for_scene(
    narration_text: str,
    audio_duration: float,
    scene_offset: float,
) -> list[SubtitleEntry]:
    """Split narration into subtitle entries timed to audio duration.

    Breaks text into phrases at sentence boundaries, distributes
    timing proportionally by word count.
    """
    phrases = _split_into_phrases(narration_text)
    if not phrases:
        return []

    total_words = sum(len(p.split()) for p in phrases)
    if total_words == 0:
        return []

    durations = []
    for phrase in phrases:
        word_count = len(phrase.split())
        durations.append((word_count / total_words) * audio_duration)

    for i, d in enumerate(durations):
        if d > MAX_SUBTITLE_DURATION:
            excess = d - MAX_SUBTITLE_DURATION
            durations[i] = MAX_SUBTITLE_DURATION
            uncapped = [j for j in range(len(durations)) if j != i and durations[j] < MAX_SUBTITLE_DURATION]
            if uncapped:
                share = excess / len(uncapped)
                for j in uncapped:
                    durations[j] += share

    entries: list[SubtitleEntry] = []
    current_time = scene_offset

    for idx, (phrase, duration) in enumerate(zip(phrases, durations)):
        lines = _wrap_text(phrase, MAX_CHARS_PER_LINE)
        display_text = "\n".join(lines)

        entries.append({
            "index": idx,
            "start": round(current_time, 3),
            "end": round(current_time + duration, 3),
            "text": display_text,
        })
        current_time += duration

    return entries


def generate_srt_file(
    all_entries: list[SubtitleEntry],
    output_path: Path,
) -> Path:
    """Write subtitle entries to an SRT file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, entry in enumerate(all_entries, start=1):
        start = _format_srt_time(entry["start"])
        end = _format_srt_time(entry["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(entry["text"])
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("✅ SRT generado: %s (%d entradas)", output_path.name, len(all_entries))
    return output_path


def _split_into_phrases(text: str) -> list[str]:
    """Split text into short subtitle phrases (~8 words max)."""
    sentences = re.split(r"(?<=[.!?…;:])\s+", text.strip())
    phrases = []
    for sentence in sentences:
        parts = re.split(r"(?<=,)\s+", sentence)
        for part in parts:
            words = part.split()
            if len(words) <= 8:
                phrases.append(part)
            else:
                for i in range(0, len(words), 7):
                    chunk = " ".join(words[i : i + 7])
                    phrases.append(chunk)
    return [p for p in phrases if p.strip()]


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Wrap text to fit subtitle display (max 2 lines)."""
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    mid = len(words) // 2
    line1 = " ".join(words[:mid])
    line2 = " ".join(words[mid:])
    return [line1, line2]


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
