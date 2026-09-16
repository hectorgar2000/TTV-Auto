"""Video pipeline state definition and reducers."""
from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict


class SubtitleEntry(TypedDict, total=False):
    index: int
    start: float
    end: float
    text: str


class Scene(TypedDict, total=False):
    index: int
    narration_text: str
    image_prompt_en: str
    wikimedia_search_terms: list[str]
    wikimedia_image_url: str
    target_duration: float
    audio_path: str
    image_path: str
    duration_real: float
    subtitle_entries: list[SubtitleEntry]


class VideoMetadata(TypedDict, total=False):
    title: str
    description: str
    tags: list[str]


def merge_scenes(existing: list[Scene], updates: list[Scene]) -> list[Scene]:
    """Reducer: merge partial scene updates by index.

    When a Send node processes scene 3 and returns
    [{"index": 3, "audio_path": "..."}], this merges it into
    the existing scenes list without overwriting other scenes.
    """
    if not existing:
        return updates or []
    if not updates:
        return existing
    result = [dict(s) for s in existing]
    for update in updates:
        idx = update.get("index")
        if idx is not None and 0 <= idx < len(result):
            merged = {k: v for k, v in update.items() if v is not None}
            result[idx] = {**result[idx], **merged}
    return result


class VideoState(TypedDict, total=False):
    topic: str
    research_notes: str
    sources: Annotated[list[str], operator.add]
    fact_check_notes: str
    script: str
    scenes: Annotated[list[Scene], merge_scenes]
    video_path: str
    metadata: VideoMetadata
    review_notes: str
    review_requested: bool
    current_scene_index: int
    error_log: Annotated[list[str], operator.add]
