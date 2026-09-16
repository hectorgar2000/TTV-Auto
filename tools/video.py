"""Video assembly: combines images, audio, subtitles and music into MP4."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from state import Scene, SubtitleEntry

logger = logging.getLogger(__name__)


def assemble_video(
    scenes: list[Scene],
    output_path: Path,
    config: Config,
) -> Path:
    """Assemble all scenes into the final MP4 video.

    For each scene: Ken Burns effect on the image + narration audio.
    Then: concatenate all scenes, add subtitles, mix background music.
    """
    from moviepy.audio.AudioClip import CompositeAudioClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.video.VideoClip import ImageClip, TextClip

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("🎬 Montando vídeo (%d escenas)...", len(scenes))

    scene_clips = []
    for scene in scenes:
        audio_path = scene.get("audio_path", "")
        image_path = scene.get("image_path", "")
        duration = scene.get("duration_real", 10.0)

        if not audio_path or not Path(audio_path).exists():
            logger.warning("Escena %d: sin audio, saltando", scene.get("index", -1))
            continue
        if not image_path or not Path(image_path).exists():
            logger.warning("Escena %d: sin imagen, saltando", scene.get("index", -1))
            continue

        audio_clip = AudioFileClip(str(audio_path))
        actual_duration = audio_clip.duration

        img_clip = _create_ken_burns_clip(
            image_path=str(image_path),
            duration=actual_duration,
            target_size=(config.video_width, config.video_height),
            zoom_factor=config.ken_burns_zoom,
        )
        img_clip = img_clip.with_audio(audio_clip)

        subtitle_entries = scene.get("subtitle_entries", [])
        if subtitle_entries:
            img_clip = _burn_subtitles(img_clip, subtitle_entries, config)

        scene_clips.append(img_clip)

    if not scene_clips:
        raise RuntimeError("No hay escenas válidas para montar")

    final = concatenate_videoclips(scene_clips, method="compose")

    music_path = _pick_music(config.music_dir)
    if music_path:
        final = _add_background_music(final, music_path, config.music_volume)

    logger.info("💾 Exportando MP4: %s", output_path)
    try:
        final.write_videofile(
            str(output_path),
            fps=config.fps,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            logger=None,
        )
    finally:
        for clip in scene_clips:
            clip.close()
        final.close()

    logger.info("✅ Vídeo exportado: %s", output_path)
    return output_path


def _create_ken_burns_clip(
    image_path: str,
    duration: float,
    target_size: tuple[int, int],
    zoom_factor: float = 1.15,
):
    """Create an image clip with a slow zoom (Ken Burns) effect."""
    from moviepy.video.VideoClip import ImageClip
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    img = img.resize(
        (int(target_size[0] * zoom_factor), int(target_size[1] * zoom_factor)),
        Image.LANCZOS,
    )
    img_array = np.array(img)
    tw, th = target_size

    def make_frame(t):
        progress = t / max(duration, 0.1)
        scale = 1.0 + (zoom_factor - 1.0) * (1.0 - progress)
        cw = int(tw * scale)
        ch = int(th * scale)
        cw = min(cw, img_array.shape[1])
        ch = min(ch, img_array.shape[0])
        x = (img_array.shape[1] - cw) // 2
        y = (img_array.shape[0] - ch) // 2
        crop = img_array[y : y + ch, x : x + cw]
        pil_crop = Image.fromarray(crop).resize(target_size, Image.LANCZOS)
        return np.array(pil_crop)

    from moviepy.video.VideoClip import VideoClip

    clip = VideoClip(make_frame, duration=duration)
    return clip


def _burn_subtitles(clip, subtitle_entries: list[SubtitleEntry], config: Config):
    """Burn subtitle entries into the video clip."""
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.video.VideoClip import TextClip

    sub_clips = []
    for entry in subtitle_entries:
        start = entry.get("start", 0)
        end = entry.get("end", 0)
        text = entry.get("text", "")
        if not text or end <= start:
            continue

        scene_start = subtitle_entries[0].get("start", 0)
        local_start = start - scene_start
        local_end = end - scene_start

        try:
            txt_clip = (
                TextClip(
                    text=text,
                    font_size=42,
                    color="white",
                    stroke_color="black",
                    stroke_width=2,
                    font="Arial",
                    size=(config.video_width - 200, None),
                )
                .with_position(("center", config.video_height - 140))
                .with_start(max(0, local_start))
                .with_duration(local_end - local_start)
            )
            sub_clips.append(txt_clip)
        except Exception as e:
            logger.warning("Error creando subtítulo: %s", e)

    if sub_clips:
        return CompositeVideoClip([clip] + sub_clips)
    return clip


def _pick_music(music_dir: Path) -> Path | None:
    """Pick a random music file from the assets/music/ directory."""
    music_dir = Path(music_dir)
    if not music_dir.exists():
        return None

    import os
    forced = os.getenv("MUSIC_TRACK")
    if forced:
        forced_path = music_dir / forced
        if forced_path.exists():
            return forced_path

    tracks = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
    if not tracks:
        logger.info("No se encontraron pistas de música en %s", music_dir)
        return None

    chosen = random.choice(tracks)
    logger.info("🎵 Música seleccionada: %s", chosen.name)
    return chosen


def _add_background_music(video_clip, music_path: Path, volume: float):
    """Mix background music at low volume under the narration."""
    from moviepy.audio.AudioClip import CompositeAudioClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

    music = AudioFileClip(str(music_path))

    if music.duration < video_clip.duration:
        loops_needed = int(video_clip.duration / music.duration) + 1
        from moviepy.audio.AudioClip import concatenate_audioclips
        music = concatenate_audioclips([music] * loops_needed)

    music = music.subclipped(0, video_clip.duration)
    music = music.with_volume_scaled(volume)

    if video_clip.duration > 4:
        music = music.with_effects([AudioFadeIn(2), AudioFadeOut(2)])

    original_audio = video_clip.audio
    if original_audio:
        mixed = CompositeAudioClip([original_audio, music])
    else:
        mixed = music

    return video_clip.with_audio(mixed)
