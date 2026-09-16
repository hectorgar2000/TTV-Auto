"""Pipeline configuration: env vars > config.yaml > defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    # LLM (desacoplado: cambiar modelo/proveedor solo aqui)
    llm_model: str = "claude-sonnet-4-20250514"
    llm_temperature: float = 0.7

    # TTS: "f5" (default, GPU) | "edge" (gratis, online) | "xtts" | "piper"
    tts_engine: str = "f5"
    f5_ref_audio: str = ""
    f5_ref_text: str = ""
    edge_tts_voice: str = "es-ES-AlvaroNeural"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_language: str = "es"
    piper_model_path: str = ""
    tts_speaker_wav: str = ""

    # Stable Diffusion
    sd_model: str = "stabilityai/sdxl-turbo"
    sd_steps: int = 4
    sd_guidance_scale: float = 0.0  # SDXL-Turbo no usa guidance
    image_width: int = 1280
    image_height: int = 720

    # Video
    video_width: int = 1920
    video_height: int = 1080
    fps: int = 24
    music_volume: float = 0.08
    ken_burns_zoom: float = 1.15

    # Rutas
    output_dir: Path = field(default_factory=lambda: Path("output"))
    music_dir: Path = field(default_factory=lambda: Path("assets/music"))
    db_path: Path = field(default_factory=lambda: Path("checkpoints.db"))

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Config:
        data: dict = {}
        if config_path and Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        env_map: dict[str, str | tuple[str, type]] = {
            "LLM_MODEL": "llm_model",
            "LLM_TEMPERATURE": ("llm_temperature", float),
            "TTS_ENGINE": "tts_engine",
            "F5_REF_AUDIO": "f5_ref_audio",
            "F5_REF_TEXT": "f5_ref_text",
            "EDGE_TTS_VOICE": "edge_tts_voice",
            "TTS_SPEAKER_WAV": "tts_speaker_wav",
            "PIPER_MODEL_PATH": "piper_model_path",
            "SD_MODEL": "sd_model",
            "SD_STEPS": ("sd_steps", int),
            "IMAGE_WIDTH": ("image_width", int),
            "IMAGE_HEIGHT": ("image_height", int),
            "VIDEO_WIDTH": ("video_width", int),
            "VIDEO_HEIGHT": ("video_height", int),
            "MUSIC_VOLUME": ("music_volume", float),
        }
        for env_key, field_info in env_map.items():
            val = os.getenv(env_key)
            if val is not None:
                if isinstance(field_info, tuple):
                    name, converter = field_info
                    data[name] = converter(val)
                else:
                    data[field_info] = val

        for pf in ("output_dir", "music_dir", "db_path"):
            if pf in data:
                data[pf] = Path(data[pf])

        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})
