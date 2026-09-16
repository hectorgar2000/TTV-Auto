"""Text-to-Speech: F5-TTS (default, GPU), Edge TTS, XTTS v2, Piper."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)

_f5_instance = None
_xtts_instance = None

_FFMPEG_DLL_DIR = Path(__file__).resolve().parent.parent / "lib" / "ffmpeg"


def _ensure_ffmpeg_dlls():
    """Add ffmpeg shared DLLs to the process search path (Windows)."""
    if _FFMPEG_DLL_DIR.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_FFMPEG_DLL_DIR))


def _get_f5(config: Config):
    """Return a cached F5-TTS model instance."""
    global _f5_instance
    if _f5_instance is None:
        _ensure_ffmpeg_dlls()
        from f5_tts.api import F5TTS
        logger.info("🔄 Cargando modelo F5-TTS (una sola vez)...")
        _f5_instance = F5TTS(device="cuda" if _has_gpu() else "cpu")
        logger.info("✅ Modelo F5-TTS cargado")
    return _f5_instance


def _get_xtts(config: Config):
    """Return a cached XTTS model instance."""
    global _xtts_instance
    if _xtts_instance is None:
        from TTS.api import TTS
        logger.info("🔄 Cargando modelo XTTS v2 (una sola vez)...")
        _xtts_instance = TTS(model_name=config.xtts_model, gpu=_has_gpu())
        logger.info("✅ Modelo XTTS cargado")
    return _xtts_instance


def generate_voice(
    text: str,
    output_path: Path,
    config: Config,
) -> Path:
    """Generate speech audio for the given text.

    Engines: "f5" (default, GPU), "edge", "xtts" (GPU, voice cloning), "piper".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("⏭️ Audio ya existe: %s", output_path.name)
        return output_path

    engine = config.tts_engine
    if engine == "piper":
        return _generate_piper(text, output_path, config)
    if engine == "xtts":
        return _generate_xtts(text, output_path, config)
    if engine == "edge":
        return _generate_edge(text, output_path, config)
    return _generate_f5(text, output_path, config)


def _generate_f5(text: str, output_path: Path, config: Config) -> Path:
    """Generate voice using F5-TTS (local GPU, high quality, multilingual)."""
    logger.info("🎙️ Generando voz con F5-TTS (%d chars)...", len(text))
    try:
        tts = _get_f5(config)

        ref_audio = config.f5_ref_audio
        ref_text = config.f5_ref_text

        if not ref_audio:
            import f5_tts.infer.utils_infer as _u
            _pkg = Path(_u.__file__).parent / "examples" / "basic"
            ref_audio = str(_pkg / "basic_ref_en.wav")
            ref_text = "Some call me nature, others call me mother nature."

        wav, sr, _ = tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
        )

        import soundfile as sf
        wav_path = output_path.with_suffix(".wav")
        sf.write(str(wav_path), wav, sr)
        logger.info("✅ Audio F5-TTS generado: %s", wav_path.name)
        return wav_path

    except ImportError:
        logger.error("f5-tts no instalado. Instala con: pip install f5-tts")
        raise
    except Exception as e:
        logger.error("Error generando audio F5-TTS: %s", e)
        raise


def _generate_edge(text: str, output_path: Path, config: Config) -> Path:
    """Generate voice using Microsoft Edge TTS (free, no GPU)."""
    logger.info("🎙️ Generando voz con Edge TTS (%d chars)...", len(text))
    try:
        import edge_tts

        voice = config.edge_tts_voice
        mp3_path = output_path.with_suffix(".mp3")

        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(mp3_path))

        asyncio.run(_run())

        logger.info("✅ Audio Edge TTS generado: %s", mp3_path.name)
        return mp3_path

    except ImportError:
        logger.error("edge-tts no instalado. Instala con: pip install edge-tts")
        raise
    except Exception as e:
        logger.error("Error generando audio Edge TTS: %s", e)
        raise


def _generate_xtts(text: str, output_path: Path, config: Config) -> Path:
    logger.info("🎙️ Generando voz con XTTS v2 (%d chars)...", len(text))
    try:
        tts = _get_xtts(config)

        kwargs = {
            "text": text,
            "file_path": str(output_path),
            "language": config.xtts_language,
        }
        if config.tts_speaker_wav and Path(config.tts_speaker_wav).exists():
            kwargs["speaker_wav"] = config.tts_speaker_wav

        tts.tts_to_file(**kwargs)
        logger.info("✅ Audio XTTS generado: %s", output_path.name)
        return output_path

    except ImportError:
        logger.error("coqui-tts no instalado. Instala con: pip install coqui-tts")
        raise
    except Exception as e:
        logger.error("Error generando audio XTTS: %s", e)
        raise


def _generate_piper(text: str, output_path: Path, config: Config) -> Path:
    logger.info("🎙️ Generando voz con Piper (%d chars)...", len(text))
    model_path = config.piper_model_path
    if not model_path:
        raise ValueError(
            "PIPER_MODEL_PATH no configurado. "
            "Descarga un modelo de https://github.com/rhasspy/piper#voices"
        )

    wav_path = output_path.with_suffix(".wav")
    cmd = [
        "piper",
        "--model", model_path,
        "--output_file", str(wav_path),
    ]
    result = subprocess.run(
        cmd,
        input=text,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Piper failed: {result.stderr}")

    logger.info("✅ Audio Piper generado: %s", wav_path.name)
    return wav_path


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds using mutagen."""
    from mutagen import File as MutagenFile

    audio = MutagenFile(str(audio_path))
    if audio is None:
        import wave
        with wave.open(str(audio_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    return audio.info.length


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
