"""Image generation: Wikimedia Commons lookup + Stable Diffusion fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from tools.wikimedia import search_wikimedia_image

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


def generate_image(
    prompt_en: str,
    wikimedia_terms: list[str],
    output_path: Path,
    config: Config,
    wikimedia_url: str = "",
) -> Path:
    """Generate or find an image for a scene.

    Priority: existing file > Wikimedia Commons > Stable Diffusion.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("⏭️ Imagen ya existe: %s", output_path.name)
        return output_path

    # 1. Try Wikimedia Commons first
    if wikimedia_terms:
        wiki_path = output_path.with_name(output_path.stem + "_wiki")
        result = search_wikimedia_image(wikimedia_terms, wiki_path)
        if result:
            final = Path(result)
            if final != output_path:
                target = output_path.with_suffix(final.suffix)
                final.rename(target)
                return target
            return final

    # 2. Generate with Stable Diffusion
    return _generate_sd(prompt_en, output_path, config)


_sd_pipe = None
_sd_device = None


def _get_sd_pipeline(config: Config):
    """Return a cached Stable Diffusion pipeline (loaded once, reused across scenes)."""
    global _sd_pipe, _sd_device
    if _sd_pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image

        _sd_device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if _sd_device == "cuda" else torch.float32

        logger.info("🔄 Cargando modelo SD (una sola vez)...")
        _sd_pipe = AutoPipelineForText2Image.from_pretrained(
            config.sd_model,
            torch_dtype=dtype,
            variant="fp16" if _sd_device == "cuda" else None,
        )
        _sd_pipe = _sd_pipe.to(_sd_device)
        if hasattr(_sd_pipe, "enable_attention_slicing"):
            _sd_pipe.enable_attention_slicing()
        logger.info("✅ Modelo SD cargado")
    return _sd_pipe, _sd_device


def _generate_sd(prompt: str, output_path: Path, config: Config) -> Path:
    logger.info("🖼️ Generando imagen con SD: %s", prompt[:80])

    try:
        pipe, device = _get_sd_pipeline(config)

        sd_w = min(config.image_width, 512)
        sd_h = min(config.image_height, 512)

        image = pipe(
            prompt=prompt,
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            width=sd_w,
            height=sd_h,
        ).images[0]

        if sd_w < config.image_width or sd_h < config.image_height:
            from PIL import Image
            image = image.resize(
                (config.image_width, config.image_height),
                Image.LANCZOS,
            )

        save_path = output_path.with_suffix(".png")
        image.save(str(save_path))
        logger.info("✅ Imagen SD generada: %s", save_path.name)
        return save_path

    except ImportError:
        logger.error(
            "diffusers/torch no instalados. "
            "Instala con: pip install diffusers torch transformers accelerate"
        )
        raise
    except Exception as e:
        logger.error("Error generando imagen SD: %s", e)
        raise


def create_placeholder_image(output_path: Path, text: str = "") -> Path:
    """Create a simple placeholder image when generation fails."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1280, 720), color=(30, 30, 40))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    label = text[:60] if text else "Imagen no disponible"
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((1280 - w) / 2, (720 - h) / 2),
        label,
        fill=(200, 200, 200),
        font=font,
    )

    save_path = output_path.with_suffix(".png")
    img.save(str(save_path))
    return save_path
