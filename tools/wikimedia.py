"""Wikimedia Commons API: search for public domain images."""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
TIMEOUT = 15


def search_wikimedia_image(
    search_terms: list[str],
    output_path: Path,
    max_results: int = 5,
) -> str | None:
    """Search Wikimedia Commons and download the best matching image.

    Returns the local file path if found and downloaded, None otherwise.
    """
    if not search_terms:
        return None

    query = " ".join(search_terms)
    logger.info("🌐 Buscando en Wikimedia Commons: %s", query)

    try:
        results = _search_images(query, max_results)
        if not results:
            logger.info("No se encontraron imágenes en Wikimedia Commons")
            return None

        for title in results:
            url = _get_image_url(title)
            if url and _is_suitable(url):
                local_path = _download_image(url, output_path)
                if local_path:
                    logger.info("✅ Imagen descargada: %s", local_path)
                    return str(local_path)

        logger.info("No se encontró imagen adecuada en Wikimedia Commons")
        return None

    except Exception as e:
        logger.warning("Error buscando en Wikimedia: %s", e)
        return None


def _search_images(query: str, limit: int) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",  # File namespace
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
    }
    resp = requests.get(COMMONS_API, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    titles = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 999)):
        info = (page.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        if mime.startswith("image/"):
            titles.append(page["title"])
    return titles


def _get_image_url(title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": "1280",
    }
    resp = requests.get(COMMONS_API, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        return info.get("thumburl") or info.get("url")
    return None


def _is_suitable(url: str) -> bool:
    """Check the image URL points to a usable format."""
    lower = url.lower()
    return any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))


def _download_image(url: str, output_path: Path) -> Path | None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ext = Path(url.split("?")[0]).suffix or ".jpg"
        final_path = output_path.with_suffix(ext)
        final_path.write_bytes(resp.content)
        return final_path
    except Exception as e:
        logger.warning("Error descargando imagen: %s", e)
        return None
