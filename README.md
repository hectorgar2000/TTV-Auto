# LangGraph Video Pipeline

Sistema multiagente que produce vídeos de YouTube de 8-15 minutos sobre
lore de universos de fantasía (El Señor de los Anillos, etc.), listos
para montar. Orquestado con **LangGraph** y **LangChain**.

## Arquitectura

```
research → fact_check → scriptwriter → storyboard
    → [Send ×N: voice] → [Send ×N: image] → subtitles
    → (review gate) → video_assembly → metadata → qa
```

- **Agentes LLM** (Claude vía langchain-anthropic): investigación, verificación,
  guion, storyboard, metadatos YouTube.
- **Procesamiento paralelo** (Send pattern): voz e imágenes por escena.
- **Persistencia** (SqliteSaver): el pipeline se puede reanudar si falla a mitad.
- **Human-in-the-loop**: flag `--review` para revisar antes del montaje.

## Requisitos

### Hardware
- **Con GPU (recomendado)**: NVIDIA con ≥6 GB VRAM para Stable Diffusion
  y XTTS v2. Una RTX 3060 o superior funciona bien.
- **Sin GPU**: usa Piper TTS (`TTS_ENGINE=piper`) y un modelo SD más
  ligero. La generación será más lenta pero funcional.

### Software
- Python 3.10+
- ffmpeg instalado y en PATH (`winget install ffmpeg` o `choco install ffmpeg`)
- ~10 GB de disco para modelos (XTTS v2 + Stable Diffusion)

## Instalación

```bash
# 1. Clonar y entrar al directorio
cd LangGraph

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API key de Anthropic
set ANTHROPIC_API_KEY=sk-ant-...        # Windows
# export ANTHROPIC_API_KEY=sk-ant-...   # Linux/Mac

# 5. (Primera vez) Los modelos de XTTS v2 y Stable Diffusion se descargan
# automáticamente en la primera ejecución (~5 GB cada uno).
```

### Piper TTS (alternativa sin GPU)

```bash
pip install piper-tts
# Descargar un modelo de voz en español:
# https://github.com/rhasspy/piper#voices
# Configurar la ruta:
set PIPER_MODEL_PATH=ruta/al/modelo.onnx
set TTS_ENGINE=piper
```

## Uso

```bash
# Generar un vídeo
python main.py --tema "La caída de Gondolin"

# Con revisión manual antes del montaje
python main.py --tema "La historia de Isildur y el Anillo Único" --review

# Reanudar un pipeline interrumpido
python main.py --tema "La caída de Gondolin" --resume

# Con config personalizada
python main.py --tema "Los Balrogs de Morgoth" --config config.yaml --verbose
```

### Salida

```
output/
└── la-caida-de-gondolin/
    ├── audio/
    │   ├── scene_000.wav
    │   ├── scene_001.wav
    │   └── ...
    ├── images/
    │   ├── scene_000.png
    │   ├── scene_001.png
    │   └── ...
    ├── subtitles.srt
    └── video_final.mp4      ← este es el entregable
```

## Configuración

Variables de entorno (sobrescriben config.yaml):

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | API key de Anthropic (obligatoria) |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Modelo de Claude a usar |
| `TTS_ENGINE` | `xtts` | `xtts` o `piper` |
| `TTS_SPEAKER_WAV` | — | Audio de referencia para clonar voz (XTTS) |
| `SD_MODEL` | `stabilityai/sdxl-turbo` | Modelo de Stable Diffusion |
| `SD_STEPS` | `4` | Pasos de inferencia SD |
| `MUSIC_VOLUME` | `0.08` | Volumen de música de fondo (0.0-1.0) |
| `MUSIC_TRACK` | — | Forzar una pista concreta de `assets/music/` |

### config.yaml (opcional)

```yaml
llm_model: claude-sonnet-4-20250514
tts_engine: xtts
sd_model: stabilityai/sdxl-turbo
sd_steps: 4
image_width: 1280
image_height: 720
video_width: 1920
video_height: 1080
music_volume: 0.08
ken_burns_zoom: 1.15
```

## Música de fondo

Coloca pistas libres de derechos (MP3/WAV) en `assets/music/`.
El sistema elige una al azar. Fuentes recomendadas:

- [Free Music Archive](https://freemusicarchive.org) (filtrar CC0)
- [Pixabay Music](https://pixabay.com/music/)
- [Incompetech](https://incompetech.com) (Kevin MacLeod, CC BY)

## Estructura del proyecto

```
├── state.py          # Estado del pipeline (TypedDict + reducers)
├── config.py         # Configuración (env/yaml)
├── graph.py          # StateGraph de LangGraph
├── main.py           # CLI
├── agents/           # Agentes LLM
│   ├── research.py   # Investigación con búsqueda web
│   ├── fact_check.py # Verificación de lore
│   ├── scriptwriter.py # Escritura del guion
│   ├── storyboard.py # Prompts de imagen por escena
│   └── metadata.py   # Título/descripción/tags YouTube
├── tools/            # Herramientas no-LLM
│   ├── voice.py      # TTS (XTTS v2 / Piper)
│   ├── images.py     # Stable Diffusion + Wikimedia
│   ├── wikimedia.py  # API de Wikimedia Commons
│   ├── subtitles.py  # Generación de SRT
│   └── video.py      # Montaje con moviepy
├── assets/music/     # Pistas de música CC0
└── output/           # Vídeos generados
```
