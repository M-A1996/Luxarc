"""
Central settings for the Architectural Render Agent.

Everything is FREE-capable and defaults to the most private option. Cloud pieces
(Groq brain, HF/fal render) are opt-in via .env. Nothing here contacts the network
unless you switch a mode to 'cloud'/'hf'/'fal' and provide a key.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Authenticate Hugging Face model DOWNLOADS with the HF key (free — just removes the
# anonymous rate-limit so weights download much faster). Separate from paid inference.
if os.getenv("HF_API_KEY") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]


class Config:
    # ---- the "director" brain (turns your words into render settings) ----
    #   local -> Ollama on your machine (private, free)
    #   cloud -> Groq llama-3 (free tier, fast). TEXT ONLY — no image is sent for this step.
    LLM_MODE = os.getenv("LLM_MODE", "cloud")            # local | cloud
    OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "qwen2.5")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ---- the vision critic (looks at the render and scores it) ----
    #   local -> Ollama vision (private; the render never leaves)   << default for privacy
    #   cloud -> Groq vision (free tier, but the render image leaves the machine)
    VISION_MODE = os.getenv("VISION_MODE", "local")      # local | cloud
    OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
    GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")

    # ---- the renderer (diffusion + ControlNet). Groq CANNOT do this. ----
    #   local -> diffusers on your GPU (free, private, but slow on an iGPU)
    #   hf    -> Hugging Face Inference API (free tier; no-train)   << easy free cloud
    #   fal   -> fal.ai (fast; free credits then paid; no-train)
    RENDER_PROVIDER = os.getenv("RENDER_PROVIDER", "hf")  # local | hf | fal
    # If the primary provider fails (rate-limited / error / no key), fall back to this one.
    #   local -> render on your GPU (free, never rate-limited, but slow on a CPU/iGPU)
    #   none  -> don't fall back; raise the error
    RENDER_FALLBACK = os.getenv("RENDER_FALLBACK", "local")  # local | none
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    # A ControlNet-capable model on the provider. Adjust to a model the provider serves.
    HF_RENDER_MODEL = os.getenv("HF_RENDER_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
    FAL_API_KEY = os.getenv("FAL_API_KEY", "")
    FAL_RENDER_MODEL = os.getenv("FAL_RENDER_MODEL", "fal-ai/sdxl-controlnet-union")
    # SD 1.5 + ControlNet-canny — light enough to run on CPU/iGPU (SDXL is too heavy there).
    LOCAL_BASE_MODEL = os.getenv("LOCAL_BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    LOCAL_CONTROLNET = os.getenv("LOCAL_CONTROLNET", "lllyasviel/sd-controlnet-canny")

    # ---- structure map: what gets sent instead of the real image ----
    #   line  -> Canny edge map (fully local, free, no extra models)   << default
    #   depth -> depth map (needs a small local depth model; heavier)
    STRUCTURE_TYPE = os.getenv("STRUCTURE_TYPE", "line")  # line | depth

    # ---- privacy knob #1: redact identifying marks before mapping ----
    REDACT_TEXT = os.getenv("REDACT_TEXT", "1") == "1"    # OCR-blur title blocks/logos/addresses/labels

    # ---- the self-correcting loop ----
    MAX_REFINES = int(os.getenv("MAX_REFINES", "2"))      # extra re-renders if the critic isn't happy
    MIN_SCORE = float(os.getenv("MIN_SCORE", "7"))        # 0-10; accept at or above this

    # ---- render params (defaults; the director can override) ----
    STEPS = int(os.getenv("STEPS", "28"))
    GUIDANCE = float(os.getenv("GUIDANCE", "7.0"))
    CONTROLNET_SCALE = float(os.getenv("CONTROLNET_SCALE", "0.8"))

    # ---- folders ----
    HOME = Path(__file__).parent.parent
    INPUT_DIR = Path(os.getenv("INPUT_DIR", str(HOME / "input")))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(HOME / "output")))
    WORK_DIR = HOME / "work"           # intermediate artifacts (redacted, structure map)

    @staticmethod
    def ensure_dirs():
        for d in (Config.INPUT_DIR, Config.OUTPUT_DIR, Config.WORK_DIR):
            Path(d).mkdir(parents=True, exist_ok=True)
