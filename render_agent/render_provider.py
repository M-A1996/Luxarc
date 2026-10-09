"""
Renderer — diffusion + ControlNet. Takes the STRUCTURE MAP (+ prompt) and paints a
photorealistic render onto that shape. This is the only step where the shape leaves
(in cloud mode). Groq cannot do this — it's a diffusion job.

Providers:
  local -> diffusers SDXL + ControlNet on your GPU. True ControlNet, free, private
           (slow on an iGPU; best on NVIDIA / a rented GPU).
  hf    -> Hugging Face Inference API (free tier, no-train). Uses image-to-image
           guided by the structure map (approximate ControlNet).
  fal   -> fal.ai (fast, no-train; free credits then paid). Real ControlNet models.

NOTE: cloud providers' exact model ids / argument names vary — adjust in .env / here
to match a model the provider currently serves. The LOCAL path is the reference.
"""
import base64
import io

from PIL import Image

from render_agent.config import Config
from render_agent.logging import log


def _to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- local (true ControlNet)
_LOCAL_PIPE = None


def _render_local(structure: Image.Image, s: dict) -> Image.Image:
    """SD 1.5 + ControlNet-canny — light enough for CPU/iGPU. Free & private."""
    global _LOCAL_PIPE
    try:
        import torch
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    except ImportError as e:
        raise RuntimeError(
            "local render needs extra libs: pip install torch diffusers transformers accelerate"
        ) from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32   # fp16 only on GPU
    if _LOCAL_PIPE is None:
        log(f"[cyan]render(local): loading SD1.5 + ControlNet-canny on {device} "
            f"(first run downloads ~4GB; a render takes minutes on CPU)…[/cyan]")
        cn = ControlNetModel.from_pretrained(Config.LOCAL_CONTROLNET, torch_dtype=dtype)
        _LOCAL_PIPE = StableDiffusionControlNetPipeline.from_pretrained(
            Config.LOCAL_BASE_MODEL, controlnet=cn, torch_dtype=dtype, safety_checker=None
        ).to(device)
    ctrl = structure.convert("RGB").resize((512, 512))   # SD1.5 works best at 512
    out = _LOCAL_PIPE(
        prompt=s["prompt"], negative_prompt=s["negative_prompt"], image=ctrl,
        num_inference_steps=int(s["steps"]), guidance_scale=float(s["guidance"]),
        controlnet_conditioning_scale=float(s["controlnet_scale"]),
    )
    return out.images[0]


# ---------------------------------------------------------------- Hugging Face (free tier)
def _render_hf(structure: Image.Image, s: dict) -> Image.Image:
    if not Config.HF_API_KEY:
        raise RuntimeError("RENDER_PROVIDER=hf but HF_API_KEY is not set.")
    from huggingface_hub import InferenceClient
    log(f"[cyan]render(hf): sending STRUCTURE MAP + prompt to {Config.HF_RENDER_MODEL} "
        f"(free tier, no-train). No original image, no materials leave.[/cyan]")
    client = InferenceClient(token=Config.HF_API_KEY)
    # image-to-image guided by the structure map (approximate ControlNet on serverless)
    return client.image_to_image(
        image=structure.convert("RGB"), prompt=s["prompt"],
        model=Config.HF_RENDER_MODEL, negative_prompt=s["negative_prompt"],
        num_inference_steps=int(s["steps"]), guidance_scale=float(s["guidance"]),
    )


# ---------------------------------------------------------------- fal.ai (real ControlNet)
def _render_fal(structure: Image.Image, s: dict) -> Image.Image:
    if not Config.FAL_API_KEY:
        raise RuntimeError("RENDER_PROVIDER=fal but FAL_API_KEY is not set.")
    import os
    os.environ["FAL_KEY"] = Config.FAL_API_KEY
    import fal_client
    log(f"[cyan]render(fal): sending STRUCTURE MAP + prompt to {Config.FAL_RENDER_MODEL} "
        f"(no-train). No original image leaves.[/cyan]")
    result = fal_client.subscribe(Config.FAL_RENDER_MODEL, arguments={
        "prompt": s["prompt"], "negative_prompt": s["negative_prompt"],
        "canny_image_url": _to_data_uri(structure),            # fal-ai/sdxl-controlnet-union input name
        "num_inference_steps": int(s["steps"]), "guidance_scale": float(s["guidance"]),
        "controlnet_conditioning_scale": float(s["controlnet_scale"]),
    })
    url = (result.get("images") or [{}])[0].get("url") or result.get("image", {}).get("url")
    import requests
    return Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("RGB")


def _render_with(provider: str, structure: Image.Image, settings: dict) -> Image.Image:
    if provider == "local":
        return _render_local(structure, settings)
    if provider == "fal":
        return _render_fal(structure, settings)
    if provider == "hf":
        return _render_hf(structure, settings)
    raise RuntimeError(f"unknown RENDER_PROVIDER: {provider}")


def render(structure: Image.Image, settings: dict) -> Image.Image:
    """Try the configured provider; if it fails (rate limit / error / no key), fall
    back to RENDER_FALLBACK (local) so a throttle never blocks a render."""
    primary = Config.RENDER_PROVIDER
    try:
        return _render_with(primary, structure, settings)
    except Exception as e:
        fb = (Config.RENDER_FALLBACK or "none").lower()
        if fb not in ("none", "", primary):
            log(f"[yellow]render: '{primary}' failed ({type(e).__name__}: {e}). "
                f"Falling back to '{fb}' (local — free, never rate-limited, slow)…[/yellow]")
            return _render_with(fb, structure, settings)
        raise
