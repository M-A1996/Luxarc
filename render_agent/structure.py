"""
Structure map — the 'faceless blob' that leaves instead of the real image.

  line  -> Canny edge map (OpenCV). Fully local, free, no extra models. DEFAULT.
  depth -> depth map (transformers Depth-Anything / MiDaS). Heavier; local model.

The map keeps the building's SHAPE only — no materials, colour, context or branding.
"""
import numpy as np
from PIL import Image

from render_agent.config import Config
from render_agent.logging import log


def to_line_map(img: Image.Image, low: int = 80, high: int = 180) -> Image.Image:
    """Canny edge map — white outlines on black. Local & free."""
    import cv2
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(edges).convert("RGB")


def to_depth_map(img: Image.Image) -> Image.Image:
    """Depth map via transformers pipeline. Falls back to a line map if unavailable."""
    try:
        from transformers import pipeline
        depth = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        d = depth(img)["depth"]
        return d.convert("RGB")
    except Exception as e:
        log(f"[yellow]structure: depth model unavailable ({e}) — using line map instead.[/yellow]")
        return to_line_map(img)


def make_structure(img: Image.Image, kind: str = None) -> Image.Image:
    kind = kind or Config.STRUCTURE_TYPE
    if kind == "depth":
        log("[cyan]structure: building DEPTH map (shape only).[/cyan]")
        return to_depth_map(img)
    log("[cyan]structure: building LINE map (Canny, shape only).[/cyan]")
    return to_line_map(img)
