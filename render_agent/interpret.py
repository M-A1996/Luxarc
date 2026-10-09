"""
Interpret — turn the architect's plain-English request into render settings.

Runs on the director LLM (Groq/local). TEXT ONLY: the image is NOT sent here, so
nothing about the design leaves at this step. Returns a settings dict the renderer
and the refine loop use.
"""
import json
import re

from render_agent.config import Config
from render_agent.providers import director_chat
from render_agent.logging import log

_SYS = """You are an architectural visualisation director. Turn the user's request into
settings for a ControlNet render that keeps the building's shape and paints the requested look.
Reply with ONLY a JSON object with these keys:
  "prompt": a rich photorealistic prompt (materials, lighting, mood, context) — no brand/client names,
  "negative_prompt": things to avoid (e.g. "blurry, distorted, extra buildings, text, watermark"),
  "guidance": number 4-9,
  "controlnet_scale": number 0.5-1.0 (higher = follow the shape more strictly),
  "steps": integer 20-40.
No prose, no code fences — just the JSON."""


def _defaults(request: str) -> dict:
    return {
        "prompt": f"photorealistic architectural render, {request}, high detail, realistic lighting",
        "negative_prompt": "blurry, distorted, extra buildings, people faces, text, watermark, logo",
        "guidance": Config.GUIDANCE,
        "controlnet_scale": Config.CONTROLNET_SCALE,
        "steps": Config.STEPS,
    }


def _parse(text: str, request: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return _defaults(request)
    try:
        d = json.loads(m.group(0))
    except Exception:
        return _defaults(request)
    base = _defaults(request)
    base.update({k: d[k] for k in base if k in d})
    return base


def interpret(request: str, refine_feedback: str = "") -> dict:
    prompt = _SYS + f"\n\nRequest: {request}"
    if refine_feedback:
        prompt += (f"\n\nThe previous render scored low. Fix this feedback by adjusting the "
                   f"prompt/settings: {refine_feedback}")
    try:
        raw = director_chat(prompt)
        settings = _parse(raw, request)
    except Exception as e:
        log(f"[yellow]interpret: director unavailable ({e}) — using default settings.[/yellow]")
        settings = _defaults(request)
    log(f"[cyan]interpret: prompt = {settings['prompt'][:80]}…[/cyan]")
    return settings
