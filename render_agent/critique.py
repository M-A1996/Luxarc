"""
Critique — the vision model looks at the render and scores it. This is what makes it
an AGENT (it judges its own output and drives the refine loop).

Runs on local Ollama vision by default (nothing leaves) or Groq vision (render leaves).
Returns {"score": float 0-10, "feedback": str}.
"""
import json
import re

from render_agent.config import Config
from render_agent.providers import vision_chat
from render_agent.logging import log

_Q = (
    "You are reviewing an AI architectural render against a client's request.\n"
    "Request: \"{request}\"\n"
    "Rate how well the render matches the request AND whether it looks like a plausible, "
    "undistorted building (correct shape, no melted/extra structures).\n"
    "Reply with ONLY JSON: {{\"score\": <0-10 number>, \"feedback\": \"<one sentence: what to improve>\"}}"
)


def critique(render_image, request: str) -> dict:
    try:
        raw = vision_chat(render_image, _Q.format(request=request))
    except Exception as e:
        log(f"[yellow]critique: vision model unavailable ({e}) — accepting render as-is.[/yellow]")
        return {"score": Config.MIN_SCORE, "feedback": "vision critic unavailable"}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"score": Config.MIN_SCORE, "feedback": raw[:120]}
    try:
        d = json.loads(m.group(0))
        score = float(d.get("score", Config.MIN_SCORE))
        fb = str(d.get("feedback", ""))
    except Exception:
        score, fb = Config.MIN_SCORE, raw[:120]
    log(f"[cyan]critique: score {score}/10 — {fb}[/cyan]")
    return {"score": score, "feedback": fb}
