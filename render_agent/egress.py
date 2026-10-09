"""
Egress posture — a single place that states (and logs) what leaves the machine.

In cloud mode the ONLY things that leave are:
  * the structure map (line/depth 'faceless blob') + the text prompt  -> render provider
  * the text request + settings                                       -> Groq (director), if LLM_MODE=cloud
  * the RENDER image                                                   -> Groq (critic), ONLY if VISION_MODE=cloud

The original design image, its materials, context, logos, addresses and client
details NEVER leave — they are used only locally (redact + structure map).
"""
from render_agent.config import Config
from render_agent.logging import log


def summary() -> str:
    parts = []
    parts.append(f"render→{Config.RENDER_PROVIDER}" +
                 ("" if Config.RENDER_PROVIDER == "local" else " (structure map + prompt only)"))
    parts.append(f"director→{'groq' if Config.LLM_MODE == 'cloud' else 'local'}" +
                 (" (text only)" if Config.LLM_MODE == "cloud" else ""))
    parts.append(f"critic→{'groq (render leaves!)' if Config.VISION_MODE == 'cloud' else 'local (nothing leaves)'}")
    return " · ".join(parts)


def announce():
    if Config.RENDER_PROVIDER == "local" and Config.LLM_MODE == "local" and Config.VISION_MODE == "local":
        log("[green]EGRESS: fully local — nothing leaves the machine.[/green]")
    else:
        log(f"[cyan]EGRESS: {summary()}[/cyan]")
        log("[cyan]  → the original image, materials, branding & client details never leave.[/cyan]")
