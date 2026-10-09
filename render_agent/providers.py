"""
LLM providers — the 'director' (text) and the 'critic' (vision).

  director_chat(prompt)          -> text.  Groq (cloud) or Ollama (local).
  vision_chat(image, prompt)     -> text.  Groq vision (cloud) or Ollama vision (local).

Groq is the free/fast option for the brain. It does NOT render images — the actual
render lives in render_provider.py.
"""
import base64
import io

from render_agent.config import Config
from render_agent.logging import log


def _img_to_data_uri(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ director (text)
def _make_text_llm():
    if Config.LLM_MODE == "cloud":
        if not Config.GROQ_API_KEY:
            raise RuntimeError("LLM_MODE=cloud but GROQ_API_KEY is not set (or use LLM_MODE=local).")
        from langchain_groq import ChatGroq
        log(f"[magenta]director: Groq {Config.GROQ_MODEL} (text only leaves)[/magenta]")
        return ChatGroq(model=Config.GROQ_MODEL, api_key=Config.GROQ_API_KEY, temperature=0.2)
    from langchain_ollama import ChatOllama
    log(f"[magenta]director: local Ollama {Config.OLLAMA_TEXT_MODEL}[/magenta]")
    return ChatOllama(model=Config.OLLAMA_TEXT_MODEL, temperature=0.2)


def director_chat(prompt: str) -> str:
    llm = _make_text_llm()
    resp = llm.invoke(prompt)
    return (getattr(resp, "content", str(resp)) or "").strip()


# ------------------------------------------------------------------ critic (vision)
def _make_vision_llm():
    if Config.VISION_MODE == "cloud":
        if not Config.GROQ_API_KEY:
            raise RuntimeError("VISION_MODE=cloud but GROQ_API_KEY is not set (or use VISION_MODE=local).")
        from langchain_groq import ChatGroq
        log(f"[magenta]critic: Groq vision {Config.GROQ_VISION_MODEL} (RENDER image leaves)[/magenta]")
        return ChatGroq(model=Config.GROQ_VISION_MODEL, api_key=Config.GROQ_API_KEY, temperature=0)
    from langchain_ollama import ChatOllama
    log(f"[magenta]critic: local Ollama vision {Config.OLLAMA_VISION_MODEL} (nothing leaves)[/magenta]")
    return ChatOllama(model=Config.OLLAMA_VISION_MODEL, temperature=0)


def vision_chat(image, prompt: str) -> str:
    """Send an image + a question to the vision model; return its text answer."""
    from langchain_core.messages import HumanMessage
    llm = _make_vision_llm()
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _img_to_data_uri(image)}},
    ])
    resp = llm.invoke([msg])
    return (getattr(resp, "content", str(resp)) or "").strip()
