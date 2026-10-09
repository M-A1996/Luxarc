"""
Luxarc — minimal Streamlit UI for the Architectural Render Agent.

Upload a 3D-model image, describe the look, hit Render. Privacy-first: in cloud
mode only the faceless structure map + prompt leave the machine — and the UI shows
you exactly that map so you can see what left.

Run:  python -m streamlit run app.py
"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from render_agent.config import Config
from render_agent import egress
from render_agent.graph import run_agent

load_dotenv()

APP_NAME = "Luxarc"     # ← rename here (Umbralux / Renvia / Arclume …)

st.set_page_config(page_title=f"{APP_NAME} — AI Architectural Renders", page_icon="🏛️", layout="wide")

# ---------------- header ----------------
st.title(f"🏛️ {APP_NAME}")
st.caption("Turn a 3D-model image into a client-ready render — geometry-faithful, privacy-first.")

# ---------------- sidebar: engine + egress ----------------
with st.sidebar:
    st.subheader("Engine")
    st.write(f"**Render:** `{Config.RENDER_PROVIDER}`")
    st.write(f"**Director:** {'Groq (cloud)' if Config.LLM_MODE == 'cloud' else 'local Ollama'}")
    st.write(f"**Critic:** {'Groq — render leaves ⚠️' if Config.VISION_MODE == 'cloud' else 'local — nothing leaves ✅'}")
    st.divider()
    if Config.RENDER_PROVIDER == "local" and Config.LLM_MODE == "local" and Config.VISION_MODE == "local":
        st.success("Fully local — nothing leaves the machine.")
    else:
        st.info(egress.summary())
        st.caption("Original image, materials, branding & client details never leave.")
    st.divider()
    st.caption("Refine loop: the agent scores its own render and re-renders up to "
               f"{Config.MAX_REFINES}× until it clears {Config.MIN_SCORE}/10.")

# ---------------- input ----------------
left, right = st.columns([1, 1])
with left:
    up = st.file_uploader("Model image / sketch", type=["png", "jpg", "jpeg", "webp"])
    request = st.text_area(
        "Describe the look",
        "photorealistic, warm dusk light, glass façade, landscaped foreground, people for scale",
        height=90,
    )
    go = st.button("✨ Render", type="primary", disabled=up is None)
with right:
    if up is not None:
        st.image(up, caption="Uploaded model (stays local)", use_container_width=True)

# ---------------- run ----------------
if go and up is not None:
    Config.ensure_dirs()
    in_path = Config.INPUT_DIR / up.name
    in_path.write_bytes(up.getbuffer())

    st.markdown("**Live agent activity**")
    feed = st.container(height=240)
    feed_area = feed.empty()
    steps = []

    def on_step(msg):
        steps.append(msg)
        feed_area.code("\n".join(steps[-200:]), language="text")

    try:
        with st.spinner("Rendering… (redact → structure map → interpret → render → critique)"):
            result = run_agent(str(in_path), request, on_step=on_step)

        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**What left the machine** (faceless structure map)")
            maps = sorted(Config.WORK_DIR.glob("structure_*.png"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if maps:
                st.image(str(maps[0]), use_container_width=True)
            st.caption(f"Critic score: {result.get('score')}/10 · renders tried: {result.get('attempts')}")
        with c2:
            st.markdown("**Final render**")
            outp = result.get("output_path")
            if outp and Path(outp).exists():
                st.image(outp, use_container_width=True)
                st.download_button("⬇️ Download render", Path(outp).read_bytes(),
                                   file_name=Path(outp).name, mime="image/png")
            else:
                st.warning("No render was produced — check the activity log above.")
    except Exception as e:
        st.error("Render failed — full traceback below:")
        st.exception(e)
