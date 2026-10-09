# Luxarc — Architectural Render Agent · Handoff Guide

Turn an architect's **3D-model screenshot / massing / wireframe** into a
**photorealistic, client-ready render** — automatically, and **privacy-first**.

You feed in a rough, untextured model view + a plain-English request
("make it dusk, Scandinavian, add trees") and it paints a realistic image that
keeps the building's exact geometry. In cloud mode, **only a faceless "structure
map" (line/depth) + your text prompt ever leave the machine** — the original
design, materials, branding and client details never do.

---

## 1. What it does (the purpose)

```
INPUT                         →   Luxarc   →   OUTPUT
a rough 3D model / massing /       (renders)    a photorealistic,
wireframe (grey, no materials,                  client-ready image
straight from SketchUp/Revit/Rhino)
```

Normally an architect spends **hours in V-Ray/Lumion** (or pays a visualizer) to
turn a model into a nice render. Luxarc does it in **seconds, automatically**.

> It is meant for **unfinished model views**, not real photos. Rendering a real
> photo is pointless — the input should be a model with *shape but no realism*.

---

## 2. How it works (the agent)

It's a self-correcting **LangGraph** agent — 6 steps in a loop:

```
1 REDACT      (local)  blur any logos / addresses / title-block text        ← privacy
2 STRUCTURE   (local)  make the "faceless map": Canny line map (or depth)   ← shape only
3 INTERPRET   (Groq)   turn your words into a render prompt + settings      ← text only
4 RENDER      (GPU)    diffusion + ControlNet paints the shape realistically
5 CRITIQUE    (local)  a vision model scores its own render; if weak → back to 4 (max ~2)
6 DELIVER     (local)  upscale + save to output/
```

- **Steps 1, 2, 5, 6 run on your machine.** Only steps 3 (text) and 4 (the
  structure map) ever touch the cloud, and only in cloud mode.
- **ControlNet** is what makes the render keep *your* building's shape instead of
  inventing a new one — it's conditioned on the structure map.

---

## 3. Privacy model (important)

Rendering on a cloud GPU means *some* image must leave — but **which** image is up
to you:

| You send… | Cloud sees | Leak |
|---|---|---|
| the **structure map** (faceless line/depth) | only the **shape** | 🟡 minimal — no materials/branding/identity |
| the **full model screenshot** | the whole design | 🔴 everything |

**Always send the structure map, not the original.** Then only the *geometry*
leaves; materials, colours, branding and client identity stay on the machine.

- `LLM_MODE=cloud` → Groq writes the SQL-of-prompts (**text only** leaves).
- `RENDER_PROVIDER=hf|fal` → the **structure map + prompt** leave to render.
- `VISION_MODE=local` (default) → the render is scored **on-device**; nothing leaves.
- **For a truly secret design → run everything local** (needs your own GPU). That
  is the only zero-leak option.

The `work/` folder always shows you the exact structure map that was sent, so you
can verify what left.

---

## 4. Setup

1. **Install Python 3.10+**, then in this folder:
   ```
   pip install -r requirements.txt
   ```
   For the optional text-redaction, also install the **Tesseract** engine
   (Windows: the UB-Mannheim build) — otherwise redaction is skipped, harmlessly.

2. **Create your `.env`** (this file is NOT shipped — it holds secret keys):
   - Rename **`.env.example` → `.env`**
   - Fill in your **own** free keys:
     - **Groq** (the text "director"): https://console.groq.com  → API Keys → Create
     - **Hugging Face** (a render option): https://huggingface.co/settings/tokens
       → *Fine-grained* token, **Inference** preset

3. **Pick a render backend** in `.env` (see §6). Default is `hf`.

---

## 5. How to run

**Web UI (recommended):**
```
python -m streamlit run app.py
```
Upload a model image → type the look you want → Render. It shows the faceless
structure map (what left) next to the final render, with a download button.

**Command line:**
```
python run_render.py path\to\model.png "warm dusk light, glass facade, add trees"
# keep only the building, drop surroundings:
python run_render.py path\to\model.png "..." --crop 100 80 900 720   # left top right bottom
```

- Final render → **`output/`**
- Structure map + redacted image (what would leave) → **`work/`**

---

## 6. Configuration (`.env` flags)

| Flag | Options | Meaning |
|---|---|---|
| `LLM_MODE` | `cloud` / `local` | director brain: Groq (fast) or local Ollama (private) |
| `VISION_MODE` | `local` / `cloud` | critic: local = render never leaves (keep this `local`) |
| `RENDER_PROVIDER` | `local` / `hf` / `fal` | where the image is rendered (see below) |
| `RENDER_FALLBACK` | `local` / `none` | if the primary render fails, fall back here |
| `STRUCTURE_TYPE` | `line` / `depth` | the faceless map type (line = Canny, free/local) |
| `REDACT_TEXT` | `1` / `0` | blur logos/addresses before mapping (needs Tesseract) |
| `MAX_REFINES` | number | extra re-renders if the critic isn't happy |
| `MIN_SCORE` | 0–10 | accept a render scoring at/above this |

### Render backends
- **`local`** — SD 1.5 + ControlNet on your GPU. **Free, fully private**, but needs
  a real **GPU + ~6 GB free RAM**; downloads ~5 GB of models on first run.
  (On a CPU-only / low-RAM machine it will be very slow or crash — see §7.)
- **`hf`** — Hugging Face Inference (free tier, rate-limited). *Note: HF's free
  serverless often doesn't support ControlNet cleanly — you may need to adjust the
  model, or use a free **HF Space** in the browser instead.*
- **`fal`** — fal.ai: fast, real ControlNet, **free signup credits then paid**
  (~$0.02–0.05/render). Set `FAL_API_KEY` and `RENDER_PROVIDER=fal`.

> **Groq cannot render images** — it only does the *text* interpret and (optionally)
> the *vision* critic. The actual render always uses local / HF / fal.

### Free ways to render (no paid API)
- **Local** on a GPU machine (free forever, private).
- **Hugging Face Spaces (ZeroGPU)** — free GPU demos in the browser: upload the
  **structure map** from `work/`, add your prompt, download the render. Manual, but
  free and keeps materials/branding private.

---

## 7. Hardware & troubleshooting

- **Rendering needs a GPU.** Diffusion models are built for NVIDIA GPUs (CUDA).
  On a **CPU-only** machine each render takes many minutes; on a machine with **low
  free RAM** (< ~6 GB) the model load can **crash (segfault)**. This is the main
  gotcha.
- **Intel/AMD integrated GPUs** are *not* used by standard PyTorch — they need
  special experimental builds and usually aren't worth it. Prefer NVIDIA or cloud.
- **Model download stalls?** Set your `HF_TOKEN` (same as `HF_API_KEY`) for faster,
  authenticated downloads, and **don't restart repeatedly** (that trips HF rate
  limits) — let one download run to completion.
- **Everything except the render works on any machine** — redact, structure map,
  and interpret are light and reliable. Only the final "paint" step needs a GPU.

---

## 8. File map

```
app.py                 Streamlit web UI
run_render.py          command-line entry
requirements.txt       dependencies
.env.example           template — copy to .env and add YOUR keys
render-agent-architecture.png/.svg   the architecture diagram
render_agent/
  config.py            all settings / flags
  redact.py            step 1 — blur identifying marks (OCR)
  structure.py         step 2 — faceless line/depth map (OpenCV)
  interpret.py         step 3 — words → render settings (Groq/Ollama)
  render_provider.py   step 4 — diffusion + ControlNet (local/hf/fal + fallback)
  critique.py          step 5 — vision model scores the render
  upscale.py           step 6 — enlarge for delivery
  graph.py             the LangGraph agent + refine loop
  providers.py         LLM factories (Groq / Ollama, text + vision)
  egress.py            states/logs exactly what leaves the machine
  logging.py           the live step feed
input/                 put model images here
output/                finished renders land here
work/                  intermediate artifacts (redacted image + structure map)
```

---

## TL;DR
1. `pip install -r requirements.txt`
2. `.env.example` → `.env`, add your free Groq + HF keys
3. `python -m streamlit run app.py`
4. Upload a **model view** (not a photo), describe the look, Render.
5. Rendering needs a **GPU** — on a weak machine, use a **free HF Space** with the
   structure map instead. Only the shape ever leaves; the design stays private.
