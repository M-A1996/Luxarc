# Luxarc — Architectural Render Agent

*Lux (light) + arc (architecture).* Turn an architect's **3D-model screenshot +
a plain-English request** into a **client-ready render** — free-capable and **privacy-first**.

In cloud mode, only a **faceless structure map** (line/depth) + a **text prompt** ever
leave the machine. The original image, materials, colours, context, logos, addresses
and client names **never leave**.

> **Status: work in progress.** The agent pipeline (redact → structure map → interpret → critique loop) is built and runs end to end on the text/structure steps. The cloud render step (fal.ai ControlNet) and the local vision critic are being wired up and tested; local rendering needs an NVIDIA GPU. Expect rough edges — see the roadmap below.

## The agent (self-correcting loop)
```
redact -> structure map -> interpret -> render -> critique
                               ▲                      │
                               └──── refine (if score low) ──┘
                                                      │ good enough
                                                   deliver (upscale + save)
```
- **redact** (local) — OCR-blur title blocks / logos / addresses      ← privacy knob #1
- **structure map** (local) — Canny line map (or depth): shape only
- **interpret** (Groq LLM, free tier — TEXT only) — request → prompt/settings
- **render** (HF / fal / local) — paints the shape; **only the map + prompt leave**  ← knob #2 (no-train provider)
- **critique** (local vision, free) — scores its own render; re-renders if weak
- **deliver** — upscale + save

## What leaves the machine
| Step | Cloud | Leaves? |
|---|---|---|
| interpret | Groq (text) | your words only |
| render | HF / fal | structure map + prompt only |
| critique | **local** (default) | **nothing** |
Original image • materials • context • logos • client details → **never leave**.

## Install
```bash
pip install -r requirements.txt
# for redaction, also install the Tesseract engine (Windows: UB-Mannheim build)
cp .env.example .env      # then fill in keys (all free)
```
Free keys: **Groq** (console.groq.com) · **Hugging Face** (huggingface.co/settings/tokens).

## Run
```bash
python run_render.py path\to\model.png "make it dusk, Scandinavian, add trees"

# keep only the building (drop surroundings): --crop left top right bottom
python run_render.py path\to\model.png "warm daylight, glass facade" --crop 100 80 900 720
```
- Final render → **`output/`**
- **What actually left the machine** (redacted image + structure map) → **`work/`** — inspect these to verify the privacy claim.

## Free vs private (pick your mode in .env)
| Goal | Settings |
|---|---|
| **Fully private, free** (slow) | `RENDER_PROVIDER=local` · `LLM_MODE=local` · `VISION_MODE=local` |
| **Fast + private-ish, free-ish** (default) | `RENDER_PROVIDER=hf` · `LLM_MODE=cloud` (Groq) · `VISION_MODE=local` |

## Notes
- **Groq is the brain, not the renderer.** Groq (an open LLM such as gpt-oss-20b) does the *text* interpret and (optionally) the *vision* critique for free — it **cannot generate images**, so the render always uses HF / fal / local diffusion.
- **Truly-free *unlimited* cloud image gen doesn't exist** — HF free tier is rate-limited; fal gives credits then charges; local is free but needs a GPU (slow on an Intel iGPU).
- Cloud render providers' exact model ids / argument names vary — adjust in `.env` / `render_provider.py` to a model the provider currently serves. The **local** path is the reference implementation of true ControlNet.

## Roadmap
- [ ] End-to-end cloud render via fal.ai ControlNet
- [ ] Local vision critic (Ollama) scoring renders in the refine loop
- [ ] Before/after examples in this README
- [ ] Small eval set to check the building's geometry is preserved
- [ ] Hosted demo
