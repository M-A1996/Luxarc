"""
The agent — a LangGraph loop that ties the steps together and self-corrects.

  redact -> structure -> interpret -> render -> critique
                              ▲                      │
                              └──── refine (if score low, max MAX_REFINES) ──┘
                                                     │ (good enough / out of tries)
                                                  finalize (upscale + save)

run_agent(image_path, request) is the single entry point.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from render_agent.config import Config
from render_agent import redact as redact_mod
from render_agent import structure as structure_mod
from render_agent import interpret as interpret_mod
from render_agent import render_provider
from render_agent import critique as critique_mod
from render_agent import upscale as upscale_mod
from render_agent import egress
from render_agent.logging import step_logger

_STEP = step_logger()          # replaced per-run in run_agent


class State(TypedDict, total=False):
    request: str
    input_path: str
    crop_box: Optional[tuple]
    redacted: object
    structure: object
    settings: dict
    render: object
    score: float
    feedback: str
    attempts: int
    output_path: str
    stamp: str


def n_redact(state: State) -> dict:
    _STEP("1/6 REDACT — blurring identifying marks (titles/logos/addresses)…")
    img = redact_mod.redact(state["input_path"], state.get("crop_box"))
    img.save(Config.WORK_DIR / f"redacted_{state['stamp']}.png")
    return {"redacted": img}


def n_structure(state: State) -> dict:
    _STEP("2/6 STRUCTURE — building the faceless map (this is what leaves in cloud mode)…")
    m = structure_mod.make_structure(state["redacted"])
    m.save(Config.WORK_DIR / f"structure_{state['stamp']}.png")
    return {"structure": m}


def n_interpret(state: State) -> dict:
    _STEP("3/6 INTERPRET — turning your words into render settings…")
    return {"settings": interpret_mod.interpret(state["request"], state.get("feedback", ""))}


def n_render(state: State) -> dict:
    attempts = state.get("attempts", 0) + 1
    _STEP(f"4/6 RENDER — attempt {attempts} ({Config.RENDER_PROVIDER})…")
    img = render_provider.render(state["structure"], state["settings"])
    img.save(Config.WORK_DIR / f"render_{state['stamp']}_a{attempts}.png")
    return {"render": img, "attempts": attempts}


def n_critique(state: State) -> dict:
    _STEP("5/6 CRITIQUE — the agent scores its own render…")
    c = critique_mod.critique(state["render"], state["request"])
    return {"score": c["score"], "feedback": c["feedback"]}


def n_finalize(state: State) -> dict:
    _STEP("6/6 DELIVER — upscaling and saving the final render…")
    final = upscale_mod.upscale(state["render"])
    out = Config.OUTPUT_DIR / f"render_final_{state['stamp']}.png"
    final.save(out)
    _STEP(f"DONE → {out}")
    return {"output_path": str(out)}


def _decide(state: State) -> str:
    if state.get("score", 0) >= Config.MIN_SCORE or state.get("attempts", 0) > Config.MAX_REFINES:
        return "finalize"
    _STEP(f"  refine: score {state.get('score')}/10 < {Config.MIN_SCORE} — re-rendering with feedback…")
    return "refine"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("redact", n_redact), ("structure", n_structure), ("interpret", n_interpret),
                     ("render", n_render), ("critique", n_critique), ("finalize", n_finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "redact")
    g.add_edge("redact", "structure")
    g.add_edge("structure", "interpret")
    g.add_edge("interpret", "render")
    g.add_edge("render", "critique")
    g.add_conditional_edges("critique", _decide, {"finalize": "finalize", "refine": "interpret"})
    g.add_edge("finalize", END)
    return g.compile()


def run_agent(image_path: str, request: str, crop_box: tuple = None, on_step=None) -> dict:
    """Run the full agent on one image + request. Returns {output_path, score, ...}."""
    global _STEP
    _STEP = step_logger(on_step)
    Config.ensure_dirs()
    egress.announce()
    app = build_graph()
    final = app.invoke({
        "request": request, "input_path": image_path, "crop_box": crop_box,
        "attempts": 0, "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    })
    return {"output_path": final.get("output_path"), "score": final.get("score"),
            "attempts": final.get("attempts")}
