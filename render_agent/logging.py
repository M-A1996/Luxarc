"""Minimal logging — a live step feed you can watch (and pass an on_step callback)."""
try:
    from rich import print as _rprint
except Exception:                       # rich optional
    _rprint = print


def log(msg: str):
    _rprint(msg)


def step_logger(on_step=None):
    """Return a step(msg) function that logs and forwards to an optional callback."""
    def step(msg: str):
        log(msg)
        if on_step:
            try:
                on_step(msg)
            except Exception:
                pass
    return step
