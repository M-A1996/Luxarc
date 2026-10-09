"""
Privacy knob #1 — redact identifying marks BEFORE we make the structure map.

Auto-blurs text regions (title blocks, logos-with-text, addresses, client names,
sheet labels) using OCR, so even the abstract map carries no identifying marks.
Optional crop keeps only the building and drops neighbouring real buildings.

All local. If Tesseract/OCR isn't installed, it degrades gracefully (returns the
image unblurred with a warning) — the structure map still strips materials/colour.
"""
from PIL import Image, ImageFilter

from render_agent.config import Config
from render_agent.logging import log


def _blur_boxes(img: Image.Image, boxes) -> Image.Image:
    out = img.convert("RGB")
    for (x, y, w, h) in boxes:
        if w <= 0 or h <= 0:
            continue
        region = out.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(12))
        out.paste(region, (x, y))
    return out


def _detect_text_boxes(img: Image.Image):
    """Return [(x,y,w,h)] of likely text regions via Tesseract, or [] if unavailable."""
    try:
        import pytesseract
    except Exception:
        log("[yellow]redact: pytesseract not installed — skipping text blur "
            "(install Tesseract + `pip install pytesseract` to enable).[/yellow]")
        return []
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        log(f"[yellow]redact: OCR failed ({e}) — skipping text blur.[/yellow]")
        return []
    boxes = []
    for i, txt in enumerate(data.get("text", [])):
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1
        if txt and txt.strip() and conf >= 40:
            pad = 4
            boxes.append((data["left"][i] - pad, data["top"][i] - pad,
                          data["width"][i] + 2 * pad, data["height"][i] + 2 * pad))
    return boxes


def redact(image_path, crop_box=None) -> Image.Image:
    """Load the image, optionally crop to the building, and blur detected text.
    crop_box: optional (left, top, right, bottom) to keep only the building."""
    img = Image.open(image_path).convert("RGB")
    if crop_box:
        img = img.crop(crop_box)
        log(f"[cyan]redact: cropped to {crop_box} (drops surroundings).[/cyan]")
    if Config.REDACT_TEXT:
        boxes = _detect_text_boxes(img)
        if boxes:
            img = _blur_boxes(img, boxes)
            log(f"[green]redact: blurred {len(boxes)} text region(s) (titles/logos/addresses).[/green]")
        else:
            log("[cyan]redact: no text regions blurred.[/cyan]")
    return img
