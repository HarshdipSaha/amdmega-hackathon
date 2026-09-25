"""Decode without throwing away evidence: EXIF transpose, first TIFF frame, RGB, aspect-safe views."""
from pathlib import Path
from PIL import Image, ImageOps

def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        im.seek(0)                                   # multipage TIFF: first frame (spec §G provisional policy)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            im = im.point(lambda v: v / 256).convert("L")
        if im.mode in ("RGBA", "LA", "P", "PA"):
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
        return im.convert("RGB")

def bounded(im: Image.Image, max_pixels: int) -> Image.Image:
    w, h = im.size
    if w * h <= max_pixels:
        return im
    s = (max_pixels / (w * h)) ** 0.5
    return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)

def upscale(im: Image.Image, factor: float, max_pixels: int) -> Image.Image:
    w, h = im.size
    f = min(factor, (max_pixels / (w * h)) ** 0.5)
    return im if f <= 1 else im.resize((int(w * f), int(h * f)), Image.Resampling.LANCZOS)

def reread_view(im: Image.Image, first_pixels: int, reread_pixels: int) -> Image.Image:
    """A genuinely different, higher-detail view for the single escalation."""
    if im.width * im.height > first_pixels:
        return bounded(im, reread_pixels)            # we downscaled the first view: give back original pixels
    return upscale(im, 2.0, reread_pixels)           # already full-res: enlarge small characters

def center_crop(im: Image.Image, frac: float) -> Image.Image:
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch))
