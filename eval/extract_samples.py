"""Extract the 10 brief sample images (in order) into eval/samples/ with gold answers. Smoke/conventions only."""
import io, json, sys
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

PDF = Path(__file__).resolve().parent.parent / "LabLab_AMD AI Challenge - Mini Challenge.pdf"
OUT = Path(__file__).resolve().parent / "samples"
SAMPLES = [  # (file name, gold, slice, degradation) — brief pp. 9-13
    ("image_01.png", "7ABC123", "us_plate", "clean"),
    ("image_02.png", "京A·12345", "cn_plate", "clean"),
    ("image_03.jpg", "JHT 2951", "us_plate", "angled"),
    ("image_04.png", "5XYZ891", "us_plate", "motion_blur"),
    ("image_05.jpg", "沪B·88888", "cn_plate", "low_light_glare"),
    ("image_06.png", "STOP", "word_sign", "clean"),
    ("image_07.tiff", "STOP", "word_sign", "noise"),
    ("image_08.jpg", "SPEED LIMIT 65", "speed_sign", "clean"),
    ("image_09.png", "ROAD WORK AHEAD", "warning_sign", "clean"),
    ("image_10.tiff", "35", "advisory_plaque", "clean"),
]

def main() -> int:
    doc = fitz.open(PDF)
    xrefs = [x[0] for page in doc for x in page.get_images(full=True)]
    if len(xrefs) != len(SAMPLES):
        print(f"expected {len(SAMPLES)} images, found {len(xrefs)}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for xref, (name, gold, sl, deg) in zip(xrefs, SAMPLES):
        im = Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("RGB")
        im.save(OUT / name)
        rows.append({"image": name, "gold": gold, "slice": sl, "degradation": deg, "source": "brief pp.9-13"})
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} samples to {OUT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
