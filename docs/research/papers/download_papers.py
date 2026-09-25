import json
import sys
from pathlib import Path

import fitz

from retrieve import ROOT, retrieve


for paper_id in sys.argv[1:]:
    try:
        retrieve(f"https://arxiv.org/pdf/{paper_id}", f"{paper_id}.pdf")
        with fitz.open(ROOT / f"{paper_id}.pdf") as doc:
            pages = [f"\n\n===== PAGE {i + 1} =====\n\n" + page.get_text() for i, page in enumerate(doc)]
        (ROOT / f"{paper_id}.txt").write_text("".join(pages), encoding="utf-8")
        print(json.dumps({"id": paper_id, "pages": len(pages), "text_characters": sum(map(len, pages))}), flush=True)
    except Exception as error:
        print(json.dumps({"id": paper_id, "error": str(error)}), flush=True)
