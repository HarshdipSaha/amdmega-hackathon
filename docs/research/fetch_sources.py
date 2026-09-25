"""Cache public primary-source text/metadata without downloading model weights."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent / "sources"


def fetch(url):
    ROOT.mkdir(exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    meta_path = ROOT / f"{key}.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    meta = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=35, headers={"User-Agent": "AMD-Mini-Challenge-Research/1.0"})
        meta.update(http_status=response.status_code, final_url=response.url,
                    content_type=response.headers.get("Content-Type", ""))
        extension = ".json" if "json" in meta["content_type"] else ".txt"
        target = ROOT / f"{key}{extension}"
        if "html" in meta["content_type"]:
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            content = soup.get_text("\n", strip=True)
        else:
            content = response.text
        target.write_text(content, encoding="utf-8")
        meta.update(path=str(target.relative_to(Path(__file__).parent)), chars=len(content))
    except Exception as error:
        meta.update(http_status=None, error=str(error))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as executor:
        for meta in executor.map(fetch, args.urls):
            print(json.dumps(meta), flush=True)
