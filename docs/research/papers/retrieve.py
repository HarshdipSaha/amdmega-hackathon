"""Cached, sequential primary-source retrieval for this literature study."""
import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AMD-OCR-literature-study/1.0 (individual research; sequential cached requests)"})
LOG = ROOT / "retrieval-log.jsonl"


def retrieve(url, target, params=None):
    target = ROOT / target
    if target.exists():
        return target.read_bytes()
    response = SESSION.get(url, params=params, timeout=40)
    record = {"url": response.url, "status": response.status_code, "retrieved_at": datetime.now(timezone.utc).isoformat(), "path": str(target.name)}
    with LOG.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if response.ok:
        target.write_bytes(response.content)
    else:
        (ROOT / (target.name + ".error.txt")).write_text(response.text, encoding="utf-8")
    response.raise_for_status()
    time.sleep(3.2)
    return response.content


def search(query, limit=12, newest=False):
    key = hashlib.sha256(query.encode()).hexdigest()[:12]
    params = {"search_query": query, "start": 0, "max_results": limit, "sortBy": "submittedDate" if newest else "relevance", "sortOrder": "descending"}
    data = retrieve("https://export.arxiv.org/api/query", f"search-{key}.xml", params)
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    rows = []
    for entry in root.findall("atom:entry", ns):
        rows.append({
            "id": entry.findtext("atom:id", "", ns),
            "title": " ".join(entry.findtext("atom:title", "", ns).split()),
            "summary": " ".join(entry.findtext("atom:summary", "", ns).split()),
            "authors": [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)],
            "published": entry.findtext("atom:published", "", ns),
            "updated": entry.findtext("atom:updated", "", ns),
            "categories": [a.get("term") for a in entry.findall("atom:category", ns)],
            "pdf_url": next((a.get("href") for a in entry.findall("atom:link", ns) if a.get("title") == "pdf"), None),
        })
    (ROOT / f"search-{key}.json").write_text(json.dumps({"query": query, "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--newest", action="store_true")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(search(args.query, args.limit, args.newest), ensure_ascii=False, indent=2))
