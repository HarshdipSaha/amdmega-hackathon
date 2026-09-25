"""Reproducible discovery using the requested DuckDuckGo fallback sequence."""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

ROOT = Path(__file__).parent / "web-search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def search(query, region="wt-wt", limit=7):
    ROOT.mkdir(exist_ok=True)
    key = hashlib.sha256(f"text|{region}|{query}".encode()).hexdigest()[:16]
    path = ROOT / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    report = {"query": query, "mode": "text", "region": region,
              "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "attempts": [], "results": []}
    try:
        rows = list(DDGS(timeout=15).text(query, region=region, max_results=limit, backend="duckduckgo"))
        report["attempts"].append({"endpoint": "ddgs.text(backend=duckduckgo)",
                                   "http_status": None, "status_note": "Library does not expose HTTP status",
                                   "outcome": "success" if rows else "empty"})
        if rows:
            report["results"] = rows
            report["source"] = "DuckDuckGo via ddgs"
    except Exception as error:
        report["attempts"].append({"endpoint": "ddgs.text(backend=duckduckgo)",
                                   "http_status": None, "outcome": "error", "error": str(error)[:600]})
    if not report["results"]:
        endpoints = ["https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/", "https://www.bing.com/search"]
        for endpoint in endpoints:
            time.sleep(2)
            try:
                response = requests.get(endpoint, params={"q": query}, headers=HEADERS, timeout=20)
                attempt = {"endpoint": endpoint, "url": response.url, "http_status": response.status_code}
                report["attempts"].append(attempt)
                if response.status_code != 200:
                    attempt["outcome"] = "failed_http"
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                rows = []
                if "bing.com" in endpoint:
                    nodes = soup.select("li.b_algo")
                    for node in nodes:
                        link, snippet = node.select_one("h2 a"), node.select_one(".b_caption p")
                        if link:
                            rows.append({"title": link.get_text(" ", strip=True), "href": link.get("href"),
                                         "body": snippet.get_text(" ", strip=True) if snippet else ""})
                else:
                    nodes = soup.select(".result")
                    for node in nodes:
                        link, snippet = node.select_one(".result__a"), node.select_one(".result__snippet")
                        if link:
                            href = link.get("href", "")
                            href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
                            rows.append({"title": link.get_text(" ", strip=True), "href": href,
                                         "body": snippet.get_text(" ", strip=True) if snippet else ""})
                    if not rows:
                        for link in soup.select("a.result-link"):
                            href = link.get("href", "")
                            href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
                            rows.append({"title": link.get_text(" ", strip=True), "href": href, "body": ""})
                attempt["outcome"] = "success" if rows else "empty_or_blocked"
                if rows:
                    report["results"] = rows[:limit]
                    report["source"] = "Bing HTML (non-DDG fallback)" if "bing.com" in endpoint else "DuckDuckGo direct HTML"
                    break
            except Exception as error:
                report["attempts"].append({"endpoint": endpoint, "http_status": None, "outcome": "error", "error": str(error)[:600]})
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--limit", type=int, default=7)
    args = parser.parse_args()
    for index, query in enumerate(args.queries):
        if index:
            time.sleep(2)
        result = search(query, limit=args.limit)
        shown = dict(result)
        shown["results"] = [dict(row, body=row.get("body", "")[:450]) for row in result["results"]]
        print(json.dumps(shown, ensure_ascii=False), flush=True)
