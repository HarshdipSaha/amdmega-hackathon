"""Read-only primary-source capture for the OCR runtime feasibility audit."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
TAG = "rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
URLS = {
    "amd-ocr-vllm": "https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/ocr_vllm.html",
    "vllm-gpu-install": "https://docs.vllm.ai/en/stable/getting_started/installation/gpu/",
    "pytorch-hip": "https://docs.pytorch.org/docs/stable/notes/hip.html",
    "pytorch-sdpa": "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
    "hf-attention": "https://huggingface.co/docs/transformers/en/attention_interface",
    "docker-image-inspect": "https://docs.docker.com/reference/cli/docker/image/inspect/",
    "docker-image-ls": "https://docs.docker.com/reference/cli/docker/image/ls/",
    "docker-storage-drivers": "https://docs.docker.com/engine/storage/drivers/",
    "docker-engine-api": "https://docs.docker.com/reference/api/engine/version/v1.53/",
    "docker-image-spec": "https://raw.githubusercontent.com/moby/docker-image-spec/main/v1.2.md",
    "docker-hub-exact-tag": f"https://hub.docker.com/v2/repositories/rocm/pytorch/tags/{TAG}",
    "docker-hub-control-tags": "https://hub.docker.com/v2/repositories/rocm/pytorch/tags?page_size=10&ordering=last_updated",
}

def capture(item):
    key, url = item
    record = {"id": key, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=(15, 40), headers={"User-Agent": "Mozilla/5.0 ResearchAudit/1.0"})
        record.update(status=response.status_code, final_url=response.url,
                      content_type=response.headers.get("Content-Type"),
                      server_date=response.headers.get("Date"))
        ext = ".json" if "json" in response.headers.get("Content-Type", "") else ".html" if "html" in response.headers.get("Content-Type", "") else ".raw.txt"
        (ROOT / (key + ext)).write_bytes(response.content)
        if "html" in response.headers.get("Content-Type", ""):
            soup = BeautifulSoup(response.text, "html.parser")
            for node in soup(["script", "style", "nav", "footer"]):
                node.decompose()
            main = soup.find("main") or soup.find("article") or soup
            txt = main.get_text("\n", strip=True)
            (ROOT / (key + ".txt")).write_text(txt, encoding="utf-8")
            record["title"] = soup.title.get_text(" ", strip=True) if soup.title else None
        record["response_bytes"] = len(response.content)
    except Exception as exc:
        record["error"] = str(exc)
    (ROOT / (key + ".metadata.json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)
    return record

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(capture, URLS.items()))
    # The token is anonymous, repository-scoped and deliberately never saved.
    manifest_url = f"https://registry-1.docker.io/v2/rocm/pytorch/manifests/{TAG}"
    registry = {"id": "docker-registry-exact-tag", "url": manifest_url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        first = requests.get(manifest_url, timeout=25)
        registry["unauthenticated_status"] = first.status_code
        auth = requests.get("https://auth.docker.io/token", params={
            "service": "registry.docker.io", "scope": "repository:rocm/pytorch:pull"}, timeout=25)
        registry["anonymous_token_status"] = auth.status_code
        token = auth.json().get("token")
        if token:
            manifest = requests.get(manifest_url, timeout=25, headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"})
            registry.update(status=manifest.status_code,
                            server_date=manifest.headers.get("Date"),
                            docker_content_digest=manifest.headers.get("Docker-Content-Digest"))
            (ROOT / "docker-registry-exact-tag.response.json").write_text(manifest.text, encoding="utf-8")
    except Exception as exc:
        registry["error"] = str(exc)
    (ROOT / "docker-registry-exact-tag.metadata.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry), flush=True)
