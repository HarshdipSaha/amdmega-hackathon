from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import requests
from fetch_evidence_import import capture, ROOT, TAG

URLS = {
    "pytorch-hip-2-13": "https://docs.pytorch.org/docs/2.13/notes/hip.html",
    "pytorch-sdpa-2-13": "https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html",
    "docker-engine-api-spec": "https://docs.docker.com/reference/api/engine/version/v1.53.yaml",
    "docker-image-spec-current": "https://raw.githubusercontent.com/moby/docker-image-spec/main/spec.md",
    "oci-image-config": "https://raw.githubusercontent.com/opencontainers/image-spec/main/config.md",
    "vllm-pypi": "https://pypi.org/pypi/vllm/json",
    "transformers-pypi": "https://pypi.org/pypi/transformers/json",
    "pillow-formats": "https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html",
    "pillow-imageops": "https://pillow.readthedocs.io/en/stable/reference/ImageOps.html",
}

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(capture, URLS.items()))
    manifest = json.loads((ROOT / "docker-registry-exact-tag.response.json").read_text(encoding="utf-8"))
    digest = manifest["config"]["digest"]
    url = f"https://registry-1.docker.io/v2/rocm/pytorch/blobs/{digest}"
    info = {"id": "docker-registry-config", "url": url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(), "digest": digest}
    auth = requests.get("https://auth.docker.io/token", params={
        "service": "registry.docker.io", "scope": "repository:rocm/pytorch:pull"}, timeout=25)
    response = requests.get(url, headers={"Authorization": f"Bearer {auth.json()['token']}"}, timeout=30)
    info.update(status=response.status_code, response_bytes=len(response.content))
    (ROOT / "docker-registry-config.response.json").write_text(response.text, encoding="utf-8")
    (ROOT / "docker-registry-config.metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)
