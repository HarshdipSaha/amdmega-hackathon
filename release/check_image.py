"""Release checks with crane, no Docker daemon (spec Acceptance checks): base-layer identity, uncompressed size,
config (no entrypoint; CMD = supervisor; env), and no obvious secrets. Prints one JSON line; exits 1 on any failure.

    MC3_IMAGE=docker.io/<user>/<repo>:<tag> python release/check_image.py [--base-bytes N]

--base-bytes skips re-streaming the 29 GiB base when its uncompressed size was already measured (gate G1).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zlib

BASE = "docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
LIMIT = 60 * 2**30
PLAT = ["--platform", "linux/amd64"]


def crane(*args: str) -> str:
    return subprocess.run(["crane", *args], check=True, capture_output=True, text=True).stdout


def layers(ref: str) -> list[dict]:
    return json.loads(crane("manifest", *PLAT, ref))["layers"]


def uncompressed_bytes(ref: str, digest: str) -> int:
    repo = ref.rsplit(":", 1)[0] if "@" not in ref else ref.split("@")[0]
    p = subprocess.Popen(["crane", "blob", f"{repo}@{digest}"], stdout=subprocess.PIPE)
    d, n = zlib.decompressobj(16 + zlib.MAX_WBITS), 0
    for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
        n += len(d.decompress(chunk))
    n += len(d.flush())
    if p.wait() != 0:
        raise RuntimeError(f"crane blob failed for {digest}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-bytes", type=int)
    a = ap.parse_args()
    img = os.environ["MC3_IMAGE"]
    base_l, img_l = layers(BASE), layers(img)
    r: dict = {"base_layers": len(base_l), "image_layers": len(img_l)}
    r["base_prefix_ok"] = [x["digest"] for x in img_l[:len(base_l)]] == [x["digest"] for x in base_l]
    ours = img_l[len(base_l):]
    r["our_layer_gib"] = [round(x["size"] / 2**30, 2) for x in ours]
    r["largest_layer_ok"] = max((x["size"] for x in ours), default=0) <= 5 * 2**30
    base_bytes = a.base_bytes or sum(uncompressed_bytes(BASE, x["digest"]) for x in base_l)
    our_bytes = sum(uncompressed_bytes(img, x["digest"]) for x in ours)
    r["base_uncompressed_gib"] = round(base_bytes / 2**30, 3)
    r["total_uncompressed_gib"] = round((base_bytes + our_bytes) / 2**30, 3)
    r["size_ok"] = base_bytes + our_bytes <= LIMIT - 2 * 2**30
    cfg = json.loads(crane("config", *PLAT, img))["config"]
    r["entrypoint"], r["cmd"], r["workdir"] = cfg.get("Entrypoint"), cfg.get("Cmd"), cfg.get("WorkingDir")
    env = dict(e.split("=", 1) for e in cfg.get("Env", []))
    r["env_ok"] = env.get("HF_HUB_OFFLINE") == "1" and "/app/pylib" in env.get("PYTHONPATH", "")
    r["entrypoint_ok"] = not cfg.get("Entrypoint") and cfg.get("Cmd") == ["python3", "-m", "sourcebound.supervisor"]
    r["no_secrets"] = not re.search(r"(TOKEN|SECRET|PASSWORD|API_KEY)=", json.dumps(cfg.get("Env", [])), re.I)
    ok = all(r[k] for k in ("base_prefix_ok", "largest_layer_ok", "size_ok", "env_ok", "entrypoint_ok", "no_secrets"))
    r["ok"] = ok
    print(json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
