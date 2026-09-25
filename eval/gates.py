"""Session-1 gates: unpacked-base proxy, GPU identity, BF16 correctness, HF throughput. Prints one JSON line."""
import json, shutil, subprocess, time

def sh(c):
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout.strip()

g = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
# Proxy for the mandated base's unpacked size: used bytes on the root overlay, excluding the persistent mount
# and our own additions. Authoritative number comes later from `docker image inspect` (Task 14).
g["root_used_bytes_excl_workspace"] = int(sh("timeout 600 du -sxb / --exclude=/workspace --exclude=/proc "
                                             "--exclude=/sys --exclude=/tmp --exclude=/root/.cache 2>/dev/null | cut -f1") or 0)
g["gfx"] = sh("rocminfo | grep -m1 -o 'gfx[0-9a-f]*'")
g["gpu"] = sh("amd-smi static --asic | grep -m1 MARKET_NAME | cut -d: -f2").strip()
g["vram"] = sh("amd-smi static --vram | grep -m1 -i size | cut -d: -f2").strip()
import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
ref = a.float() @ b.float()
g["bf16_matmul_rel_err"] = float(((a @ b).float() - ref).abs().max() / ref.abs().max())
g["bf16_ok"] = g["bf16_matmul_rel_err"] < 2e-2
g["torch"], g["hip"] = torch.__version__, torch.version.hip
t = time.time()
n = sh("timeout 60 curl -sL -o /dev/null -w '%{size_download}' $HF_ENDPOINT/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors")
g["hf_bytes_per_s"] = int(float(n or 0) / max(1e-6, time.time() - t))
g["workspace_free_bytes"] = shutil.disk_usage("/workspace").free
GiB = 2**30
g["headroom_bytes"] = 60 * GiB - g["root_used_bytes_excl_workspace"] - 3 * GiB
g["tier"] = "bake_9b" if g["headroom_bytes"] >= 20e9 else "bake_4b_or_download_9b" if g["headroom_bytes"] >= 9e9 else "download_at_startup"
print(json.dumps(g))
