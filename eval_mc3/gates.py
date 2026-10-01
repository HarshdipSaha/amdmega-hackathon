"""MC3 first-session gates (spec Further Notes G2-G4). Prints one JSON line. Run on the GPU notebook."""
import importlib
import json
import os
import shutil
import subprocess
import time


def sh(c: str) -> tuple[int, str]:
    r = subprocess.run(["bash", "-lc", c], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


g = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
import torch  # noqa: E402

g["torch"], g["torch_file"], g["hip"] = torch.__version__, torch.__file__, torch.version.hip
g["torch_is_rocm"] = "rocm" in torch.__version__
g["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
g["gfx"] = sh("rocminfo | grep -m1 -o 'gfx[0-9a-f]*'")[1]
for mod in ("transformers", "tokenizers", "safetensors", "fitz", "openpyxl", "numpy", "PIL"):
    try:
        m = importlib.import_module(mod)
        g[f"{mod}_version"] = getattr(m, "__version__", getattr(m, "VersionBind", "?"))
    except Exception as e:  # noqa: BLE001
        g[f"{mod}_version"] = f"IMPORT FAILED: {e}"
g["unshare_n"] = sh("unshare -n true")[0] == 0
g["unshare_rn"] = sh("unshare -rn true")[0] == 0
NODAC = "setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search"
g["setpriv_drop_dac"] = sh(f"{NODAC} true")[0] == 0
g["capsh"] = shutil.which("capsh") is not None
probe = "/tmp/sb_dac_probe.txt"
open(probe, "w").write("x")
os.chmod(probe, 0)
g["root_reads_mode_000"] = sh(f"cat {probe}")[0] == 0
g["dac_drop_blocks_read"] = sh(f"{NODAC} cat {probe}")[0] != 0
g["workspace_free_gb"] = round(shutil.disk_usage("/workspace").free / 1e9, 1)
g["root_free_gb"] = round(shutil.disk_usage("/").free / 1e9, 1)
for slot in ("reader", "embedder"):
    p = f"/workspace/models/mc3-{slot}"
    g[f"{slot}_path"] = os.path.realpath(p) if os.path.exists(p) else None
    g[f"{slot}_bytes"] = int(sh(f"du -sbL {p} | cut -f1")[1] or 0) if os.path.exists(p) else 0
print(json.dumps(g))
