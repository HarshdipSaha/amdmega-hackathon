"""G2: determine whether a local ROCm environment exposes a GPU target."""
from __future__ import annotations

import re
import shutil
import subprocess


AGENT_SPLIT = re.compile(r"\n(?=\*{3,}\s*\nAgent )")
NAME_RE = re.compile(r"^\s*Name:\s*(\S+)", re.MULTILINE)
TYPE_RE = re.compile(r"^\s*Device Type:\s*(\w+)", re.MULTILINE)


def parse_rocminfo(text: str) -> list[str]:
    """Return GPU `gfx` targets in rocminfo agent order."""
    targets: list[str] = []
    for block in AGENT_SPLIT.split(text):
        device_type = TYPE_RE.search(block)
        if not device_type or device_type.group(1).upper() != "GPU":
            continue
        target = next((name for name in NAME_RE.findall(block) if name.startswith("gfx")), None)
        if target:
            targets.append(target)
    return targets


def main() -> int:
    if shutil.which("rocminfo") is None:
        print("[g2] VERDICT: BLOCKED — rocminfo is unavailable; use an AMD cloud GPU.")
        return 1
    output = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False).stdout
    targets = parse_rocminfo(output)
    print(f"[g2] GPU agents: {targets or 'none'}")
    if not targets:
        print("[g2] VERDICT: BLOCKED — ROCm sees no GPU agent.")
        return 1
    print("[g2] VERDICT: OK — local GPU visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
