"""Conservative identity matching for requested and observed attention paths."""
from __future__ import annotations

import re


BACKEND_LOG_RE = re.compile(r"Using (?!the default\b)(.+?) backend", re.IGNORECASE)
BACKEND_ALIASES: dict[str, set[str]] = {
    "ROCM_ATTN": {"rocmattentionbackend", "rocmflashattentionbackend", "rocmflashattention"},
    "ROCM_AITER_FA": {"aiterflashattentionbackend", "rocmaiterflashattentionbackend"},
    "TRITON_ATTN": {"tritonattentionbackend", "tritonattention"},
    "TRITON_MLA": {"tritonmlabackend", "tritonmla"},
    "AITER_MLA": {"aitermlabackend", "aitermla"},
    "FLASH_ATTENTION": {"flashattention", "pytorchsdpbackendflashattention", "sdpbackendflashattention"},
    "MATH": {"math", "pytorchsdpbackendmath", "sdpbackendmath"},
    "EFFICIENT_ATTENTION": {"efficientattention", "pytorchsdpbackendefficientattention", "sdpbackendefficientattention"},
}
ALL_KNOWN_NAMES = {name for names in BACKEND_ALIASES.values() for name in names}
VERSION_RE = re.compile(r"v[123]")


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _strip(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for token in ("backend", "impl", "v1", "v2", "v3"):
            if name.startswith(token) and len(name) > len(token):
                name, changed = name[len(token):], True
            if name.endswith(token) and len(name) > len(token):
                name, changed = name[:-len(token)], True
    return name


def names_agree(left: str, right: str) -> bool:
    """Compare class/log names without equating unknown partial substrings."""
    a, b = normalise(left), normalise(right)
    versions_a, versions_b = set(VERSION_RE.findall(a)), set(VERSION_RE.findall(b))
    if versions_a and versions_b and versions_a != versions_b:
        return False
    a, b = _strip(a), _strip(b)
    if min(len(a), len(b)) < 3:
        return normalise(left) == normalise(right)
    return a.startswith(b) or b.startswith(a)


def backend_matches(requested: str, observed: str) -> bool | None:
    """Return same, different-known, or unknown evidence for a requested path."""
    req, obs = normalise(requested), normalise(observed)
    if req == obs:
        return True
    aliases = BACKEND_ALIASES.get(requested.upper().strip())
    if aliases is None or obs not in ALL_KNOWN_NAMES:
        return None
    return obs in aliases


def observed_backends_in(stderr: str) -> list[str]:
    """Extract distinct backend log names in their first-observed order."""
    return list(dict.fromkeys(match.strip() for match in BACKEND_LOG_RE.findall(stderr or "") if match.strip()))
