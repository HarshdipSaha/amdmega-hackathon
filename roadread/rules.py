"""Narrow, test-pinned transcription rules (spec §5). No step may shorten a run of identical characters."""
from dataclasses import dataclass
import re

PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
_CN = re.compile(rf"^([{PROVINCES}])([A-Z])([·.\- ]?)([A-Z0-9]+)(.*)$")
_DOTS = str.maketrans({"•": "·", "･": "·", "‧": "·", "∙": "·", "⋅": "·", "●": "·", "\u3000": " "})
_US_BANNERS = sorted({
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA",
    "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE",
    "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS",
    "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
    "THE LONE STAR STATE", "EMPIRE STATE", "EXCELSIOR", "SUNSHINE STATE", "THE GOLDEN STATE", "LAND OF LINCOLN",
    "GARDEN STATE", "KEYSTONE STATE", "GREAT LAKES STATE", "PURE MICHIGAN", "GRAND CANYON STATE",
    "FAMOUS POTATOES", "LIVE FREE OR DIE", "FIRST IN FLIGHT", "THE OCEAN STATE", "DMV.CA.GOV",
}, key=len, reverse=True)

@dataclass
class Rule:
    text: str
    suspect: bool = False

def canonicalize(s: str) -> str:
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)   # full-width ASCII
    s = s.translate(_DOTS)
    s = re.sub(r'([·.\-])\s+', r'\1', s)
    s = re.sub(r'\s+([·.\-])', r'\1', s)
    return " ".join(s.split())

def is_chinese_plate(s: str) -> bool:
    return bool(_CN.match(s.strip()))

def chinese_format_ok(s: str) -> bool:
    m = _CN.match(s.strip())
    if not m or m.group(5):
        return False
    serial = m.group(4)
    return len(serial) == 5 or (len(serial) == 6 and (serial[0] in "DF" or serial[-1] in "DF"))

def _map_serial_io(s: str) -> str:
    m = _CN.match(s)
    serial = m.group(4).replace("I", "1").replace("O", "0")
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{serial}{m.group(5)}"

def _strip_banners(s: str) -> str:
    out = f" {s.upper()} "
    for b in _US_BANNERS:
        out = re.sub(rf"(?<=\s){re.escape(b)}(?=\s)", " ", out)
    out = " ".join(out.split())
    plausible = bool(out) and (any(ch.isdigit() for ch in out) or re.fullmatch(r"[A-Z0-9]{2,8}", out))
    return out if plausible else s

def postprocess(raw: str, kind: str) -> Rule:
    text = canonicalize(raw)
    upper = text.upper()
    if is_chinese_plate(upper):                     # content wins over the model's KIND label
        text = _map_serial_io(upper)
        return Rule(text, suspect=not chinese_format_ok(text))
    if kind != "plate":
        return Rule(text)
    return Rule(_strip_banners(text))
