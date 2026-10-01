"""FakeEngine script for the starter-kit corpus. TEST FIXTURE ONLY: never imported by sourcebound/.

The replies imitate what a reader model plausibly returns, including its mistakes (citing the withdrawn
datasheet, over-citing the log, offering a near-miss unit cost), so the contract test proves that the
deterministic gates, not the script, produce the right answers and citation sets.
"""
import json
import sys
from pathlib import Path

TRANSCRIPTS = {
    "backplane_pinout.png": "TQ-40 backplane connector - pin assignment\nB11: GND\nB12: SDA\nB13: SCL\nB14: THERM_ALERT#\n"
                            "B15: PRSNT#\nB16: GND\nTHERM_ALERT# is asserted low when the die exceeds the warning threshold.\n"
                            "IMAGE: connector pin assignment diagram",
    "asset_label.jpg": "Orrery Systems\nMODEL: TQ-40\nBOARD REVISION: REV-C2\nSN: OS4-118823-77\nMADE IN MALAYSIA\n"
                       "IMAGE: photo of an asset label",
}


def _r(answer, *ev, status="answered"):
    return {"status": status, "answer": answer, "answer_type": "extracted",
            "evidence": [{"file": f, "quote": q, "role": role} for f, q, role in ev], "lookup": []}


R1 = "specs/tq40_datasheet_r1_WITHDRAWN.pdf"
R2 = "specs/tq40_datasheet_r2.pdf"
CSV = "support/bug_database.csv"
LOG = "logs/prod_inference_2026-09-02.log"
ROW = "ticket: ORR-1847 | component: power | severity: S2 | summary: Die temperature sensor reports high under sustained small-batch load | status: closed | fixed_in: 4.3.2"

REPLIES = [
    ("maximum junction temperature", [_r("105", (R1, "Maximum junction temperature .......... 105 C", "value")),
                                      _r("94 C", (R2, "Maximum junction temperature .......... 94 C", "value"))]),
    ("customer sampling", _r("Q3", ("planning/roadmap_fy27.docx", "TQ-60 enters customer sampling in Q3 FY27.", "value"))),
    ("fan assembly", _r("ORR-FAN-2214-B", ("support/rma_parts.xlsx", "part_number: ORR-FAN-2214-B | description: Fan assembly, dual-rotor, field replaceable | compatible_with: TQ-40", "value"))),
    ("fixed ticket ORR-1847", _r("4.3.2", (CSV, ROW, "value"), (LOG, "incident logged against ORR-1847", "link"))),
    ("error code", _r("E7731", (LOG, "ERROR E7731: thermal throttle engaged on die 0", "value"))),
    ("batch timeout", _r("180 seconds", ("engineering/ingest_service.py", "DEFAULT_BATCH_TIMEOUT_S = 180", "value"))),
    ("THERM_ALERT#", _r("B14", ("specs/backplane_pinout.png", "B14: THERM_ALERT#", "value"))),
    ("board revision", _r("C2", ("support/asset_label.jpg", "BOARD REVISION: REV-C2", "value"))),
    ("production log shows", _r("4.3.2", (CSV, ROW, "value"), (LOG, "incident logged against ORR-1847", "link"),
                                ("engineering/meridian_release_notes.txt", "4.3.1  Fixes a rare hang", "link"))),
    ("unit price", [_r("84.5", ("support/rma_parts.xlsx", "unit_cost_usd: 84.5", "value")), _r("", status="not_found")]),
]


def warrant(question, prompt):
    return "NO" if "unit price" in question.lower() else "YES"


def dump(dest: Path) -> None:
    """Write the JSON files the worker reads with SB_ENGINE=fake (used by the CI contract job)."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "replies.json").write_text(json.dumps(REPLIES), encoding="utf-8")
    (dest / "transcripts.json").write_text(json.dumps(TRANSCRIPTS), encoding="utf-8")


if __name__ == "__main__":
    dump(Path(sys.argv[1]))
