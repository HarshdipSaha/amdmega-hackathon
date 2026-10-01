"""Seeded generator of MC3-style corpora with every trap of the sample kit, but new names and values.

    python eval_mc3/generate_corpus.py --out eval_mc3/data/dev --seeds 101 102 103 104
    python eval_mc3/generate_corpus.py --out eval_mc3/data/holdout --seeds 901 902

Each <out>/c<seed>/ holds: corpus/, questions.json (sample-questions.json format plus an "oracle" per question),
transcripts.json (ground-truth image text, for FakeEngine runs) and hazards.json (files to chmod 000).
Nothing from the starter-kit answers is reused. Spec: Testing Decisions, "Evaluation corpora".
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

COMPANIES = ["Halvard Dynamics", "Brightwater Silicon", "Cobalt Reach Systems", "Nimbus Forge", "Tessellate Labs",
             "Quillon Devices", "Verdant Logic", "Ostrava Compute", "Kestrel Microworks", "Lumen Arc"]
PREFIXES = ["HX", "BW", "CR", "NF", "TL", "QD", "VL", "OC", "KM", "LA"]
CODENAMES = ["Basalt", "Cinder", "Drift", "Ember", "Fjord", "Garnet", "Heron", "Ibis", "Juniper", "Kelp"]
FIRMWARE = ["Aurora", "Beacon", "Corvid", "Dynamo", "Eclipse", "Fathom", "Gossamer", "Halcyon"]
SERVICES = [("ingest", "batch timeout", "BATCH_TIMEOUT_S"), ("export", "upload timeout", "UPLOAD_TIMEOUT_S"),
            ("scheduler", "lease timeout", "LEASE_TIMEOUT_S"), ("telemetry", "flush interval", "FLUSH_INTERVAL_S")]
SIGNALS = ["THERM_ALERT#", "FAN_TACH", "PWR_GOOD", "SMB_ALERT#", "RESET_N", "PRSNT#"]
GENERIC = ["GND", "SDA", "SCL", "3V3_AUX", "GND", "12V_SENSE"]


# ------------------------------------------------------------------ writers
def write_pdf(path: Path, pages: list[list[str]], user_pw: str | None = None) -> None:
    import fitz
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for ln in lines:
            page.insert_text((60, y), ln, fontsize=11, fontname="helv")
            y += 18
    if user_pw:
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw=user_pw + "-owner", user_pw=user_pw)
    else:
        doc.save(path)


def write_docx(path: Path, paragraphs: list[str], table: list[list[str]] | None = None) -> None:
    def p(t):
        return f'<w:p><w:r><w:t xml:space="preserve">{escape(t)}</w:t></w:r></w:p>'
    body = "".join(p(t) for t in paragraphs)
    if table:
        body += "<w:tbl>" + "".join("<w:tr>" + "".join(f"<w:tc>{p(c)}</w:tc>" for c in r) + "</w:tr>" for r in table) + "</w:tbl>"
    body += p("End of document.")
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)


def write_xlsx(path: Path, sheets: dict[str, list[list]]) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)


def _font(size):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def draw_pinout(path: Path, title: str, pins: list[tuple[str, str]], note: str) -> None:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1000, 460), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), title, fill="black", font=_font(26))
    d.rectangle((40, 90, 960, 300), outline="black", width=3)
    for i, (pin, sig) in enumerate(pins):
        x = 60 + i * 150
        d.rectangle((x, 120, x + 118, 190), outline="black", width=2)
        d.text((x + 38, 142), pin, fill="black", font=_font(22))
        d.line((x + 59, 190, x + 59, 212), fill="black", width=2)
        d.text((x + 10, 220), sig, fill="black", font=_font(14))
    d.text((40, 330), note, fill="black", font=_font(17))
    im.save(path)


def draw_label(path: Path, lines: list[str], rng: random.Random) -> None:
    from PIL import Image, ImageDraw, ImageFilter
    im = Image.new("RGB", (760, 440), (232, 233, 228))
    lab = Image.new("RGB", (640, 320), "white")
    d = ImageDraw.Draw(lab)
    d.rectangle((0, 0, 639, 319), outline="black", width=4)
    for i, ln in enumerate(lines):
        d.text((28, 30 + i * 52), ln, fill="black", font=_font(26 if i else 28))
    lab = lab.rotate(rng.uniform(-5, 5), expand=True, fillcolor=(232, 233, 228))
    im.paste(lab, (40, 40))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    px = im.load()
    for _ in range(4000):
        x, y = rng.randrange(im.width), rng.randrange(im.height)
        v = rng.randrange(150, 255)
        px[x, y] = (v, v, v)
    im.save(path, quality=rng.randint(55, 75))


# ------------------------------------------------------------------ world and corpus
def generate(out: Path, seed: int) -> dict:
    rng = random.Random(seed)
    k = rng.randrange(len(COMPANIES))
    company, px = COMPANIES[k], PREFIXES[k]
    base = rng.randint(3, 7) * 10
    cur, nxt, old = f"{px}-{base}", f"{px}-{base + 20}", f"{px}-{base - 20}"
    code = rng.choice(CODENAMES)
    fw = rng.choice(FIRMWARE)
    tk = f"{px}{rng.choice('DQX')}"
    yy = rng.randint(27, 29)
    corpus = out / "corpus"
    for d in ("archive", "engineering", "logs", "planning", "specs", "support", "vendor", "ops"):
        (corpus / d).mkdir(parents=True, exist_ok=True)
    q: list[dict] = []
    transcripts: dict[str, str] = {}

    def add(query, answer, cites, category, oracle):
        q.append({"n": len(q) + 1, "query": query, "expected_answer": answer, "expected_citations": cites,
                  "category": category, "oracle": oracle})

    # --- datasheets: r1 withdrawn, r2 current, three pages, bandwidth only on page 3
    tj1, tj2 = rng.randint(100, 110), rng.randint(85, 97)
    bw1, bw2 = round(rng.uniform(2.0, 3.0), 1), round(rng.uniform(3.2, 5.0), 1)
    power = rng.randint(300, 450)
    slug = cur.lower().replace("-", "")
    r1, r2 = f"specs/{slug}_datasheet_r1_WITHDRAWN.pdf", f"specs/{slug}_datasheet_r2.pdf"
    write_pdf(corpus / r1, [[f'{company} {cur} "{code}" Accelerator', "Datasheet, revision 1 - WITHDRAWN, SUPERSEDED BY REVISION 2",
                             "Preliminary silicon. Do not design to these numbers.", "Electrical and thermal",
                             f"Maximum junction temperature .......... {tj1} C", f"Board power (TBP) ..................... {power - 25} W"],
                            ["Memory", f"Peak bandwidth ........................ {bw1} TB/s"]])
    write_pdf(corpus / r2, [[f'{company} {cur} "{code}" Accelerator', "Datasheet, revision 2 - supersedes revision 1",
                             "Electrical and thermal", f"Maximum junction temperature .......... {tj2} C",
                             f"Board power (TBP) ..................... {power} W"],
                            ["Mechanical", "Form factor ........................... FHFL dual slot", "Weight ................................ 1.9 kg"],
                            ["Memory subsystem", f"Peak bandwidth ........................ {bw2} TB/s",
                             "Revision history", f"r2  Junction temperature corrected to {tj2} C.", "r1  Initial release, preliminary silicon."]])
    add(f"What is the maximum junction temperature of the {cur}?", str(tj2), [r2], "pdf, superseded-document trap",
        [{"file": r2, "quote": f"Maximum junction temperature .......... {tj2} C", "role": "value"}])
    add(f"What is the peak memory bandwidth of the {cur}, in TB/s?", str(bw2), [r2], "pdf, page 3 of a multi-page datasheet",
        [{"file": r2, "quote": f"Peak bandwidth ........................ {bw2} TB/s", "role": "value"}])

    # --- roadmap docx: paragraph + table
    qs, qv = rng.randint(1, 4), rng.randint(1, 4)
    road = f"planning/roadmap_fy{yy}.docx"
    write_docx(corpus / road, [f"{company} - Accelerator Roadmap (Internal)", "Dates are targets, not commitments.",
                               f"{cur} is in volume production. Remaining work is firmware only.",
                               f"{nxt} enters customer sampling in Q{qs} FY{yy}. It doubles memory bandwidth.",
                               f"{old} reaches end of life in Q{rng.randint(1, 4)} FY{yy}."],
               [["product", "milestone", "quarter"], [nxt, "tape-out", f"Q{rng.randint(1, 4)} FY{yy - 1}"],
                [nxt, "volume production", f"Q{qv} FY{yy + 1}"], [cur, "volume production", f"Q{rng.randint(1, 4)} FY{yy - 2}"]])
    add(f"In which quarter does the {nxt} enter customer sampling?", f"Q{qs} FY{yy}", [road], "docx, paragraph",
        [{"file": road, "quote": f"{nxt} enters customer sampling in Q{qs} FY{yy}.", "role": "value"}])
    add(f"In which quarter does the {nxt} reach volume production?", f"Q{qv} FY{yy + 1}", [road], "docx, table",
        [{"file": road, "quote": f"product: {nxt} | milestone: volume production | quarter: Q{qv} FY{yy + 1}", "role": "value"}])

    # --- parts workbook: near-miss rows + a second sheet
    fan = f"{px}-FAN-{rng.randint(1000, 9999)}-{rng.choice('ABC')}"
    fan_old = f"{px}-FAN-{rng.randint(1000, 9999)}-A"
    hsk = f"{px}-HSK-{rng.randint(1000, 9999)}-{rng.choice('ABC')}"
    cbl = f"{px}-CBL-{rng.randint(1000, 9999)}-C"
    cost = round(rng.uniform(40, 140), 2)
    lt_fan, lt_hsk, lt_cbl = rng.randint(10, 40), rng.randint(20, 60), rng.randint(5, 15)
    parts = "support/rma_parts.xlsx"
    write_xlsx(corpus / parts, {
        "parts": [["part_number", "description", "compatible_with", "unit_cost_usd"],
                  [fan, "Fan assembly, dual-rotor, field replaceable", cur, cost],
                  [hsk, "Heatsink, vapour chamber", cur, round(rng.uniform(150, 260), 2)],
                  [fan_old, "Fan assembly, single-rotor (legacy), field replaceable", old, round(rng.uniform(30, 70), 2)],
                  [cbl, "Backplane cable, 400mm", cur, 29.75]],
        "lead_times": [["part_number", "lead_time_days", "supplier"], [fan, lt_fan, "Arcwind"], [hsk, lt_hsk, "Thermacore"],
                       [cbl, lt_cbl, "Linkway"], [fan_old, rng.randint(50, 90), "Arcwind"]]})
    add(f"What is the part number of the field-replaceable fan assembly for the {cur}?", fan, [parts], "xlsx, near-miss row",
        [{"file": parts, "quote": f"part_number: {fan} | description: Fan assembly, dual-rotor, field replaceable | compatible_with: {cur}", "role": "value"}])
    add(f"What is the supplier lead time, in days, for part {fan}?", str(lt_fan), [parts], "xlsx, second sheet",
        [{"file": parts, "quote": f"sheet lead_times | part_number: {fan} | lead_time_days: {lt_fan}", "role": "value"}])

    # --- bug database (target row worded away from the incident vocabulary) + decoy release notes
    tid = f"{tk}-{rng.randint(1200, 1999)}"
    fixv = f"{rng.randint(3, 6)}.{rng.randint(0, 9)}.{rng.randint(2, 7)}"
    a, b, c = (int(x) for x in fixv.split("."))
    bugs = "support/bug_database.csv"
    rows = [["ticket", "component", "severity", "summary", "status", "fixed_in"],
            [tid, "power", "S2", "Die sensor over-reports under sustained small-batch load", "closed", fixv],
            [f"{tk}-{rng.randint(2000, 2999)}", "pcie", "S3", "Doorbell hang under sustained load", "closed", f"{a}.{b}.{c - 1}"],
            [f"{tk}-{rng.randint(3000, 3999)}", "scheduler", "S3", "Occupancy counter drifts after 18h uptime", "open", ""],
            [f"{tk}-{rng.randint(4000, 4999)}", "firmware", "S4", f"Release notes omit ticket list for {fixv}", "open", ""]]
    rng.shuffle(rows[1:])
    with open(corpus / bugs, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    row_text = f"ticket: {tid} | component: power | severity: S2 | summary: Die sensor over-reports under sustained small-batch load | status: closed | fixed_in: {fixv}"
    notes = f"engineering/{fw.lower()}_release_notes.txt"
    (corpus / notes).write_text(f"{fw} firmware - release notes\n\n{a}.{b}.0  Scheduler fairness rework.\n"
                                f"{a}.{b}.{c - 1}  Fixes a rare hang in the doorbell path.\n{a}.{b}.{c + 1}  Packaging only.\n\n"
                                "Fixes are tracked per ticket in the bug database, which is authoritative.\n", encoding="utf-8")
    add(f"Which firmware version fixed ticket {tid}?", fixv, [bugs], "csv, identifier given in the question",
        [{"file": bugs, "quote": row_text, "role": "value"}])

    # --- production log: error code, ticket, replaced part
    err = f"E{rng.randint(1000, 9999)}"
    day = f"2026-{rng.randint(7, 9):02d}-{rng.randint(10, 28):02d}"
    log = f"logs/prod_inference_{day}.log"
    node = f"inference-node-{rng.randint(1, 19):02d}"
    lines = [f"{day}T03:11:04Z {node} {fw.lower()}[2211]: INFO  scheduler: batch {rng.randint(10000, 99999)} accepted",
             f"{day}T03:14:22Z {node} {fw.lower()}[2211]: WARN  thermal: die 0 at {tj2 - 3}C, approaching limit",
             f"{day}T03:14:41Z {node} {fw.lower()}[2211]: ERROR {err}: thermal throttle engaged on die 0, clocks reduced to 60%",
             f"{day}T03:19:03Z {node} {fw.lower()}[2211]: INFO  thermal: throttle released",
             f"{day}T03:19:03Z {node} {fw.lower()}[2211]: INFO  incident logged against {tid}",
             f"{day}T05:40:12Z {node} fieldsvc: INFO  replaced part {hsk} during incident follow-up"]
    (corpus / log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (corpus / "logs" / f"prod_inference_{day[:-2]}{int(day[-2:]) + 1:02d}.log").write_text(
        f"{day}T01:00:00Z {node} {fw.lower()}[2211]: INFO  scheduler: nominal\n", encoding="utf-8")
    add("What error code is logged when the thermal throttle engages?", err, [log], "log, single-hop",
        [{"file": log, "quote": f"ERROR {err}: thermal throttle engaged on die 0", "role": "value"}])
    add("The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?", fixv,
        [log, bugs], "multi-hop log -> csv",
        [{"file": bugs, "quote": row_text, "role": "value"}, {"file": log, "quote": f"incident logged against {tid}", "role": "link"}])
    add("A part was replaced during the logged incident follow-up. What is that part's supplier lead time in days?", str(lt_hsk),
        [log, parts], "multi-hop log -> xlsx second sheet",
        [{"file": parts, "quote": f"sheet lead_times | part_number: {hsk} | lead_time_days: {lt_hsk}", "role": "value"},
         {"file": log, "quote": f"replaced part {hsk}", "role": "link"}])

    # --- source code constant vs. old value in a comment
    svc, thing, const = rng.choice(SERVICES)
    newv, oldv = rng.choice([90, 120, 180, 240, 300]), rng.choice([30, 45, 60])
    py = f"engineering/{svc}_service.py"
    (corpus / py).write_text(f'"""{svc.title()} service for {company} telemetry."""\n\n'
                             f"# Seconds; raised from {oldv} after the backlog incident.\nDEFAULT_{const} = {newv}\n\n"
                             "MAX_INFLIGHT = 12\n\n\ndef run(job, timeout_s=DEFAULT_" + const + "):\n    return job.start(timeout_s)\n",
                             encoding="utf-8")
    add(f"What is the default {thing}, in seconds, in the {svc} service?", str(newv), [py], "python source, constant vs comment",
        [{"file": py, "quote": f"DEFAULT_{const} = {newv}", "role": "value"}])

    # --- images: pinout diagram and an asset label
    sigs = rng.sample(SIGNALS, 2)
    first = rng.randint(1, 3) * 10
    pins = [(f"B{first + i}", GENERIC[i]) for i in range(6)]
    hit = rng.randrange(6)
    pins[hit] = (pins[hit][0], sigs[0])
    png = f"specs/{slug}_backplane_pinout.png"
    draw_pinout(corpus / png, f"{cur} backplane connector - pin assignment", pins,
                f"{sigs[0]} is the only open-drain pin in the row.")
    transcripts[Path(png).name] = "\n".join(f"{p}: {s}" for p, s in pins) + f"\n{sigs[0]} is the only open-drain pin in the row."
    add(f"Which backplane pin carries {sigs[0]} on the {cur}?", pins[hit][0], [png], "png, image-only",
        [{"file": png, "quote": f"{pins[hit][0]}: {sigs[0]}", "role": "value"}])
    rev = f"REV-{rng.choice('BCDEF')}{rng.randint(1, 4)}"
    sn = f"{px}{rng.randint(1, 9)}-{rng.randint(100000, 999999)}-{rng.randint(10, 99)}"
    jpg = "support/asset_label.jpg"
    draw_label(corpus / jpg, [company, f"MODEL  {cur}", f"BOARD REVISION  {rev}", f"SN  {sn}"], rng)
    transcripts[Path(jpg).name] = f"{company}\nMODEL: {cur}\nBOARD REVISION: {rev}\nSN: {sn}"
    add("What board revision is printed on the asset label?", rev, [jpg], "jpg, image-only, qualifier kept",
        [{"file": jpg, "quote": f"BOARD REVISION: {rev}", "role": "value"}])

    # --- hazards and unanswerables
    vol = rng.choice(["5,000", "10,000", "25,000"])
    write_pdf(corpus / "vendor/supplier_agreement_ENCRYPTED.pdf",
              [[f"Supplier agreement - {cur}", f"Unit price at {vol} unit volume ........ {rng.randint(3000, 9000)} USD"]],
              user_pw=f"pw{seed}")
    add(f"What is the unit price of the {cur} at {vol} unit volume?", "", [], "refusal, answer only in an encrypted file", [])
    audit = f"AUD-{rng.randint(100, 999)}"
    (corpus / "vendor/internal_audit.txt").write_text(f"Internal audit findings. Finding reference {audit}.\n", encoding="utf-8")
    add("What is the reference number of the internal audit finding?", "", [], "refusal, answer only in an unreadable file", [])
    add(f"What is the board power of the {nxt}?", "", [], "refusal, attribute exists only for a sibling product", [])
    (corpus / "vendor/telemetry_capture.dat").write_bytes(bytes(range(256)) * 64)
    (corpus / "ops/oncall_runbook.md").write_text(f"# On-call runbook\n\nFor {err}, check fan curves before escalating.\n",
                                                   encoding="utf-8")

    (out / "questions.json").write_text(json.dumps({"challenge": "mc3-synthetic", "seed": seed, "corpus": "corpus",
                                                    "queries": q}, indent=1), encoding="utf-8")
    (out / "transcripts.json").write_text(json.dumps(transcripts, indent=1), encoding="utf-8")
    (out / "hazards.json").write_text(json.dumps({"chmod000": ["vendor/internal_audit.txt"], "empty_dirs": ["archive"]}),
                                      encoding="utf-8")
    return {"seed": seed, "questions": len(q)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    for s in a.seeds:
        print(json.dumps(generate(Path(a.out) / f"c{s}", s)))


if __name__ == "__main__":
    main()
