PROMPT_VERSION = "p1"
INSTRUCTION = (
    "You read one road image. Decide whether the main target is a vehicle license plate or a road sign.\n"
    "Plate: output only the registration characters exactly as printed. Omit state or province names, slogans, "
    "web addresses and dealer frames. For Chinese plates keep the leading province character and letter, e.g. 京A·12345.\n"
    "Sign: output all printed words and numbers top to bottom, joined by single spaces. Do not add units or words "
    "that are not printed.\n"
    "Copy characters literally; do not correct spelling. Reply in exactly two lines:\nKIND: plate or sign\nTEXT: <text>"
)
