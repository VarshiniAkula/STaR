import re

FINAL_ANS_PATTERN = re.compile(r"####\s*([\-]?\d+(?:\.\d+)?)")

def parse_final_answer(text: str) -> str | None:
    m = FINAL_ANS_PATTERN.search(text)
    return m.group(1) if m else None
