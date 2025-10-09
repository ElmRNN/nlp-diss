import re, yaml
from typing import List

MASK = "<DOT>"

def load_abbrev(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    lst = [a.strip() for a in data.get("abbrev", []) if a and isinstance(a, str)]
    return sorted(set(lst), key=len, reverse=True)

def _mask_abbrev(text: str, abbr: List[str]) -> str:
    if not abbr: return text
    pat = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in abbr) + r")\b")
    return pat.sub(lambda m: m.group(0).replace(".", MASK), text)

def _mask_decimals(text: str) -> str:
    return re.sub(r"(?<=\d)\.(?=\d)", MASK, text)

def _mask_ellipses(text: str) -> str:
    return text.replace("...", MASK*3).replace("…", MASK*3)

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")

def safe_split_sentences(text: str, abbr: List[str]) -> list[str]:
    if not text: return []
    t = _mask_abbrev(text, abbr)
    t = _mask_decimals(t)
    t = _mask_ellipses(t)
    parts = _SPLIT.split(t)
    return [p.replace(MASK, ".").strip() for p in parts if p.strip()]
