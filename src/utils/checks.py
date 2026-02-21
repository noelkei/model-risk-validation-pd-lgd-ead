# src/utils/checks.py
from typing import List
from src.config import LEAKAGE_PATTERNS

def leakage_candidates(columns: List[str]) -> List[str]:
    flagged = []
    for c in columns:
        cl = c.lower()
        if any(p in cl for p in LEAKAGE_PATTERNS):
            flagged.append(c)
    return sorted(set(flagged))