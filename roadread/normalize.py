"""Evaluator normalization (brief p.7). Used only for scoring/comparison, never to rewrite predictions."""
import re

_DROP = re.compile(r"[\s\-\.·_]")

def normalize(text: str) -> str:
    return _DROP.sub("", text.upper())

def matches(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)
