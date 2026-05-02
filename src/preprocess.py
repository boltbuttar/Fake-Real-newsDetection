from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

try:
    from nltk.corpus import stopwords
except Exception:
    stopwords = None

_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


@lru_cache(maxsize=1)
def _get_stopwords() -> set[str]:
    if stopwords is None:
        return _FALLBACK_STOPWORDS
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return _FALLBACK_STOPWORDS


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    stopwords_set = _get_stopwords()
    tokens = [t for t in text.split() if t and t not in stopwords_set]
    return " ".join(tokens)


def clean_texts(texts: Iterable[str]) -> List[str]:
    return [clean_text(t) for t in texts]
