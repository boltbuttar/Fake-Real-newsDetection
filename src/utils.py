from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

TEXT_COLUMNS = ("text", "news", "content", "article", "title")
LABEL_COLUMNS = ("label", "class", "target", "y")


def resolve_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        "Dataset must contain a text column named one of: " + ", ".join(TEXT_COLUMNS)
    )


def resolve_label_column(df: pd.DataFrame) -> str:
    for col in LABEL_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        "Dataset must contain a label column named one of: " + ", ".join(LABEL_COLUMNS)
    )


def normalize_label(value: object) -> int:
    if value is None:
        raise ValueError("Label is empty.")

    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"fake", "false", "fa", "f", "0"}:
            return 0
        if label in {"real", "true", "r", "1"}:
            return 1

    if isinstance(value, (int, float, bool)):
        if pd.isna(value):
            raise ValueError("Label is empty.")
        numeric = int(value)
        if numeric in (0, 1):
            return numeric

    raise ValueError(f"Unsupported label: {value}")


def load_dataset(path: str | Path) -> Tuple[List[str], List[int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    text_col = resolve_text_column(df)
    label_col = resolve_label_column(df)

    df = df[[text_col, label_col]].dropna()

    texts = df[text_col].astype(str).tolist()
    labels = [normalize_label(value) for value in df[label_col].tolist()]

    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain both Fake and Real labels.")

    return texts, labels
