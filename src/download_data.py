from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASET_ID = "clmentbisaillon/fake-and-real-news-dataset"
FAKE_FILE = "Fake.csv"
TRUE_FILE = "True.csv"


def _build_text_column(df: pd.DataFrame) -> pd.Series:
    if "text" in df.columns and "title" in df.columns:
        return (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    if "text" in df.columns:
        return df["text"].astype(str)
    if "title" in df.columns:
        return df["title"].astype(str)
    return df.astype(str).agg(" ".join, axis=1)


def _resolve_dataset_dir(source_dir: Path | None) -> Path:
    if source_dir is not None:
        candidate = source_dir.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Source directory not found: {candidate}")
        if (candidate / FAKE_FILE).exists() and (candidate / TRUE_FILE).exists():
            return candidate
        raise FileNotFoundError(
            "Expected Fake.csv and True.csv in source directory: " + str(candidate)
        )

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd(), repo_root):
        if (candidate / FAKE_FILE).exists() and (candidate / TRUE_FILE).exists():
            return candidate

    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError(
            "kagglehub is required to download the dataset. "
            "Install it or provide --source-dir with Fake.csv and True.csv."
        ) from exc

    return Path(kagglehub.dataset_download(DATASET_ID))


def prepare_dataset(output_path: Path, source_dir: Path | None = None) -> Path:
    dataset_dir = _resolve_dataset_dir(source_dir)
    fake_path = dataset_dir / FAKE_FILE
    true_path = dataset_dir / TRUE_FILE

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Expected Fake.csv and True.csv in dataset folder: " + str(dataset_dir)
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_text = _build_text_column(fake_df)
    true_text = _build_text_column(true_df)

    combined = pd.DataFrame(
        {
            "text": pd.concat([fake_text, true_text], ignore_index=True),
            "label": ["FAKE"] * len(fake_text) + ["REAL"] * len(true_text),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build data/news.csv from local files or Kaggle"
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Folder containing Fake.csv and True.csv (skips Kaggle download)",
    )
    parser.add_argument(
        "--output",
        default="data/news.csv",
        help="Output CSV path",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = Path(args.output)
    source_dir = Path(args.source_dir) if args.source_dir else None
    result_path = prepare_dataset(output_path, source_dir=source_dir)
    print(f"Wrote dataset to {result_path}")


if __name__ == "__main__":
    main()
