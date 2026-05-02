from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .train import train_model


DEFAULT_RUNS: List[Dict[str, Any]] = [
    {
        "name": "baseline-20k",
        "max_features": 20000,
        "ngram_range": (1, 2),
        "sample_size": None,
    },
    {
        "name": "full-50k",
        "max_features": 50000,
        "ngram_range": (1, 2),
        "sample_size": None,
    },
    {
        "name": "mid-30k-sample",
        "max_features": 30000,
        "ngram_range": (1, 2),
        "sample_size": 30000,
    },
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple training experiments for MLflow tracking"
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--model-dir", default="models", help="Output directory")
    parser.add_argument(
        "--experiment",
        default="fake-news-detection",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress output",
    )
    return parser


def _run_all(
    data_path: Path,
    model_dir: Path,
    experiment_name: str,
    runs: List[Dict[str, Any]],
    quiet: bool,
) -> None:
    for config in runs:
        run_name = str(config.get("name", "run"))
        max_features = int(config.get("max_features", 20000))
        ngram_range = config.get("ngram_range", (1, 2))
        sample_size = config.get("sample_size")

        metrics = train_model(
            data_path=data_path,
            model_dir=model_dir,
            experiment_name=experiment_name,
            run_name=run_name,
            max_features=max_features,
            ngram_range=ngram_range,
            sample_size=sample_size,
            verbose=not quiet,
            track_mlflow=True,
        )

        print(
            "Run {name}: accuracy={accuracy:.4f}, f1={f1:.4f}".format(
                name=run_name,
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
            )
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    _run_all(
        data_path=Path(args.data),
        model_dir=Path(args.model_dir),
        experiment_name=args.experiment,
        runs=DEFAULT_RUNS,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
