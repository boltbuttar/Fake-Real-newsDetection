from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .preprocess import clean_text
from .utils import load_dataset

LABELS = {"0": "Fake News", "1": "Real News"}


def _ensure_stopwords() -> None:
    try:
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
    except Exception:
        pass


def _stratified_sample(
    texts: list[str],
    labels: list[int],
    sample_size: int,
    random_state: int,
) -> tuple[list[str], list[int]]:
    if sample_size >= len(texts):
        return texts, labels

    rng = random.Random(random_state)
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    total = len(texts)
    sampled_indices: list[int] = []
    for label, indices in label_to_indices.items():
        proportion = len(indices) / total
        k = max(1, int(round(sample_size * proportion)))
        k = min(k, len(indices))
        sampled_indices.extend(rng.sample(indices, k))

    if len(sampled_indices) < sample_size:
        sampled_set = set(sampled_indices)
        remaining = [i for i in range(total) if i not in sampled_set]
        extra = rng.sample(
            remaining, min(sample_size - len(sampled_indices), len(remaining))
        )
        sampled_indices.extend(extra)
    elif len(sampled_indices) > sample_size:
        sampled_indices = rng.sample(sampled_indices, sample_size)

    sampled_indices.sort()
    return [texts[i] for i in sampled_indices], [labels[i] for i in sampled_indices]


def train_model(
    data_path: str | Path,
    model_dir: str | Path = "models",
    experiment_name: str = "fake-news-detection",
    run_name: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    track_mlflow: bool = True,
    verbose: bool = True,
    sample_size: int | None = None,
) -> Dict[str, float]:
    start_time = time.perf_counter()
    last_time = start_time

    def log(message: str) -> None:
        nonlocal last_time
        if not verbose:
            return
        now = time.perf_counter()
        step = now - last_time
        total = now - start_time
        print(f"[{total:6.1f}s] {message} (+{step:5.1f}s)")
        last_time = now

    log("Starting training")
    _ensure_stopwords()
    log("Loading dataset")
    texts, labels = load_dataset(data_path)
    if len(texts) < 4:
        raise ValueError("Dataset must contain at least 4 samples.")
    log(f"Loaded {len(texts)} samples")

    if sample_size is not None and sample_size < len(texts):
        log(f"Sampling {sample_size} rows")
        texts, labels = _stratified_sample(texts, labels, sample_size, random_state)
        log(f"Sampled {len(texts)} rows")

    log("Cleaning text")
    cleaned = [clean_text(text) for text in texts]

    log("Splitting train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    log("Vectorizing (fit)")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_vec = vectorizer.fit_transform(X_train)
    log("Vectorizing (transform)")
    X_test_vec = vectorizer.transform(X_test)

    log("Training model")
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train_vec, y_train)

    log("Evaluating")
    preds = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    log("Saving artifacts")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    vectorizer_path = model_dir / "vectorizer.pkl"
    labels_path = model_dir / "labels.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    labels_path.write_text(json.dumps(LABELS, indent=2), encoding="utf-8")

    if track_mlflow:
        log("Logging to MLflow")
        mlflow.set_experiment(experiment_name)
        run_display_name = run_name or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_display_name):
            mlflow.log_params(
                {
                    "test_size": test_size,
                    "random_state": random_state,
                    "max_features": max_features,
                    "ngram_range": str(ngram_range),
                    "model": "LogisticRegression",
                }
            )
            mlflow.log_metrics({"accuracy": accuracy, "f1": f1})
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(vectorizer_path))
            mlflow.log_artifact(str(labels_path))

            log("Training complete")

    return {"accuracy": float(accuracy), "f1": float(f1)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fake news model")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--model-dir", default="models", help="Output directory")
    parser.add_argument(
        "--experiment",
        default="fake-news-detection",
        help="MLflow experiment name",
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Train on a subset of rows (faster)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast training mode (smaller sample, unigrams only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress output",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sample_size = args.sample_size
    ngram_range = (1, 2)
    max_features = args.max_features
    if args.fast:
        ngram_range = (1, 1)
        max_features = min(args.max_features, 5000)
        if sample_size is None:
            sample_size = 10000

    metrics = train_model(
        data_path=args.data,
        model_dir=args.model_dir,
        experiment_name=args.experiment,
        run_name=args.run_name,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=max_features,
        ngram_range=ngram_range,
        verbose=not args.quiet,
        sample_size=sample_size,
    )
    print(f"Training complete. Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
