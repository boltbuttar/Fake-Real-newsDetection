import pandas as pd

from src.train import train_model


def test_training_outputs(tmp_path):
    data_path = tmp_path / "news.csv"
    df = pd.DataFrame(
        {
            "text": [
                "real story about economy",
                "real story about science",
                "fake rumor about economy",
                "fake rumor about science",
            ],
            "label": ["REAL", "REAL", "FAKE", "FAKE"],
        }
    )
    df.to_csv(data_path, index=False)

    model_dir = tmp_path / "models"
    metrics = train_model(
        data_path=data_path,
        model_dir=model_dir,
        test_size=0.5,
        random_state=1,
        track_mlflow=False,
    )

    assert (model_dir / "model.pkl").exists()
    assert (model_dir / "vectorizer.pkl").exists()
    assert (model_dir / "labels.json").exists()
    assert 0.0 <= metrics["accuracy"] <= 1.0
