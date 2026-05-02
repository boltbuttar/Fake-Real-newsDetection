import json

import joblib
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _create_artifacts(model_dir):
    texts = ["real report about economy", "fake rumor about economy"]
    labels = [1, 0]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(features, labels)

    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    (model_dir / "labels.json").write_text(
        json.dumps({"0": "Fake News", "1": "Real News"}), encoding="utf-8"
    )


def test_predict_endpoint(monkeypatch, tmp_path):
    _create_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    from api.main import app

    client = TestClient(app)
    response = client.post("/predict", json={"news": "Official report from agency"})

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_health_endpoint(monkeypatch, tmp_path):
    _create_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    from api.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
