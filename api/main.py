from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.preprocess import clean_text

app = FastAPI(title="Fake News Detection API")


UI_HTML = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Fake News Detector</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link
            href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap"
            rel="stylesheet"
        />
        <style>
            :root {
                --bg: #f7f4ee;
                --card: #ffffff;
                --ink: #141414;
                --muted: #575757;
                --accent: #1e6f5c;
                --accent-2: #f6b73c;
                --border: #e7e1d8;
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                color: var(--ink);
                background: radial-gradient(circle at 10% 10%, #fff2d9, var(--bg));
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 32px 18px;
            }

            .wrap {
                width: min(980px, 100%);
                display: grid;
                gap: 24px;
            }

            .header {
                display: grid;
                gap: 8px;
            }

            .kicker {
                text-transform: uppercase;
                letter-spacing: 0.2em;
                font-size: 12px;
                color: var(--muted);
            }

            h1 {
                margin: 0;
                font-size: clamp(28px, 4vw, 44px);
                line-height: 1.1;
            }

            .sub {
                margin: 0;
                color: var(--muted);
                font-size: 16px;
            }

            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 22px;
                box-shadow: 0 16px 40px rgba(20, 20, 20, 0.08);
                display: grid;
                gap: 16px;
                animation: floatIn 0.6s ease-out;
            }

            textarea {
                width: 100%;
                min-height: 170px;
                resize: vertical;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px;
                font-size: 15px;
                font-family: inherit;
            }

            .actions {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
            }

            button {
                border: none;
                border-radius: 999px;
                padding: 12px 20px;
                font-weight: 700;
                font-size: 14px;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .primary {
                background: var(--accent);
                color: #ffffff;
                box-shadow: 0 10px 20px rgba(30, 111, 92, 0.28);
            }

            .secondary {
                background: #fef4de;
                color: var(--ink);
                border: 1px solid var(--border);
            }

            button:hover {
                transform: translateY(-1px);
            }

            .result {
                display: grid;
                gap: 6px;
                padding: 14px;
                border-radius: 12px;
                background: #f6f6f6;
                border: 1px dashed var(--border);
                min-height: 70px;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                font-weight: 700;
                font-size: 14px;
            }

            .dot {
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: var(--accent-2);
            }

            .meta {
                color: var(--muted);
                font-size: 13px;
            }

            @keyframes floatIn {
                from {
                    opacity: 0;
                    transform: translateY(12px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @media (max-width: 720px) {
                .card {
                    padding: 16px;
                }
            }
        </style>
    </head>
    <body>
        <main class="wrap">
            <section class="header">
                <div class="kicker">MLOps demo</div>
                <h1>Fake News Detector</h1>
                <p class="sub">
                    Paste a headline or article. The model predicts Fake or Real with confidence.
                </p>
            </section>
            <section class="card">
                <label for="news">News text</label>
                <textarea
                    id="news"
                    placeholder="Paste news text here..."
                ></textarea>
                <div class="actions">
                    <button class="primary" id="predict">Predict</button>
                    <button class="secondary" id="sample">Use sample</button>
                    <button class="secondary" id="clear">Clear</button>
                </div>
                <div class="result" id="result">
                    <div class="badge"><span class="dot"></span>Awaiting input</div>
                    <div class="meta">Submit text to see the prediction.</div>
                </div>
            </section>
        </main>

        <script>
            const newsInput = document.getElementById("news");
            const result = document.getElementById("result");
            const predictBtn = document.getElementById("predict");
            const sampleBtn = document.getElementById("sample");
            const clearBtn = document.getElementById("clear");

            sampleBtn.addEventListener("click", () => {
                newsInput.value =
                    "Government officials released a detailed report this morning on the economic outlook for the next quarter.";
            });

            clearBtn.addEventListener("click", () => {
                newsInput.value = "";
                result.innerHTML =
                    '<div class="badge"><span class="dot"></span>Awaiting input</div>' +
                    '<div class="meta">Submit text to see the prediction.</div>';
            });

            predictBtn.addEventListener("click", async () => {
                const text = newsInput.value.trim();
                if (!text) {
                    result.innerHTML =
                        '<div class="badge"><span class="dot"></span>No text provided</div>' +
                        '<div class="meta">Please paste some news text.</div>';
                    return;
                }

                result.innerHTML =
                    '<div class="badge"><span class="dot"></span>Predicting...</div>' +
                    '<div class="meta">Model is analyzing the text.</div>';

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ news: text }),
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        result.innerHTML =
                            '<div class="badge"><span class="dot"></span>Error</div>' +
                            `<div class="meta">${error.detail || "Request failed"}</div>`;
                        return;
                    }

                    const data = await response.json();
                    const confidence = Math.round(data.confidence * 100);
                    result.innerHTML =
                        `<div class="badge"><span class="dot"></span>${data.prediction}</div>` +
                        `<div class="meta">Confidence: ${confidence}%</div>`;
                } catch (error) {
                    result.innerHTML =
                        '<div class="badge"><span class="dot"></span>Error</div>' +
                        '<div class="meta">Unable to reach the API.</div>';
                }
            });
        </script>
    </body>
</html>
"""


class PredictRequest(BaseModel):
    news: str = Field(..., min_length=3, max_length=10000)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


def _load_artifacts(model_dir: Path) -> Dict[str, Any]:
    model_path = model_dir / "model.pkl"
    vectorizer_path = model_dir / "vectorizer.pkl"
    labels_path = model_dir / "labels.json"

    missing = [
        str(path)
        for path in (model_path, vectorizer_path, labels_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    return {"model": model, "vectorizer": vectorizer, "labels": labels}


@app.on_event("startup")
def startup() -> None:
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    try:
        artifacts = _load_artifacts(model_dir)
        app.state.model = artifacts["model"]
        app.state.vectorizer = artifacts["vectorizer"]
        app.state.labels = artifacts["labels"]
        app.state.model_dir = str(model_dir)
        app.state.confidence_threshold = float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.75")
        )
    except Exception as exc:
        app.state.model_error = str(exc)


@app.get("/health")
def health() -> Dict[str, str]:
    if hasattr(app.state, "model"):
        return {
            "status": "ok",
            "model_dir": app.state.model_dir,
            "confidence_threshold": str(app.state.confidence_threshold),
        }
    return {"status": "error", "detail": getattr(app.state, "model_error", "unknown")}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_HTML


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail=getattr(app.state, "model_error", "Model not loaded"),
        )

    cleaned = clean_text(payload.news)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing.")

    vector = app.state.vectorizer.transform([cleaned])
    prediction = int(app.state.model.predict(vector)[0])
    label = app.state.labels.get(str(prediction), str(prediction))

    confidence = 0.5
    if hasattr(app.state.model, "predict_proba"):
        probabilities = app.state.model.predict_proba(vector)[0]
        confidence = float(max(probabilities))
        if confidence < app.state.confidence_threshold:
            label = "Uncertain"

    return PredictResponse(prediction=label, confidence=round(confidence, 4))
