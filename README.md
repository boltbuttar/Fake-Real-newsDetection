# Fake News Detection MLOps

This project trains a TF-IDF + Logistic Regression model to classify news text and serves predictions via a FastAPI REST API. It includes Docker, CI, and MLflow tracking.

## Project layout

fake-news-mlops/
- data/
- src/
- api/
- models/
- tests/
- .github/workflows/
- Dockerfile
- requirements.txt
- README.md

## Dataset format

Place a CSV file at data/news.csv (or pass a different path to the trainer).
Required columns:
- text column: text (or news, content, article, title)
- label column: label (or class, target, y)

Label values can be:
- FAKE/REAL (case-insensitive), or
- 0/1 (0 = Fake, 1 = Real)

Example:

text,label
"Breaking story about ...",FAKE
"Official report ...",REAL

## Build the dataset

If you already extracted Fake.csv and True.csv into the repo root, run:

python -m src.download_data --source-dir . --output data/news.csv

If the files are not present, the helper will download from Kaggle:

python -m src.download_data --output data/news.csv

KaggleHub requires Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY
or configure ~/.kaggle/kaggle.json before running.

## Train the model

python -m src.train --data data/news.csv

Faster training (smaller sample, unigrams, fewer features):

python -m src.train --data data/news.csv --fast

Or set a custom sample size:

python -m src.train --data data/news.csv --sample-size 10000

Artifacts are saved to models/:
- model.pkl
- vectorizer.pkl
- labels.json

## Run the API

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Example request:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"news\": \"Breaking news text here...\"}"

If your model artifacts live elsewhere, set MODEL_DIR:

MODEL_DIR=/path/to/models uvicorn api.main:app --host 0.0.0.0 --port 8000

## Run tests

pytest -q

## MLflow tracking

By default, training logs metrics and artifacts to ./mlruns.

mlflow ui --backend-store-uri mlruns

Run multiple experiments for MLflow comparison:

python -m src.run_experiments --data data/news.csv

## Docker

Build an image after training so models/ is included in the image:

docker build -t fake-news-app .

docker run -p 8000:8000 fake-news-app

## CI/CD

GitHub Actions workflow runs tests and builds the Docker image on each push and pull request.
See .github/workflows/ci-cd.yml.

## Deployment

Railway or Render can run the Docker image or a Python service. Ensure models/ is present in the image or mounted, and set MODEL_DIR if needed.
