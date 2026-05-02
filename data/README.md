Place your dataset CSV in this folder.
Default path used by the trainer: data/news.csv

Expected columns:
- text: the news content
- label: FAKE or REAL (case-insensitive) or 0/1

Build data/news.csv from local or Kaggle:

python -m src.download_data --source-dir . --output data/news.csv

If Fake.csv and True.csv are not present, the helper downloads from Kaggle.
KaggleHub requires Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY
or configure ~/.kaggle/kaggle.json before running.
