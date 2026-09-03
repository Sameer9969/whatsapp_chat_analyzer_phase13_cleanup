"""
train_sentiment_model.py
-------------------------
Trains the ML sentiment classifier used by the app and saves it to
models/sentiment_model.joblib.

Pipeline: TF-IDF vectorizer -> Logistic Regression (multi-class:
positive / negative / neutral).

Run this once before starting the app:
    python scripts/train_sentiment_model.py

The bundled dataset (data/sentiment_train.csv) is a small, hand-curated
set of chat-style sentences. Swap in a larger/real-world labeled dataset
here for better accuracy -- the training code doesn't need to change.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PROJECT_ROOT, 'data', 'sentiment_train.csv')
MODEL_DIR = os.path.join(_PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.joblib')


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ('clf', LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("Validation accuracy: {:.2f}%".format(accuracy_score(y_test, preds) * 100))
    print(classification_report(y_test, preds))

    # Refit on ALL data before saving, so the shipped model uses every
    # labeled example available.
    pipeline.fit(df['text'], df['label'])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model -> {MODEL_PATH}")


if __name__ == '__main__':
    main()
