"""
legacy_sentiment_model.py
-------------------
Thin wrapper around the trained ML sentiment classifier
(models/sentiment_model.joblib, produced by scripts/train_sentiment_model.py).

If the model file hasn't been trained yet, this module falls back to a
small lexicon-based scorer so the app still runs end-to-end (with a
clear warning shown in the UI to run the training script for the real
ML-based results).
"""

import os
import re
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, 'models', 'sentiment_model.joblib')

_model = None
_model_load_attempted = False


def _load_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    if os.path.exists(MODEL_PATH):
        import joblib
        _model = joblib.load(MODEL_PATH)
    return _model


def is_ml_model_available() -> bool:
    return _load_model() is not None


# --- fallback lexicon scorer (used only if the ML model isn't trained yet) --
_POS_WORDS = set("good great awesome amazing happy love thanks thank glad nice "
                  "excited proud wonderful fantastic best relieved fun grateful "
                  "congrats congratulations yay super perfect".split())
_NEG_WORDS = set("bad sad angry hate frustrated frustrating annoyed annoying "
                  "worst terrible horrible upset disappointed disappointing "
                  "stressed stress tired exhausted regret unacceptable furious "
                  "hopeless boring rude unfair".split())


def _lexicon_predict(text: str) -> str:
    words = set(re.findall(r"[a-z']+", text.lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    if pos > neg:
        return 'positive'
    if neg > pos:
        return 'negative'
    return 'neutral'


def predict_sentiment(text: str) -> str:
    """Predict sentiment label ('positive' / 'negative' / 'neutral') for one message."""
    if not text or not text.strip():
        return 'neutral'
    model = _load_model()
    if model is not None:
        return model.predict([text])[0]
    return _lexicon_predict(text)


def predict_sentiments_bulk(messages) -> list:
    model = _load_model()
    messages = list(messages)
    if model is not None and len(messages) > 0:
        return list(model.predict(messages))
    return [_lexicon_predict(m) for m in messages]


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Given a DataFrame with a 'message' column, returns sentiment counts."""
    labels = predict_sentiments_bulk(df['message'].tolist())
    out = pd.Series(labels).value_counts().reindex(['positive', 'neutral', 'negative']).fillna(0).astype(int)
    return out
