"""
sentiment.py
------------------------
Phase 6: Sentiment Analysis Integration.

Wires the Phase 5 ML pipeline (ml_pipeline.py) into reusable,
Streamlit-agnostic functions covering:
    - a registry of the locally-trained classifiers available to choose
      from ("Model Selection")
    - overall sentiment counts/percentages + a distribution chart
    - per-user sentiment breakdown + a user-comparison chart
    - a sentiment-over-time timeline
    - single-message prediction (with per-class confidence, when the
      chosen model supports it)

Everything here runs 100% locally against joblib-saved scikit-learn
pipelines already present on disk. No network calls are made, and no
message text or chat data is ever sent to an external API or service.

Class labels: this project's trained models only ever produce
{'positive', 'negative', 'neutral'} because that is the only label set
present in the labeled training data (see ml_pipeline.py /
train_hinglish_pipeline.py / README.md for dataset provenance). No
additional emotion classes (e.g. "joy", "anger") are invented here --
none of the available training data supports them.

DISCLAIMER (also shown in the UI): sentiment classification is an
ML-based estimate. It is not, and is not claimed to be, always correct
-- see `DISCLAIMER` below.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

__all__ = [
    "DISCLAIMER",
    "available_models",
    "predict_messages",
    "predict_single_with_confidence",
    "overall_sentiment_stats",
    "fig_sentiment_distribution",
    "user_sentiment_stats",
    "fig_user_sentiment_comparison",
    "sentiment_timeline",
    "fig_sentiment_timeline",
]

DISCLAIMER = (
    "Sentiment classification is an ML-based estimate and may be inaccurate "
    "for sarcasm, context-dependent Hinglish, slang, and multi-turn conversations."
)

# Labels are learned from whichever model is selected (via its
# `classes_` attribute) rather than hardcoded, so the code stays honest
# if a future, differently-labeled dataset is ever swapped in. This
# preferred display order is only used when these particular labels are
# present -- any other labels are appended afterwards, sorted.
_PREFERRED_LABEL_ORDER = ["positive", "neutral", "negative"]
_LABEL_COLORS = {"positive": "#25D366", "neutral": "#9CA3AF", "negative": "#F87171"}

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
_MODEL_DIR = os.path.join(_BASE_DIR, "models")

# Candidate model files, in preferred display order. Only the ones that
# actually exist on disk are offered in the UI (see available_models()),
# so this list works whether or not train_hinglish_pipeline.py and/or
# scripts/train_sentiment_model.py have been run yet.
_CANDIDATE_MODELS: List[Tuple[str, str]] = [
    ("Random Forest (Hinglish-aware)", os.path.join(_MODEL_DIR, "hinglish_random_forest.joblib")),
    ("Logistic Regression (Hinglish-aware)", os.path.join(_MODEL_DIR, "hinglish_logistic_regression.joblib")),
    ("Multinomial Naive Bayes (Hinglish-aware)", os.path.join(_MODEL_DIR, "hinglish_multinomial_nb.joblib")),
    ("Legacy Model (English-only, original)", os.path.join(_MODEL_DIR, "sentiment_model.joblib")),
]

#: Sentinel path used to select the keyword-lexicon fallback (no trained
#: model needed) -- kept so the app can still classify messages, with a
#: clear caveat, even before any training script has been run.
LEXICON_FALLBACK_PATH = "__lexicon_fallback__"

_POS_WORDS = set(
    "good great awesome amazing happy love thanks thank glad nice mast badhiya "
    "badiya accha achha khush maza mazedaar excited proud wonderful fantastic "
    "best relieved fun grateful congrats congratulations yay super perfect "
    "sahi jhakas zabardast".split()
)
_NEG_WORDS = set(
    "bad sad angry hate frustrated frustrating annoyed annoying worst terrible "
    "horrible upset disappointed disappointing stressed stress tired exhausted "
    "regret unacceptable furious hopeless boring rude unfair bura ganda ghatiya "
    "bakwas faltu udaas dukhi pareshan tension gussa naraz".split()
)


# ---------------------------------------------------------------------------
# Model registry + loading
# ---------------------------------------------------------------------------

def available_models() -> Dict[str, str]:
    """Returns {display_name: path} for every trained classifier
    currently found on disk, plus the always-available keyword-lexicon
    fallback. Used to populate the "Model Selection" dropdown -- nothing
    is hardcoded to assume a particular model exists."""
    found = {name: path for name, path in _CANDIDATE_MODELS if os.path.exists(path)}
    found["Keyword Lexicon (fallback, no ML model)"] = LEXICON_FALLBACK_PATH
    return found


_pipeline_cache: Dict[str, tuple] = {}


def _load_cached_pipeline(path: str):
    """Loads a joblib pipeline, cached per-path (invalidated if the file
    on disk changes), so re-running predictions across Streamlit reruns
    doesn't re-read a multi-MB model file from disk every time."""
    mtime = os.path.getmtime(path)
    cached = _pipeline_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    pipeline = joblib.load(path)
    _pipeline_cache[path] = (mtime, pipeline)
    return pipeline


def _lexicon_predict_one(text: str) -> str:
    import re
    words = set(re.findall(r"[a-z']+", (text or "").lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_messages(messages: Sequence[str], model_path: str) -> List[str]:
    """Predicts a sentiment label for every message in `messages` using
    the model at `model_path` (or the keyword-lexicon fallback if
    `model_path == LEXICON_FALLBACK_PATH`). Runs entirely locally.

    Returns a plain Python list of label strings (never a numpy array),
    since downstream code (db.py) expects `.count()` to work on it.
    """
    messages = list(messages)
    if not messages:
        return []

    if model_path == LEXICON_FALLBACK_PATH:
        return [_lexicon_predict_one(m) for m in messages]

    pipeline = _load_cached_pipeline(model_path)
    preds = pipeline.predict(messages)
    return [str(p) for p in preds]


def predict_single_with_confidence(
    text: str, model_path: str
) -> Tuple[str, Optional[Dict[str, float]]]:
    """Predicts the sentiment of a single message, returning
    (label, confidence_dict). `confidence_dict` maps each class to its
    predicted probability (rounded), or is None if the chosen model
    doesn't support probability estimates (or the lexicon fallback is
    in use, which has no calibrated confidence).
    """
    if not text or not text.strip():
        return "neutral", None

    if model_path == LEXICON_FALLBACK_PATH:
        return _lexicon_predict_one(text), None

    pipeline = _load_cached_pipeline(model_path)
    label = str(pipeline.predict([text])[0])

    confidence = None
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    if clf is not None and hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        classes = clf.classes_
        confidence = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}

    return label, confidence


# ---------------------------------------------------------------------------
# Label ordering helper
# ---------------------------------------------------------------------------

def _order_labels(labels_present: Sequence[str]) -> List[str]:
    present = set(labels_present)
    ordered = [l for l in _PREFERRED_LABEL_ORDER if l in present]
    remaining = sorted(l for l in present if l not in _PREFERRED_LABEL_ORDER)
    return ordered + remaining


def _color_for(label: str) -> str:
    return _LABEL_COLORS.get(label, "#6366F1")


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, font=dict(size=16, color="#9CA3AF"),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Overall Sentiment
# ---------------------------------------------------------------------------

def overall_sentiment_stats(labels: Sequence[str]) -> dict:
    """Real counts + percentages for a list of predicted sentiment
    labels. Returns {'total', 'label_order', 'counts', 'percentages'}."""
    labels = list(labels)
    total = len(labels)
    if total == 0:
        return {"total": 0, "label_order": [], "counts": {}, "percentages": {}}

    counter = Counter(labels)
    label_order = _order_labels(counter.keys())
    counts = {l: counter[l] for l in label_order}
    percentages = {l: round(counter[l] / total * 100, 2) for l in label_order}
    return {"total": total, "label_order": label_order, "counts": counts, "percentages": percentages}


def fig_sentiment_distribution(labels: Sequence[str]) -> go.Figure:
    """Pie chart of overall sentiment distribution."""
    stats = overall_sentiment_stats(labels)
    if stats["total"] == 0:
        return _empty_figure("No messages to classify")

    order = stats["label_order"]
    fig = px.pie(
        names=[l.capitalize() for l in order],
        values=[stats["counts"][l] for l in order],
        color=[l.capitalize() for l in order],
        color_discrete_map={l.capitalize(): _color_for(l) for l in order},
        title="Overall Sentiment Distribution",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    return fig


# ---------------------------------------------------------------------------
# User Sentiment
# ---------------------------------------------------------------------------

def user_sentiment_stats(df_with_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Per-user sentiment breakdown. `df_with_sentiment` must have
    'user' and 'sentiment' columns (real/non-system messages only).

    Columns: user, <label columns with counts>, total, dominant_sentiment.
    """
    if df_with_sentiment is None or df_with_sentiment.empty:
        return pd.DataFrame(columns=["user", "total", "dominant_sentiment"])

    label_order = _order_labels(df_with_sentiment["sentiment"].unique())
    rows = []
    for user, group in df_with_sentiment.groupby("user"):
        counts = group["sentiment"].value_counts()
        row = {"user": user}
        for label in label_order:
            row[label] = int(counts.get(label, 0))
        row["total"] = int(len(group))
        row["dominant_sentiment"] = counts.idxmax() if not counts.empty else None
        rows.append(row)

    cols = ["user"] + label_order + ["total", "dominant_sentiment"]
    return pd.DataFrame(rows, columns=cols).sort_values("total", ascending=False).reset_index(drop=True)


def fig_user_sentiment_comparison(df_with_sentiment: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing sentiment counts across users."""
    if df_with_sentiment is None or df_with_sentiment.empty:
        return _empty_figure("No messages to compare")

    label_order = _order_labels(df_with_sentiment["sentiment"].unique())
    counts = (
        df_with_sentiment.groupby(["user", "sentiment"]).size()
        .reset_index(name="count")
    )
    counts["sentiment"] = pd.Categorical(counts["sentiment"], categories=label_order, ordered=True)
    counts = counts.sort_values(["user", "sentiment"])

    fig = px.bar(
        counts, x="user", y="count", color="sentiment", barmode="group",
        category_orders={"sentiment": label_order},
        color_discrete_map={l: _color_for(l) for l in label_order},
        title="Sentiment Comparison Between Users",
    )
    fig.update_layout(xaxis_title="User", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    return fig


# ---------------------------------------------------------------------------
# Sentiment Timeline
# ---------------------------------------------------------------------------

def sentiment_timeline(df_with_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Daily sentiment counts over time. `df_with_sentiment` must have
    'only_date' and 'sentiment' columns.

    Returns a long-format DataFrame: only_date, sentiment, count.
    """
    cols = ["only_date", "sentiment", "count"]
    if df_with_sentiment is None or df_with_sentiment.empty:
        return pd.DataFrame(columns=cols)

    counts = (
        df_with_sentiment.groupby(["only_date", "sentiment"]).size()
        .reset_index(name="count")
        .sort_values("only_date")
    )
    return counts[cols]


def fig_sentiment_timeline(df_with_sentiment: pd.DataFrame) -> go.Figure:
    """Line chart of message sentiment counts over time, one line per label."""
    timeline = sentiment_timeline(df_with_sentiment)
    if timeline.empty:
        return _empty_figure("No messages to plot")

    label_order = _order_labels(timeline["sentiment"].unique())
    fig = px.line(
        timeline, x="only_date", y="count", color="sentiment", markers=True,
        category_orders={"sentiment": label_order},
        color_discrete_map={l: _color_for(l) for l in label_order},
        title="Sentiment Over Time",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    return fig
