"""
tests/test_sentiment_analysis.py
-----------------------------------
Unit tests for sentiment_analysis.py (Phase 6: Sentiment Analysis
Integration).

Run with:
    pytest tests/test_sentiment_analysis.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import pytest

from whatsapp_chat_analyzer.scripts.src import sentiment as sa

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_MODEL_PATH = os.path.join(REPO_ROOT, "models", "hinglish_random_forest.joblib")
LR_MODEL_PATH = os.path.join(REPO_ROOT, "models", "hinglish_logistic_regression.joblib")


def _skip_if_missing(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present -- run train_hinglish_pipeline.py first")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def test_available_models_always_includes_lexicon_fallback():
    available = sa.available_models()
    assert "Keyword Lexicon (fallback, no ML model)" in available
    assert available["Keyword Lexicon (fallback, no ML model)"] == sa.LEXICON_FALLBACK_PATH


def test_available_models_only_lists_existing_files():
    available = sa.available_models()
    for name, path in available.items():
        if path == sa.LEXICON_FALLBACK_PATH:
            continue
        assert os.path.exists(path), f"{name} points to a non-existent file: {path}"


def test_available_models_includes_trained_hinglish_models_if_present():
    _skip_if_missing(RF_MODEL_PATH)
    available = sa.available_models()
    assert "Random Forest (Hinglish-aware)" in available


# ---------------------------------------------------------------------------
# Predictions -- only positive/neutral/negative (no invented classes)
# ---------------------------------------------------------------------------

def test_predict_messages_only_uses_trained_classes():
    _skip_if_missing(RF_MODEL_PATH)
    labels = sa.predict_messages(
        ["mast hai bro", "mood off hai", "kal milte hai", "amazing news!", "this is terrible"],
        RF_MODEL_PATH,
    )
    assert len(labels) == 5
    assert set(labels).issubset({"positive", "negative", "neutral"})


def test_predict_messages_empty_list():
    assert sa.predict_messages([], RF_MODEL_PATH if os.path.exists(RF_MODEL_PATH) else sa.LEXICON_FALLBACK_PATH) == []


def test_predict_messages_returns_plain_list_with_count_method():
    _skip_if_missing(RF_MODEL_PATH)
    labels = sa.predict_messages(["good", "bad"], RF_MODEL_PATH)
    assert isinstance(labels, list)
    assert hasattr(labels, "count")  # db.py relies on list.count()


def test_predict_messages_lexicon_fallback_no_ml_model_needed():
    labels = sa.predict_messages(["good great awesome", "bad terrible awful", "kal milte hai"], sa.LEXICON_FALLBACK_PATH)
    assert labels == ["positive", "negative", "neutral"]


def test_predict_single_with_confidence_returns_valid_label():
    _skip_if_missing(RF_MODEL_PATH)
    label, confidence = sa.predict_single_with_confidence("mast hai bro", RF_MODEL_PATH)
    assert label in {"positive", "negative", "neutral"}
    assert confidence is not None
    assert set(confidence.keys()) == {"positive", "negative", "neutral"}
    assert abs(sum(confidence.values()) - 1.0) < 0.01  # probabilities sum to ~1


def test_predict_single_with_confidence_empty_text():
    label, confidence = sa.predict_single_with_confidence("", RF_MODEL_PATH if os.path.exists(RF_MODEL_PATH) else sa.LEXICON_FALLBACK_PATH)
    assert label == "neutral"
    assert confidence is None


def test_predict_single_lexicon_fallback_has_no_confidence():
    label, confidence = sa.predict_single_with_confidence("mast hai bro", sa.LEXICON_FALLBACK_PATH)
    assert label in {"positive", "negative", "neutral"}
    assert confidence is None


def test_predict_single_multinomial_nb_supports_confidence():
    path = os.path.join(REPO_ROOT, "models", "hinglish_multinomial_nb.joblib")
    _skip_if_missing(path)
    label, confidence = sa.predict_single_with_confidence("bahut accha laga", path)
    assert confidence is not None  # MultinomialNB supports predict_proba


# ---------------------------------------------------------------------------
# Overall Sentiment
# ---------------------------------------------------------------------------

def test_overall_sentiment_stats_counts_and_percentages():
    labels = ["positive", "positive", "negative", "neutral"]
    stats = sa.overall_sentiment_stats(labels)
    assert stats["total"] == 4
    assert stats["counts"] == {"positive": 2, "neutral": 1, "negative": 1}
    assert stats["percentages"]["positive"] == 50.0
    assert stats["percentages"]["neutral"] == 25.0
    assert stats["percentages"]["negative"] == 25.0


def test_overall_sentiment_stats_preferred_label_order():
    labels = ["negative", "positive", "neutral"]
    stats = sa.overall_sentiment_stats(labels)
    assert stats["label_order"] == ["positive", "neutral", "negative"]


def test_overall_sentiment_stats_empty():
    stats = sa.overall_sentiment_stats([])
    assert stats == {"total": 0, "label_order": [], "counts": {}, "percentages": {}}


def test_overall_sentiment_stats_percentages_sum_to_100():
    labels = ["positive"] * 7 + ["negative"] * 3 + ["neutral"] * 5
    stats = sa.overall_sentiment_stats(labels)
    assert abs(sum(stats["percentages"].values()) - 100.0) < 0.01


def test_fig_sentiment_distribution_returns_figure():
    labels = ["positive", "negative", "neutral", "positive"]
    fig = sa.fig_sentiment_distribution(labels)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_fig_sentiment_distribution_empty_placeholder():
    fig = sa.fig_sentiment_distribution([])
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 1


# ---------------------------------------------------------------------------
# User Sentiment
# ---------------------------------------------------------------------------

@pytest.fixture
def user_sentiment_df():
    return pd.DataFrame({
        "user": ["Aditi", "Aditi", "Rohan", "Rohan", "Rohan"],
        "sentiment": ["positive", "negative", "positive", "positive", "neutral"],
        "only_date": pd.to_datetime([
            "2023-05-12", "2023-05-12", "2023-05-13", "2023-05-13", "2023-05-14"
        ]).date,
    })


def test_user_sentiment_stats_counts(user_sentiment_df):
    stats = sa.user_sentiment_stats(user_sentiment_df)
    aditi = stats[stats["user"] == "Aditi"].iloc[0]
    rohan = stats[stats["user"] == "Rohan"].iloc[0]
    assert aditi["positive"] == 1
    assert aditi["negative"] == 1
    assert aditi["total"] == 2
    assert rohan["positive"] == 2
    assert rohan["neutral"] == 1
    assert rohan["total"] == 3
    assert rohan["dominant_sentiment"] == "positive"


def test_user_sentiment_stats_empty():
    empty = pd.DataFrame(columns=["user", "sentiment", "only_date"])
    stats = sa.user_sentiment_stats(empty)
    assert stats.empty


def test_fig_user_sentiment_comparison_returns_figure(user_sentiment_df):
    fig = sa.fig_user_sentiment_comparison(user_sentiment_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    users_in_chart = set()
    for trace in fig.data:
        users_in_chart.update(trace.x)
    assert users_in_chart == {"Aditi", "Rohan"}


def test_fig_user_sentiment_comparison_empty_placeholder():
    empty = pd.DataFrame(columns=["user", "sentiment", "only_date"])
    fig = sa.fig_user_sentiment_comparison(empty)
    assert len(fig.layout.annotations) == 1


# ---------------------------------------------------------------------------
# Sentiment Timeline
# ---------------------------------------------------------------------------

def test_sentiment_timeline_shape(user_sentiment_df):
    timeline = sa.sentiment_timeline(user_sentiment_df)
    assert list(timeline.columns) == ["only_date", "sentiment", "count"]
    assert timeline["count"].sum() == len(user_sentiment_df)


def test_sentiment_timeline_empty():
    empty = pd.DataFrame(columns=["user", "sentiment", "only_date"])
    timeline = sa.sentiment_timeline(empty)
    assert timeline.empty


def test_fig_sentiment_timeline_returns_figure(user_sentiment_df):
    fig = sa.fig_sentiment_timeline(user_sentiment_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_fig_sentiment_timeline_empty_placeholder():
    empty = pd.DataFrame(columns=["user", "sentiment", "only_date"])
    fig = sa.fig_sentiment_timeline(empty)
    assert len(fig.layout.annotations) == 1


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

def test_disclaimer_text_present_and_matches_requirement():
    expected = (
        "Sentiment classification is an ML-based estimate and may be inaccurate "
        "for sarcasm, context-dependent Hinglish, slang, and multi-turn conversations."
    )
    assert sa.DISCLAIMER == expected


# ---------------------------------------------------------------------------
# No external calls (static/behavioral check)
# ---------------------------------------------------------------------------

def test_module_does_not_import_network_libraries():
    import inspect
    source = inspect.getsource(sa)
    for forbidden in ["requests", "urllib", "http.client", "socket", "openai", "anthropic"]:
        assert forbidden not in source, f"sentiment_analysis.py must not use {forbidden}"


# ---------------------------------------------------------------------------
# Integration: real sample_chat.txt end-to-end
# ---------------------------------------------------------------------------

def test_full_sentiment_pipeline_on_sample_chat():
    from whatsapp_chat_analyzer.scripts.src import parser as preprocessor
    sample_path = os.path.join(REPO_ROOT, "sample_chat.txt")
    with open(sample_path, encoding="utf-8") as f:
        data = f.read()
    df = preprocessor.preprocess(data)
    real = df[df["user"] != "group_notification"].copy()

    available = sa.available_models()
    # Lexicon fallback is always available, so this never skips entirely.
    model_path = list(available.values())[0]

    labels = sa.predict_messages(real["message"].tolist(), model_path)
    assert len(labels) == len(real)
    real["sentiment"] = labels

    stats = sa.overall_sentiment_stats(labels)
    assert stats["total"] == len(real)

    user_stats = sa.user_sentiment_stats(real)
    assert not user_stats.empty

    timeline = sa.sentiment_timeline(real)
    assert not timeline.empty

    for fn in [sa.fig_sentiment_distribution]:
        fig = fn(labels)
        assert isinstance(fig, go.Figure)
    for fn in [sa.fig_user_sentiment_comparison, sa.fig_sentiment_timeline]:
        fig = fn(real)
        assert isinstance(fig, go.Figure)
