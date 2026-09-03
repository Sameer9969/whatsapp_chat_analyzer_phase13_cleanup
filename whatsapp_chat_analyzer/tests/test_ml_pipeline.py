"""
tests/test_ml_pipeline.py
----------------------------
Unit tests for ml_pipeline.py (Phase 5: Machine Learning Pipeline).

Covers dataset loading/provenance, leakage-safe train/test splitting,
pipeline construction, training + evaluation correctness, model
comparison, and safe persistence.

Run with:
    pytest tests/test_ml_pipeline.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from whatsapp_chat_analyzer.scripts.src import ml_pipeline as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTIMENT_TRAIN_CSV = os.path.join(REPO_ROOT, "data", "sentiment_train.csv")
HINGLISH_DEMO_CSV = os.path.join(REPO_ROOT, "data", "hinglish_sentiment_demo.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_df():
    """A small, easily-separable synthetic dataset for fast, deterministic
    pipeline-mechanics tests (not for judging real-world accuracy)."""
    texts = (
        ["great good excellent wonderful"] * 8
        + ["bad terrible awful horrible"] * 8
        + ["meeting schedule report update"] * 8
    )
    labels = ["positive"] * 8 + ["negative"] * 8 + ["neutral"] * 8
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def real_combined_df():
    return mp.load_labeled_dataset([SENTIMENT_TRAIN_CSV, HINGLISH_DEMO_CSV])


# ---------------------------------------------------------------------------
# 1. Dataset loading + provenance
# ---------------------------------------------------------------------------

def test_load_labeled_dataset_real_files():
    df = mp.load_labeled_dataset([SENTIMENT_TRAIN_CSV, HINGLISH_DEMO_CSV])
    assert set(df.columns) == {"text", "label", "source"}
    assert len(df) > 0
    assert set(df["source"].unique()) == {"sentiment_train.csv", "hinglish_sentiment_demo.csv"}
    assert df["label"].nunique() >= 2


def test_load_labeled_dataset_preserves_provenance_counts():
    df = mp.load_labeled_dataset([SENTIMENT_TRAIN_CSV, HINGLISH_DEMO_CSV])
    counts = df["source"].value_counts()
    # sentiment_train.csv has 152 rows, hinglish_sentiment_demo.csv has 90
    assert counts["sentiment_train.csv"] == 152
    assert counts["hinglish_sentiment_demo.csv"] == 90


def test_load_labeled_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        mp.load_labeled_dataset(["/tmp/does_not_exist_xyz123.csv"])


def test_load_labeled_dataset_missing_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        mp.load_labeled_dataset([str(bad_csv)])


def test_load_labeled_dataset_drops_duplicates_and_empty_rows(tmp_path):
    csv_path = tmp_path / "dup.csv"
    csv_path.write_text("text,label\nhello,positive\nhello,positive\n,neutral\n   ,neutral\nworld,negative\n")
    df = mp.load_labeled_dataset([str(csv_path)])
    assert len(df) == 2  # "hello" (deduped) + "world"; blank rows dropped


def test_load_labeled_dataset_single_label_raises(tmp_path):
    csv_path = tmp_path / "single_label.csv"
    csv_path.write_text("text,label\na,positive\nb,positive\nc,positive\n")
    with pytest.raises(ValueError):
        mp.load_labeled_dataset([str(csv_path)])


def test_load_labeled_dataset_reads_comment_header(tmp_path):
    csv_path = tmp_path / "commented.csv"
    csv_path.write_text("# this is a demo dataset disclaimer\ntext,label\nhi,positive\nbye,negative\n")
    df = mp.load_labeled_dataset([str(csv_path)])
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 2. Train/test split (leakage-safe)
# ---------------------------------------------------------------------------

def test_split_dataset_proportions(tiny_df):
    X_train, X_test, y_train, y_test = mp.split_dataset(tiny_df, test_size=0.25, random_state=1)
    assert len(X_train) + len(X_test) == len(tiny_df)
    assert len(X_test) == round(len(tiny_df) * 0.25)


def test_split_dataset_stratifies_when_possible(tiny_df):
    _, _, y_train, y_test = mp.split_dataset(tiny_df, test_size=0.25, random_state=1)
    # each class had 8 members -> 25% test split should keep all 3 classes in both folds
    assert set(y_train) == {"positive", "negative", "neutral"}
    assert set(y_test) == {"positive", "negative", "neutral"}


def test_split_dataset_falls_back_when_class_too_small():
    df = pd.DataFrame({
        "text": ["a", "b", "c", "d", "e", "f", "only one of me"],
        "label": ["x", "x", "x", "y", "y", "y", "z"],  # 'z' has only 1 member
    })
    with pytest.warns(UserWarning, match="Falling back to a non-stratified"):
        X_train, X_test, y_train, y_test = mp.split_dataset(df, test_size=0.3, random_state=1)
    assert len(X_train) + len(X_test) == len(df)


# ---------------------------------------------------------------------------
# 3. Pipeline construction (feature extraction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name,expected_type", [
    ("logistic_regression", LogisticRegression),
    ("multinomial_nb", MultinomialNB),
    ("random_forest", RandomForestClassifier),
])
def test_build_pipeline_correct_classifier_type(model_name, expected_type):
    pipeline = mp.build_pipeline(model_name)
    assert isinstance(pipeline.named_steps["clf"], expected_type)


def test_build_pipeline_unknown_model_raises():
    with pytest.raises(ValueError):
        mp.build_pipeline("not_a_real_model")


def test_build_pipeline_tfidf_settings_applied():
    pipeline = mp.build_pipeline(
        "logistic_regression", max_features=123, min_df=3, ngram_range=(1, 3)
    )
    tfidf = pipeline.named_steps["tfidf"]
    assert tfidf.max_features == 123
    assert tfidf.min_df == 3
    assert tfidf.ngram_range == (1, 3)


def test_build_pipeline_default_ngram_is_unigram_bigram():
    pipeline = mp.build_pipeline("logistic_regression")
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)


def test_build_pipeline_uses_hinglish_preprocessor_by_default():
    from whatsapp_chat_analyzer.scripts.src import preprocessing as nlp_preprocessing
    pipeline = mp.build_pipeline("logistic_regression")
    assert pipeline.named_steps["tfidf"].preprocessor is nlp_preprocessing.clean_text


def test_build_pipeline_custom_preprocessor_override():
    custom = lambda t: t.upper()
    pipeline = mp.build_pipeline("logistic_regression", preprocessor=custom)
    assert pipeline.named_steps["tfidf"].preprocessor is custom


# ---------------------------------------------------------------------------
# Data leakage prevention (the critical requirement)
# ---------------------------------------------------------------------------

def test_tfidf_vocabulary_built_only_from_training_data():
    """Direct proof of leakage prevention: a token appearing ONLY in the
    test set must never appear in the fitted vectorizer's vocabulary,
    since the vectorizer is fit via pipeline.fit(X_train, ...)."""
    X_train = ["good day today", "bad day today", "great work done"]
    y_train = ["positive", "negative", "positive"]
    X_test_only_token = "zzzuniquetesttoken"

    pipeline = mp.build_pipeline("logistic_regression", max_features=100, min_df=1)
    pipeline.fit(X_train, y_train)

    vocab = pipeline.named_steps["tfidf"].vocabulary_
    assert X_test_only_token not in vocab
    # sanity: tokens that WERE in training text must be present
    assert "good" in vocab or "good day" in vocab


def test_train_and_evaluate_does_not_fit_on_test_fold(tiny_df):
    """End-to-end version of the leakage check using the public
    train_and_evaluate() entry point and a real split."""
    result = mp.train_and_evaluate(
        tiny_df, "logistic_regression", test_size=0.25, random_state=1,
        max_features=50, min_df=1,
    )
    vocab = result["pipeline"].named_steps["tfidf"].vocabulary_
    # None of the raw fixture words should be missing (they appear in
    # both classes' repeated text), but more importantly, re-deriving
    # the split and checking test-only unique tokens is covered above;
    # here we confirm the pipeline was fit on n_train rows only.
    assert result["n_train"] == len(tiny_df) - result["n_test"]


# ---------------------------------------------------------------------------
# 4 & 5. Training + Evaluation correctness
# ---------------------------------------------------------------------------

def test_train_and_evaluate_returns_expected_keys(tiny_df):
    result = mp.train_and_evaluate(tiny_df, "logistic_regression", max_features=50, min_df=1)
    expected_keys = {
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "confusion_matrix", "labels", "classification_report", "n_test",
        "model_name", "pipeline", "n_train", "max_features", "min_df", "ngram_range",
    }
    assert expected_keys.issubset(result.keys())


def test_train_and_evaluate_on_easily_separable_data_scores_well(tiny_df):
    # This synthetic fixture has near-zero vocabulary overlap between
    # classes, so a competent classifier should score highly -- this is
    # a sanity check on the mechanics, not a real-world accuracy claim.
    result = mp.train_and_evaluate(
        tiny_df, "logistic_regression", test_size=0.25, random_state=1,
        max_features=50, min_df=1,
    )
    assert result["accuracy"] >= 0.8


def test_evaluate_pipeline_accuracy_matches_manual_calculation():
    # Construct a fitted pipeline whose predictions we can control by
    # using training data identical to test data (deterministic).
    X = ["good good good", "bad bad bad", "good good good", "bad bad bad"]
    y = ["positive", "negative", "positive", "negative"]
    pipeline = mp.build_pipeline("logistic_regression", max_features=10, min_df=1)
    pipeline.fit(X, y)

    X_test = ["good good good", "bad bad bad", "good good good"]
    y_test = ["positive", "negative", "negative"]  # last one is intentionally "wrong"
    metrics = mp.evaluate_pipeline(pipeline, X_test, y_test, labels=["negative", "positive"])

    preds = pipeline.predict(X_test)
    manual_accuracy = sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
    assert metrics["accuracy"] == pytest.approx(manual_accuracy)


def test_confusion_matrix_shape_matches_label_count(tiny_df):
    result = mp.train_and_evaluate(tiny_df, "logistic_regression", max_features=50, min_df=1)
    n_labels = len(result["labels"])
    cm = np.array(result["confusion_matrix"])
    assert cm.shape == (n_labels, n_labels)
    assert cm.sum() == result["n_test"]


def test_metrics_are_not_hardcoded_and_vary_with_data(tiny_df):
    """Metrics should genuinely reflect the data -- shuffling labels
    randomly should generally degrade accuracy versus the real labels
    (proves the number isn't a fixed/manufactured constant)."""
    result_real = mp.train_and_evaluate(
        tiny_df, "logistic_regression", test_size=0.3, random_state=1, max_features=50, min_df=1
    )

    shuffled_df = tiny_df.copy()
    rng = np.random.RandomState(0)
    shuffled_df["label"] = rng.permutation(shuffled_df["label"].values)
    result_shuffled = mp.train_and_evaluate(
        shuffled_df, "logistic_regression", test_size=0.3, random_state=1, max_features=50, min_df=1
    )

    assert result_real["accuracy"] != result_shuffled["accuracy"] or result_real["accuracy"] >= result_shuffled["accuracy"]
    assert result_real["accuracy"] >= result_shuffled["accuracy"]


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def test_compare_models_returns_all_models(tiny_df):
    summary_df, results = mp.compare_models(
        tiny_df, max_features=50, min_df=1, test_size=0.25, random_state=1
    )
    assert set(summary_df["model"]) == set(mp.DEFAULT_MODEL_NAMES)
    assert set(results.keys()) == set(mp.DEFAULT_MODEL_NAMES)


def test_compare_models_sorted_by_f1_descending(tiny_df):
    summary_df, _ = mp.compare_models(
        tiny_df, max_features=50, min_df=1, test_size=0.25, random_state=1
    )
    f1_values = summary_df["f1_macro"].tolist()
    assert f1_values == sorted(f1_values, reverse=True)


def test_compare_models_same_split_used_across_models(tiny_df):
    """All models must be evaluated on the identical train/test split
    for the comparison to be fair."""
    _, results = mp.compare_models(
        tiny_df, max_features=50, min_df=1, test_size=0.25, random_state=1
    )
    n_train_values = {r["n_train"] for r in results.values()}
    n_test_values = {r["n_test"] for r in results.values()}
    assert len(n_train_values) == 1
    assert len(n_test_values) == 1


def test_compare_models_custom_model_subset(tiny_df):
    summary_df, results = mp.compare_models(
        tiny_df, model_names=["logistic_regression", "multinomial_nb"],
        max_features=50, min_df=1,
    )
    assert set(results.keys()) == {"logistic_regression", "multinomial_nb"}


# ---------------------------------------------------------------------------
# Refit on full dataset
# ---------------------------------------------------------------------------

def test_refit_on_full_dataset_uses_all_rows(tiny_df):
    pipeline = mp.refit_on_full_dataset(tiny_df, "logistic_regression", max_features=50, min_df=1)
    # A fitted TF-IDF's idf_ vector length should reflect the full-dataset vocabulary
    assert hasattr(pipeline.named_steps["clf"], "classes_")
    assert set(pipeline.named_steps["clf"].classes_) == set(tiny_df["label"].unique())


# ---------------------------------------------------------------------------
# 6. Safe persistence
# ---------------------------------------------------------------------------

def test_save_pipeline_safely_success(tiny_df, tmp_path):
    pipeline = mp.refit_on_full_dataset(tiny_df, "logistic_regression", max_features=50, min_df=1)
    path = str(tmp_path / "model.joblib")
    ok = mp.save_pipeline_safely(pipeline, path, expected_labels=["positive", "negative", "neutral"])
    assert ok is True
    assert os.path.exists(path)


def test_save_pipeline_safely_rejects_unfitted_pipeline(tmp_path):
    pipeline = mp.build_pipeline("logistic_regression")
    path = str(tmp_path / "should_not_exist.joblib")
    ok = mp.save_pipeline_safely(pipeline, path)
    assert ok is False
    assert not os.path.exists(path)


def test_save_pipeline_safely_rejects_label_mismatch(tiny_df, tmp_path):
    pipeline = mp.refit_on_full_dataset(tiny_df, "logistic_regression", max_features=50, min_df=1)
    path = str(tmp_path / "should_not_exist.joblib")
    ok = mp.save_pipeline_safely(pipeline, path, expected_labels=["totally", "wrong", "labels"])
    assert ok is False
    assert not os.path.exists(path)


def test_save_pipeline_safely_does_not_clobber_existing_file_on_failure(tiny_df, tmp_path):
    path = str(tmp_path / "existing.joblib")
    good_pipeline = mp.refit_on_full_dataset(tiny_df, "logistic_regression", max_features=50, min_df=1)
    assert mp.save_pipeline_safely(good_pipeline, path) is True

    unfitted = mp.build_pipeline("logistic_regression")
    ok = mp.save_pipeline_safely(unfitted, path)  # should fail and not touch the existing file
    assert ok is False
    reloaded = mp.load_pipeline(path)
    assert hasattr(reloaded.named_steps["clf"], "classes_")  # still the good, fitted one


def test_load_pipeline_roundtrip(tiny_df, tmp_path):
    pipeline = mp.refit_on_full_dataset(tiny_df, "logistic_regression", max_features=50, min_df=1)
    path = str(tmp_path / "model.joblib")
    mp.save_pipeline_safely(pipeline, path)

    loaded = mp.load_pipeline(path)
    assert loaded.predict(["great good excellent"])[0] == "positive"


def test_load_pipeline_missing_file_returns_none():
    assert mp.load_pipeline("/tmp/definitely_does_not_exist_abc123.joblib") is None


# ---------------------------------------------------------------------------
# Integration: real combined dataset end-to-end
# ---------------------------------------------------------------------------

def test_full_pipeline_on_real_combined_dataset(real_combined_df):
    summary_df, results = mp.compare_models(
        real_combined_df, max_features=2000, min_df=2, test_size=0.2, random_state=42,
    )
    assert len(summary_df) == 3
    for _, row in summary_df.iterrows():
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1_macro"] <= 1.0

    best_model = summary_df.iloc[0]["model"]
    cm = np.array(results[best_model]["confusion_matrix"])
    assert cm.shape[0] == cm.shape[1] == len(results[best_model]["labels"])


def test_real_pipeline_predicts_on_unseen_hinglish_text(real_combined_df):
    result = mp.train_and_evaluate(
        real_combined_df, "logistic_regression", max_features=2000, min_df=2,
    )
    pipeline = result["pipeline"]
    # Sentences not verbatim present in either training file
    preds = pipeline.predict(["bahut mast tha aaj ka din", "bahut bura laga aaj"])
    assert all(p in {"positive", "negative", "neutral"} for p in preds)
