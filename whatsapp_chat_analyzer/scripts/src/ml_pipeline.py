"""
ml_pipeline.py
---------------
Phase 5: Machine Learning Pipeline.

A reusable, leakage-safe supervised text-classification pipeline for the
sentiment/tone models this project trains. Everything here operates on
a genuinely LABELED dataset (text + label columns) -- never on a user's
uploaded WhatsApp chat, which has no ground-truth labels and must never
be treated as training data (see `load_labeled_dataset` docstring).

Pipeline stages implemented here:
    1. Dataset loading + provenance tracking (`load_labeled_dataset`)
    2. Train/test split with a stratification fallback for tiny classes
       (`split_dataset`)
    3. Feature extraction: TF-IDF (unigrams + bigrams by default,
       configurable `max_features` / `min_df`), built INSIDE an
       sklearn Pipeline so the vectorizer is only ever fit on the
       training fold -- this is what prevents train/test leakage.
       (`build_pipeline`)
    4. Model training for 3 classifiers -- Logistic Regression,
       Multinomial Naive Bayes, Random Forest (`train_and_evaluate`,
       `compare_models`)
    5. Evaluation: accuracy, macro precision/recall/F1, a full
       per-class classification report, and a confusion matrix -- all
       computed from `sklearn.metrics`, nothing hardcoded
       (`evaluate_pipeline`)
    6. Safe model persistence that validates a pipeline is actually
       fitted (and optionally that its label set matches what's
       expected) before writing to disk, using an atomic write so a
       crash mid-save can never corrupt an existing model file
       (`save_pipeline_safely`, `load_pipeline`)

Text preprocessing reuses the Phase 4 Hinglish-aware pipeline
(`nlp_preprocessing.clean_text`) as the TF-IDF vectorizer's
`preprocessor`, so slang expansion / stopword removal / repeated-
character squeezing / punctuation handling all happen before TF-IDF
ever sees the text.

On dataset size: the datasets available to this project (see
`data/sentiment_train.csv` and `data/hinglish_sentiment_demo.csv`) are
small, hand-curated sets meant for development/demonstration -- not a
scientifically representative corpus of real-world Hinglish sentiment.
Metrics computed against them describe fit-to-this-data only. See each
data file's header comment and README.md for the full provenance
disclaimer.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import preprocessing as nlp_preprocessing

__all__ = [
    "DEFAULT_MODEL_NAMES",
    "load_labeled_dataset",
    "split_dataset",
    "build_pipeline",
    "evaluate_pipeline",
    "train_and_evaluate",
    "compare_models",
    "refit_on_full_dataset",
    "save_pipeline_safely",
    "load_pipeline",
]

DEFAULT_MODEL_NAMES: Tuple[str, ...] = (
    "logistic_regression", "multinomial_nb", "random_forest",
)

# Factories rather than pre-built instances, so every call gets a fresh,
# unfitted estimator (important -- reusing a fitted estimator across
# multiple train_and_evaluate() calls would itself be a subtle leakage
# bug).
_MODEL_FACTORIES: Dict[str, Callable[..., object]] = {
    "logistic_regression": lambda **kw: LogisticRegression(
        max_iter=1000, class_weight="balanced", **kw
    ),
    "multinomial_nb": lambda **kw: MultinomialNB(**kw),
    "random_forest": lambda **kw: RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", **kw
    ),
}


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------

def load_labeled_dataset(
    paths: Sequence[str],
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Loads and combines one or more labeled CSV files into a single
    DataFrame with columns [text, label, source].

    IMPORTANT -- this function must only ever be pointed at genuinely
    labeled data (a CSV with real text+label columns produced by a
    human annotator or a documented labeling process). It must NEVER be
    used to synthesize labels for a user's uploaded, unlabeled WhatsApp
    chat -- there is no ground truth for personal chat sentiment, and
    treating unlabeled data as if it were labeled would make any
    resulting "accuracy" number meaningless/fabricated.

    Each source file's provenance is preserved in the `source` column
    so downstream code/reports can show exactly where every row of
    training data came from.

    Raises
    ------
    FileNotFoundError
        If any path doesn't exist.
    ValueError
        If a file is missing the expected columns, or if the combined
        dataset ends up with fewer than 2 distinct labels (which would
        make classification meaningless).
    """
    frames = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Labeled dataset not found: {path}")

        df = pd.read_csv(path, comment="#")
        missing = {text_col, label_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s) {missing}. "
                f"Expected a CSV with '{text_col}' and '{label_col}' columns."
            )

        df = df[[text_col, label_col]].copy()
        df["source"] = os.path.basename(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Clean up: drop empty/NaN text, strip whitespace, drop exact duplicates.
    combined[text_col] = combined[text_col].astype(str).str.strip()
    combined = combined[combined[text_col].str.len() > 0]
    combined[label_col] = combined[label_col].astype(str).str.strip().str.lower()
    combined = combined.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    if combined.empty:
        raise ValueError("Combined labeled dataset is empty after cleaning.")
    if combined[label_col].nunique() < 2:
        raise ValueError(
            "Combined labeled dataset has fewer than 2 distinct labels -- "
            "cannot train a classifier."
        )

    return combined


# ---------------------------------------------------------------------------
# 2. Train/test split (leakage-safe: caller must fit vectorizers only on
#    the returned training fold)
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Splits `df` into train/test folds. Stratifies by label when every
    class has enough members to support it; otherwise falls back to a
    plain random split with a warning (this matters for small
    demonstration datasets where a class might have very few examples).
    """
    X = df[text_col]
    y = df[label_col]

    label_counts = y.value_counts()
    min_class_count = int(label_counts.min())
    # Stratified splitting requires at least 2 members per class, and
    # enough of them that the requested test_size doesn't round a class
    # down to zero test examples.
    can_stratify = min_class_count >= 2 and (min_class_count * test_size) >= 1

    if can_stratify:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    warnings.warn(
        f"At least one class has too few examples (smallest class has "
        f"{min_class_count}) to stratify a {test_size:.0%} test split. "
        f"Falling back to a non-stratified random split.",
        stacklevel=2,
    )
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# 3. Feature extraction + model pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(
    model_name: str,
    max_features: Optional[int] = 5000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
    preprocessor: Optional[Callable[[str], str]] = None,
    model_kwargs: Optional[dict] = None,
) -> Pipeline:
    """Builds an unfitted sklearn Pipeline: TF-IDF vectorizer -> classifier.

    The TF-IDF vectorizer is configured with:
      - `ngram_range` defaulting to (1, 2) -- unigrams + bigrams.
      - configurable `max_features` (cap on vocabulary size).
      - configurable `min_df` (a term must appear in at least this many
        documents to be kept -- filters out one-off noise).
      - `preprocessor` defaulting to `nlp_preprocessing.clean_text`
        (the Phase 4 Hinglish-aware cleaning pipeline) so slang
        expansion / stopword removal / normalization happen before
        TF-IDF tokenizes.

    Because the vectorizer lives inside the Pipeline, calling
    `pipeline.fit(X_train, y_train)` fits the TF-IDF vocabulary/IDF
    weights using ONLY `X_train` -- this is what prevents test-set
    vocabulary from leaking into training, as required. Never call
    `.fit()` on the vectorizer directly with the full dataset.

    Parameters
    ----------
    model_name : one of DEFAULT_MODEL_NAMES
        ('logistic_regression', 'multinomial_nb', 'random_forest')
    """
    if model_name not in _MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Choose from {DEFAULT_MODEL_NAMES}."
        )

    if preprocessor is None:
        preprocessor = nlp_preprocessing.clean_text

    vectorizer = TfidfVectorizer(
        preprocessor=preprocessor,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )
    classifier = _MODEL_FACTORIES[model_name](**(model_kwargs or {}))

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# ---------------------------------------------------------------------------
# 4 & 5. Training + Evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: Iterable[str],
    y_test: Iterable[str],
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """Evaluates an already-fitted pipeline on held-out data. Every
    number here is computed from `sklearn.metrics` against the actual
    predictions -- nothing is hardcoded or assumed.

    Returns a dict with: accuracy, precision_macro, recall_macro,
    f1_macro, confusion_matrix (list of lists), labels (the row/column
    order of the confusion matrix), classification_report (per-class
    precision/recall/F1/support as a dict), n_test.
    """
    y_test = list(y_test)
    if labels is None:
        labels = sorted(set(y_test) | set(pipeline.classes_)) if hasattr(
            pipeline, "classes_"
        ) else sorted(set(y_test))

    preds = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(
        y_test, preds, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, preds, labels=labels)

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
        "classification_report": report,
        "n_test": len(y_test),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    model_name: str,
    text_col: str = "text",
    label_col: str = "label",
    max_features: Optional[int] = 5000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
    test_size: float = 0.2,
    random_state: int = 42,
    model_kwargs: Optional[dict] = None,
) -> dict:
    """Full train + evaluate cycle for one model on a held-out split.

    Data-leakage safety: the TF-IDF vectorizer is fit via
    `pipeline.fit(X_train, y_train)` -- it never sees `X_test` during
    fitting, and evaluation metrics come exclusively from the untouched
    test fold.

    Returns everything `evaluate_pipeline` returns, plus: model_name,
    the fitted `pipeline` itself, n_train, and the feature-extraction
    settings used (max_features, min_df, ngram_range) for reporting.
    """
    X_train, X_test, y_train, y_test = split_dataset(
        df, text_col=text_col, label_col=label_col,
        test_size=test_size, random_state=random_state,
    )

    pipeline = build_pipeline(
        model_name, max_features=max_features, min_df=min_df,
        ngram_range=ngram_range, model_kwargs=model_kwargs,
    )
    pipeline.fit(X_train, y_train)  # TF-IDF fit happens here, on X_train only

    labels = sorted(df[label_col].unique())
    metrics = evaluate_pipeline(pipeline, X_test, y_test, labels=labels)

    metrics.update({
        "model_name": model_name,
        "pipeline": pipeline,
        "n_train": len(X_train),
        "max_features": max_features,
        "min_df": min_df,
        "ngram_range": ngram_range,
    })
    return metrics


def compare_models(
    df: pd.DataFrame,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Trains and evaluates every model in `model_names` on the SAME
    train/test split (same `random_state`/`test_size` passed through
    `kwargs`), so the comparison is apples-to-apples.

    Returns
    -------
    (summary_df, results) where:
      - summary_df: one row per model with accuracy/precision/recall/F1,
        sorted by f1_macro descending (best model first).
      - results: dict[model_name] -> the full dict from
        `train_and_evaluate` (including the fitted pipeline and
        confusion matrix), for deeper inspection.
    """
    results = {name: train_and_evaluate(df, name, **kwargs) for name in model_names}

    summary_rows = [
        {
            "model": name,
            "accuracy": r["accuracy"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "f1_macro": r["f1_macro"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
        }
        for name, r in results.items()
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "f1_macro", ascending=False
    ).reset_index(drop=True)

    return summary_df, results


def refit_on_full_dataset(
    df: pd.DataFrame,
    model_name: str,
    text_col: str = "text",
    label_col: str = "label",
    max_features: Optional[int] = 5000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
    model_kwargs: Optional[dict] = None,
) -> Pipeline:
    """Refits a pipeline on the ENTIRE labeled dataset (train + test
    combined) -- the standard final step once a model has already been
    evaluated on a held-out split and its reported metrics are trusted.

    IMPORTANT: any accuracy/F1 you report for this final pipeline must
    come from the earlier `train_and_evaluate`/`compare_models` call on
    the held-out split, NOT from testing this refit pipeline on data it
    was just trained on (that would be leakage in the reporting, even
    though refitting on all data before deployment is itself standard
    and fine practice).
    """
    pipeline = build_pipeline(
        model_name, max_features=max_features, min_df=min_df,
        ngram_range=ngram_range, model_kwargs=model_kwargs,
    )
    pipeline.fit(df[text_col], df[label_col])
    return pipeline


# ---------------------------------------------------------------------------
# 6. Safe persistence
# ---------------------------------------------------------------------------

def save_pipeline_safely(
    pipeline: Pipeline,
    path: str,
    expected_labels: Optional[Sequence[str]] = None,
) -> bool:
    """Saves `pipeline` to `path` only if it passes basic safety checks,
    using an atomic write so a crash mid-save can never leave a
    corrupted or half-written file at `path`.

    Safety checks performed:
      1. The pipeline must actually be fitted (its classifier must have
         a `classes_` attribute) -- refuses to save an unfitted/broken
         pipeline.
      2. If `expected_labels` is given, the fitted classifier's label
         set must exactly match it -- refuses to silently save a model
         that would return unexpected labels to calling code (e.g. the
         app's inference layer expects exactly {'positive','negative',
         'neutral'}).

    Returns True if the pipeline was saved, False if a safety check
    failed (in which case nothing is written -- any existing file at
    `path` is left untouched). Never raises for a failed safety check;
    only raises for genuine I/O errors.
    """
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    if clf is None or not hasattr(clf, "classes_"):
        warnings.warn(
            "Refusing to save: pipeline's classifier is not fitted "
            "(no 'classes_' attribute found).", stacklevel=2,
        )
        return False

    if expected_labels is not None:
        actual = set(str(c) for c in clf.classes_)
        expected = set(str(c) for c in expected_labels)
        if actual != expected:
            warnings.warn(
                f"Refusing to save: fitted label set {sorted(actual)} does not "
                f"match expected_labels {sorted(expected)}.", stacklevel=2,
            )
            return False

    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)

    # Atomic write: dump to a temp file in the same directory, then
    # os.replace() onto the final path (a single filesystem operation),
    # so a crash mid-write never leaves a corrupted model file behind.
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".joblib.tmp")
    try:
        os.close(fd)
        joblib.dump(pipeline, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return True


def load_pipeline(path: str) -> Optional[Pipeline]:
    """Loads a pipeline saved by `save_pipeline_safely`. Returns None
    (rather than raising) if `path` doesn't exist, so callers can
    implement a graceful fallback."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)
