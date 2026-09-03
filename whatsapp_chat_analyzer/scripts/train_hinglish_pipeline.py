"""
train_hinglish_pipeline.py
----------------------------
Phase 5 entry-point script: trains and compares Logistic Regression,
Multinomial Naive Bayes, and Random Forest sentiment classifiers using
the reusable functions in ml_pipeline.py, on the combined labeled
dataset described below.

Run:
    python train_hinglish_pipeline.py

WHAT THIS DOES NOT DO:
    - It does NOT touch models/sentiment_model.joblib, the model the
      live Streamlit app currently loads (see legacy_sentiment_model.py). This
      script's output is saved to a separate path
      (models/hinglish_sentiment_pipeline.joblib) so the running app's
      behavior is completely unaffected until a future phase explicitly
      decides to switch the app over.
    - It does NOT train on the user's uploaded WhatsApp chat. Only the
      labeled CSV files below are used as training data.

DATASET PROVENANCE (documented per project requirement):
    1. data/sentiment_train.csv (152 rows)
       - Hand-curated, English-only, generic chat-style sentences.
       - Balanced across positive/negative/neutral (~50 each).
       - Synthetic (written by the developer), not collected from real
         chats. Does not exercise Hinglish/code-mixed text at all.
    2. data/hinglish_sentiment_demo.csv (90 rows)
       - NEW in this phase. Hand-written, romanized-Hindi/Hinglish
         sentences (e.g. "mast hai bro", "mood off hai"), balanced
         30/30/30 across positive/negative/neutral.
       - Explicitly marked as DEMONSTRATION DATA in its own header
         comment and in README.md -- written by the developer for
         pipeline development/testing, not collected from real users.

    Combined: 242 rows across 2 sources. This is a demonstration-scale
    dataset suitable for exercising a genuine train/test/evaluate ML
    pipeline end-to-end. The accuracy/F1 numbers below describe how
    well each model fits THIS combined dataset -- they are NOT a
    scientifically valid estimate of real-world Hinglish sentiment
    classification accuracy, and should not be reported as such. A
    larger, properly collected and independently annotated corpus
    would be needed for that.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whatsapp_chat_analyzer.scripts.src import ml_pipeline as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATHS = [
    os.path.join(BASE_DIR, "data", "sentiment_train.csv"),
    os.path.join(BASE_DIR, "data", "hinglish_sentiment_demo.csv"),
]
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "hinglish_sentiment_pipeline.joblib")
OUTPUT_REPORT_PATH = os.path.join(MODEL_DIR, "model_comparison_report.json")

# Every compared model is saved individually (not just the best one) so
# the app's "Model Selection" feature (Phase 6) can let the user choose
# among all three trained classifiers, not only the winner.
INDIVIDUAL_MODEL_PATHS = {
    "logistic_regression": os.path.join(MODEL_DIR, "hinglish_logistic_regression.joblib"),
    "multinomial_nb": os.path.join(MODEL_DIR, "hinglish_multinomial_nb.joblib"),
    "random_forest": os.path.join(MODEL_DIR, "hinglish_random_forest.joblib"),
}

# Feature-extraction settings (all configurable -- see ml_pipeline.build_pipeline)
MAX_FEATURES = 3000
MIN_DF = 2
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _print_dataset_provenance(df):
    print("=" * 70)
    print("DATASET PROVENANCE")
    print("=" * 70)
    for source, count in df["source"].value_counts().items():
        print(f"  {source:32s} {count:4d} rows")
    print(f"  {'TOTAL':32s} {len(df):4d} rows")
    print(f"  Label balance: {dict(df['label'].value_counts())}")
    print(
        "\n  NOTE: this is a demonstration-scale combined dataset "
        "(see this script's module docstring and each CSV's header "
        "comment). Metrics below describe fit to THIS data, not a "
        "general-purpose accuracy claim.\n"
    )


def _print_comparison(summary_df, results):
    print("=" * 70)
    print("MODEL COMPARISON (held-out test split, never seen during TF-IDF fit)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()

    for name, r in results.items():
        print("-" * 70)
        print(f"Model: {name}")
        print(f"  n_train={r['n_train']}  n_test={r['n_test']}  "
              f"max_features={r['max_features']}  min_df={r['min_df']}  "
              f"ngram_range={r['ngram_range']}")
        print(f"  Accuracy:         {r['accuracy']:.4f}")
        print(f"  Precision (macro):{r['precision_macro']:.4f}")
        print(f"  Recall (macro):   {r['recall_macro']:.4f}")
        print(f"  F1 (macro):       {r['f1_macro']:.4f}")
        print(f"  Confusion matrix (rows=actual, cols=predicted, labels={r['labels']}):")
        for row in r["confusion_matrix"]:
            print(f"    {row}")
    print("-" * 70)


def main():
    df = mp.load_labeled_dataset(DATA_PATHS)
    _print_dataset_provenance(df)

    summary_df, results = mp.compare_models(
        df,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        ngram_range=NGRAM_RANGE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    _print_comparison(summary_df, results)

    best_model_name = summary_df.iloc[0]["model"]
    print(f"\nBest model by macro-F1 on the held-out split: {best_model_name}")

    # Refit EVERY compared model on the FULL combined dataset and save
    # each individually, so the app's Model Selection dropdown can offer
    # all three -- not just the winner. Their reported accuracy/F1 above
    # came from the held-out split computed before this refit -- not
    # from testing on data any of them was just trained on.
    expected_labels = sorted(df["label"].unique())
    saved_paths = {}
    for name in results:
        pipeline_for_name = mp.refit_on_full_dataset(
            df, name, max_features=MAX_FEATURES, min_df=MIN_DF, ngram_range=NGRAM_RANGE,
        )
        path = INDIVIDUAL_MODEL_PATHS[name]
        ok = mp.save_pipeline_safely(pipeline_for_name, path, expected_labels=expected_labels)
        if ok:
            saved_paths[name] = path
            print(f"Saved {name} -> {path}")
        else:
            print(f"WARNING: {name} failed a safety check and was NOT saved.")

    # Also keep saving the single best model under its Phase 5 filename,
    # for backward compatibility with anything already pointing at it.
    final_pipeline = saved_paths.get(best_model_name) and mp.load_pipeline(saved_paths[best_model_name])
    if final_pipeline is None:
        final_pipeline = mp.refit_on_full_dataset(
            df, best_model_name, max_features=MAX_FEATURES, min_df=MIN_DF, ngram_range=NGRAM_RANGE,
        )
    saved = mp.save_pipeline_safely(
        final_pipeline, OUTPUT_MODEL_PATH, expected_labels=expected_labels
    )
    if saved:
        print(f"Saved final (best) pipeline ({best_model_name}) -> {OUTPUT_MODEL_PATH}")
    else:
        print("WARNING: best pipeline failed a safety check and was NOT saved.")

    # Save a JSON-serializable comparison report (metrics only -- not
    # the fitted pipeline objects, which aren't JSON-serializable).
    os.makedirs(MODEL_DIR, exist_ok=True)
    report = {
        "dataset": {
            "sources": df["source"].value_counts().to_dict(),
            "total_rows": len(df),
            "label_counts": df["label"].value_counts().to_dict(),
            "disclaimer": (
                "Demonstration-scale combined dataset (see "
                "train_hinglish_pipeline.py docstring). Not a "
                "scientifically representative corpus."
            ),
        },
        "settings": {
            "max_features": MAX_FEATURES, "min_df": MIN_DF,
            "ngram_range": list(NGRAM_RANGE), "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "models": {
            name: {
                "accuracy": r["accuracy"],
                "precision_macro": r["precision_macro"],
                "recall_macro": r["recall_macro"],
                "f1_macro": r["f1_macro"],
                "confusion_matrix": r["confusion_matrix"],
                "labels": r["labels"],
                "classification_report": r["classification_report"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            }
            for name, r in results.items()
        },
        "best_model": best_model_name,
        "saved_model_path": OUTPUT_MODEL_PATH if saved else None,
        "saved_individual_model_paths": saved_paths,
    }
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved comparison report -> {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
