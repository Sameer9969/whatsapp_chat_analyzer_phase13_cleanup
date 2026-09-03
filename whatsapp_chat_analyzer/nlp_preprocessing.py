"""
nlp_preprocessing.py (compatibility shim)
------------------------------------------
Do not import from this file in new code -- use `from src import
preprocessing` instead.

Why this file still exists: the trained models under `models/` (e.g.
hinglish_random_forest.joblib) were saved with scikit-learn's
`TfidfVectorizer(preprocessor=nlp_preprocessing.clean_text)`. joblib/pickle
stores that as a reference to the exact module path the function lived in
at save time -- top-level `nlp_preprocessing`, before this Phase 13
cleanup moved the real implementation to `src/preprocessing.py`.

Without this shim, loading any existing model would raise
`ModuleNotFoundError: No module named 'nlp_preprocessing'` and every
already-trained model would have to be retrained from scratch. This file
just re-exports the same functions under the old module path so those
artifacts keep loading. If the models are ever retrained (see
scripts/train_hinglish_pipeline.py, which imports `src.preprocessing`
directly), this shim can be deleted.
"""

from whatsapp_chat_analyzer.scripts.src.preprocessing import *  # noqa: F401,F403
from whatsapp_chat_analyzer.scripts.src.preprocessing import clean_text, HinglishTextPreprocessor  # noqa: F401
