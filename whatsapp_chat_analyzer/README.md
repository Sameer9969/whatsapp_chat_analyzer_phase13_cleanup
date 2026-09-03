# WhatsApp Chat Analysis (Mini Project)

A Streamlit web app that turns a raw, exported WhatsApp chat (`.txt`) into
statistics, timelines, activity maps, word clouds, emoji analysis, and
**ML-powered sentiment analysis** — with every run persisted to a database.

This matches the accompanying project presentation (`WhatsApp_Chat_Analysis_Mini_Project.pptx`).

## Features

- Parses raw WhatsApp `.txt` exports (handles both 12-hour and 24-hour,
  2-digit and 4-digit year formats, iOS bracket-style exports, multi-line
  messages, and media/deleted/system messages)
- Top statistics: messages, words, media, links
- **Chat & user analytics**: per-user message/word counts, average
  message length, most active hours/days, interactive Plotly charts
  (by date/month/day-of-week/hour/user), and user rankings
- **Advanced behavioral analytics**: response latency (avg/median/
  fastest/slowest, per-user and per-user-pair), night-owl activity
  analysis, conversation-starter estimation, activity streaks, and a
  user-interaction heatmap — all with a documented methodology and none
  of it claiming to measure interest/personality/relationship strength
- **Hinglish-aware NLP preprocessing** — a reusable, rule-based text
  pipeline for code-mixed English/Hinglish chat text (see below)
- **Machine learning pipeline** (Phase 5) — a leakage-safe, reusable
  train/evaluate/compare pipeline (Logistic Regression vs. Naive Bayes
  vs. Random Forest) with real accuracy/precision/recall/F1/confusion-
  matrix metrics on a documented, honestly-labeled demonstration dataset
  (see below)
- **Sentiment analysis integration** (Phase 6) — classify every message
  using any of the trained models (Model Selection), with overall
  sentiment counts/percentages/chart, per-user comparison, a sentiment
  timeline, and live single-message prediction — all local, with a
  visible accuracy disclaimer (see below)
- **Phase 8 professional dashboard** (`app.py`) — a 10-tab Streamlit app
  (Overview, User Analytics, Activity Analysis, Response Latency,
  Hinglish NLP, Sentiment Analysis, Emoji Analysis, Word & Phrase
  Analysis, ML Model Performance, Chat Search) with a sidebar
  file-uploader, date/user filters, a sentiment-model selector,
  interactive Plotly charts, downloadable CSV/JSON exports, a visible
  parsing pipeline status, and defensive error handling throughout
- **Phase 9 chat search & filtering** (`chat_search.py`, wired into the
  "Chat Search" tab) — search the uploaded chat by keyword (plain text
  or regex, case-sensitive optional), sender, message type, an explicit
  date range, and predicted sentiment (once a model has classified the
  chat) — any combination at once. Results are paginated (25/50/100/200
  per page, with Prev/Next) so a huge export never freezes the UI, and
  everything runs locally in-process — no search query or chat content
  is ever sent to an external API. See `tests/test_chat_search.py` for
  keyword/sender/date/sentiment/combined/empty-result/pagination tests.
- **Phase 10 ML model performance dashboard** ("ML Model Performance"
  tab) — accuracy/precision/recall/F1 (macro-averaged) for Logistic
  Regression, Multinomial Naive Bayes, and Random Forest, all read
  straight from `models/model_comparison_report.json` (produced by
  `scripts/train_hinglish_pipeline.py`) — nothing is hardcoded. Shows dataset
  provenance (source files, total rows, class distribution), the
  train/test split size, TF-IDF settings, a per-model confusion matrix
  + full classification report, a grouped comparison bar chart, and a
  plain-language explanation of why the three algorithms score
  differently on the same TF-IDF features. Independently re-running
  `python scripts/train_hinglish_pipeline.py` reproduces the exact same numbers
  (fixed `random_state=42`), confirming the dashboard reflects real
  evaluation, not fabricated metrics.
- Monthly & daily activity timelines
- Busiest day / busiest month activity map + weekly heatmap
- Most active group members (with % share)
- Word cloud & most frequently used words (stop-words removed)
- Emoji usage analysis
- **Database-backed storage** — every analysis run (and its messages +
  predicted sentiment) is saved via SQLite by default, so past runs
  persist between sessions

## Project Structure

```
whatsapp_chat_analyzer/
├── app.py                     # Streamlit app (run this)
├── requirements.txt
├── README.md
├── .gitignore                  # Keeps chat_analysis.db, real chat exports, secrets out of git
├── nlp_preprocessing.py        # Compat shim only -- see its docstring (do not import from it)
├── sample_chat.txt             # Sample chat export to try the app with
│
├── data/
│   ├── sentiment_train.csv           # English-only labeled dataset (see below)
│   ├── generate_dataset.py           # Script that generated sentiment_train.csv
│   ├── hinglish_sentiment_demo.csv   # DEMONSTRATION Hinglish labeled dataset
│   └── build_hinglish_demo_dataset.py  # Script that generated the demo dataset
│
├── models/
│   ├── sentiment_model.joblib                  # Legacy English-only model
│   ├── hinglish_logistic_regression.joblib     # Hinglish-aware LR model
│   ├── hinglish_multinomial_nb.joblib          # Hinglish-aware NB model
│   ├── hinglish_random_forest.joblib           # Hinglish-aware RF model (usually best)
│   ├── hinglish_sentiment_pipeline.joblib      # Copy of the best model, Phase-5 filename
│   └── model_comparison_report.json            # Accuracy/F1/confusion matrix per model
│
├── src/
│   ├── parser.py               # Parses raw .txt export into a DataFrame
│   ├── preprocessing.py        # Hinglish-aware text cleaning pipeline
│   ├── analytics.py            # Chat & user analytics + Plotly activity charts
│   ├── behavioral.py           # Response latency, night-owl, streaks, interaction
│   ├── emoji_analysis.py       # Emoji frequency analysis
│   ├── helper.py               # Stats, timelines, wordcloud
│   ├── chat_search.py          # Phase 9: keyword/sender/date/sentiment search + pagination
│   ├── ml_pipeline.py          # Reusable ML training/evaluation pipeline
│   ├── sentiment.py            # Phase 6: sentiment UI logic, model selection, disclaimer
│   ├── legacy_sentiment_model.py  # Legacy inference wrapper (kept for reference)
│   ├── db.py                   # Opt-in local history layer (SQLite by default)
│   └── common.py               # Shared filter/empty-check helpers used across the above
│
├── scripts/
│   ├── train_hinglish_pipeline.py  # Trains & saves 3 models on the combined dataset
│   └── train_sentiment_model.py    # Trains the original legacy model
│
├── tests/                      # test suite for every module above
└── assets/                     # screenshots / presentation assets (if any)
```

> **Note on `nlp_preprocessing.py`:** the trained models in `models/` were
> pickled with a reference to this exact top-level module name (as the
> TF-IDF preprocessor function). It's kept as a thin backward-compatible
> shim re-exporting `src/preprocessing.py` so the existing `.joblib` files
> keep loading without retraining. New code should import
> `from src import preprocessing` directly, not this shim.

## Setup & Run

1. **Install dependencies** (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the sentiment model** (pre-trained models are already included
   in `models/`, but re-run this any time you update the training data):
   ```bash
   python scripts/train_sentiment_model.py
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser (usually `http://localhost:8501`).

4. **Try it out**: upload `sample_chat.txt` (included) from the sidebar,
   or export a real WhatsApp chat: open a chat in WhatsApp → More options
   → More → Export chat → **Without Media** → save the `.txt` file.

## Sentiment Analysis Integration (Phase 6)

The Streamlit app's Sentiment Analysis section (`src/sentiment.py`)
wires the Phase 5 ML pipeline directly into the UI:

- **Model Selection** — a dropdown lets you pick which trained
  classifier to use: any of the 3 Hinglish-aware models compared in
  Phase 5 (Logistic Regression / Multinomial Naive Bayes / Random
  Forest), the original English-only legacy model, or a simple
  always-available keyword-lexicon fallback. Only classes genuinely
  present in the training data are ever shown — `{'positive',
  'negative', 'neutral'}` — no invented emotion categories.
- **Overall Sentiment** — real counts, percentages, and a distribution
  pie chart for the currently selected user/chat.
- **By User** — a per-user sentiment breakdown table and a grouped bar
  chart comparing sentiment across users.
- **Sentiment Timeline** — a line chart of sentiment counts over time.
- **Try a Message** — type any message (English or Hinglish) and get an
  instant local prediction, with per-class confidence when the chosen
  model supports it.

**⚠️ Disclaimer**, shown directly in the app:
> Sentiment classification is an ML-based estimate and may be inaccurate
> for sarcasm, context-dependent Hinglish, slang, and multi-turn
> conversations.

**Privacy**: every prediction happens locally against a joblib-saved
scikit-learn model already on disk. No chat text or message content is
ever sent to an external API or third-party service.

## How Sentiment Analysis Works (Live App)

As of Phase 6, the app's Sentiment Analysis section uses
`src/sentiment.py`, which lets you pick from any of the models
below (see "Model Selection" above) rather than being hardwired to one.
`scripts/train_sentiment_model.py` (the original Phase-0 script) and
`scripts/train_hinglish_pipeline.py` (Phase 5) both remain available to
(re)train the underlying model files independently — see the Machine
Learning Pipeline section below for how the newer, Hinglish-aware
models are produced.

If no trained model file exists at all, the app automatically falls
back to a simple keyword-based sentiment scorer so it never breaks —
this fallback is always listed in the Model Selection dropdown.

## Machine Learning Pipeline (`ml_pipeline.py`, Phase 5)

This phase adds a genuine, leakage-safe, **reusable** supervised
text-classification pipeline, separate from (and not yet wired into)
the live app's model above.

### Labeled dataset — inspected and documented

A WhatsApp chat you upload has **no ground-truth sentiment labels** —
it's personal, unlabeled data, and this project never pretends
otherwise. Training data must come from a genuinely labeled source.
Two exist in this project, and both are used and clearly documented:

| File | Rows | Source | Notes |
|---|---|---|---|
| `data/sentiment_train.csv` | 152 | Hand-curated by the developer (`generate_dataset.py`) | English-only, generic chat-style sentences, balanced ~50/label. Does **not** exercise Hinglish text. |
| `data/hinglish_sentiment_demo.csv` | 90 | Hand-written by the developer (`build_hinglish_demo_dataset.py`), **new in Phase 5** | Romanized Hindi/Hinglish sentences (e.g. `"mast hai bro"`, `"mood off hai"`), balanced 30/30/30. **Explicitly marked as demonstration data** in its own file header — not collected from real users, not representative of general Hinglish sentiment. |

The two are combined (242 rows total) purely so the pipeline has
code-mixed examples to train/evaluate against. **Any accuracy/F1 number
produced from this combined set describes fit to this small
demonstration-scale dataset only — it is not a scientifically valid
estimate of real-world Hinglish sentiment accuracy.** Swap in a larger,
independently annotated corpus to make a real-world claim; the pipeline
code does not need to change.

### Pipeline

- **Preprocessing**: the Phase 4 `nlp_preprocessing.clean_text` function
  is used directly as the TF-IDF vectorizer's `preprocessor`, so slang
  expansion, stopword removal, and normalization happen before
  vectorization.
- **Feature extraction**: `TfidfVectorizer` with unigram+bigram features
  (`ngram_range=(1, 2)`) by default, and configurable `max_features` /
  `min_df`.
- **Models compared**: Logistic Regression, Multinomial Naive Bayes,
  Random Forest — each built as its own `sklearn.pipeline.Pipeline`
  (TF-IDF → classifier), so every model is trained and evaluated
  identically.
- **No data leakage**: because the vectorizer lives *inside* the
  pipeline, `pipeline.fit(X_train, y_train)` builds the TF-IDF
  vocabulary from the training fold only — the test fold is never seen
  until `.predict()`. This is verified directly by a test that plants a
  token only in the test set and asserts it's absent from the fitted
  vocabulary (`tests/test_ml_pipeline.py`).
- **Evaluation**: accuracy, macro precision/recall/F1, a full per-class
  classification report, and a confusion matrix — all computed via
  `sklearn.metrics` against real held-out predictions, nothing
  hardcoded.
- **Safe persistence**: `save_pipeline_safely()` refuses to write a
  model file if the pipeline isn't actually fitted, or if its label set
  doesn't match what's expected, and writes atomically so a crash
  mid-save can never corrupt an existing model file.

### Running it

```bash
python scripts/train_hinglish_pipeline.py
```

This prints the dataset provenance, a side-by-side comparison table
(accuracy/precision/recall/F1 for all 3 models on the same held-out
split), and each model's confusion matrix, then saves **each of the 3
models individually** (`models/hinglish_logistic_regression.joblib`,
`models/hinglish_multinomial_nb.joblib`, `models/hinglish_random_forest
.joblib` — so the app's Model Selection dropdown can offer all three),
plus the overall best model under `models/hinglish_sentiment_pipeline
.joblib`, and the full metrics to `models/model_comparison_report.json`.

**This does not overwrite `models/sentiment_model.joblib`** (the
original legacy model) — that file is left untouched; it's simply
offered as one more option in the app's Model Selection dropdown
alongside the newer Hinglish-aware models (see Phase 6 below).

## Hinglish-Aware NLP Preprocessing (`src/preprocessing.py`)

This is one of the project's differentiating features: a reusable,
rule-based text-cleaning pipeline built specifically for the kind of
short, informal, **code-mixed** text that shows up in Indian WhatsApp
chats — English, Hindi written in Roman script ("Hinglish"), common
chat slang/abbreviations, and WhatsApp-specific noise like URLs,
@mentions, and elongated letters ("sooooo", "yesss").

**What it is not.** This module does *not* claim to understand Hindi or
Hinglish the way a native speaker or a proper NLP language model would.
It performs no part-of-speech tagging, no transliteration, no language
identification, and no semantic parsing. It cannot resolve ambiguous
romanized spellings or understand sentence grammar. It is a lightweight,
explainable, dictionary- and regex-based **normalization layer** —
appropriate and honest for a student project, not a research-grade
Hindi NLP system.

**What it actually does**, via the `HinglishTextPreprocessor` class:

1. Lowercases text.
2. Strips URLs and `@mentions`.
3. Squeezes runs of 3+ repeated characters ("sooooo" → "soo") — a common
   fix for expressive elongation in chat text.
4. Deletes apostrophes so contractions collapse cleanly ("don't" →
   "dont", "How's" → "hows") instead of leaving stray letters behind.
5. Strips remaining punctuation (optionally keeping emoji characters
   intact).
6. Expands a configurable dictionary of common English/Hinglish chat
   abbreviations to a fuller form (e.g. "kr" → "kar", "rha" → "raha",
   "h" → "hai", "thnx" → "thanks").
7. Removes a configurable stopword list of high-frequency English and
   Hinglish filler words (e.g. "hai", "kya", "the", "is") to surface the
   more meaningful content words — e.g. `"kya scene hai"` → `["scene"]`.

Every stage (stopwords, slang dictionary, URL/mention handling, repeated
-character squeezing, emoji preservation) is independently configurable
via constructor arguments, and the whole pipeline is built to be reused
directly by the ML sentiment pipeline (`src/legacy_sentiment_model.py` /
`scripts/train_sentiment_model.py`) in an upcoming phase — nothing about it is
tied to the Streamlit UI.

See the module docstring in `src/preprocessing.py` and
`tests/test_nlp_preprocessing.py` for the full behavior and examples.

## Privacy & Security (Phase 11)

This app is built to process WhatsApp exports **entirely on your own
machine**:

- **No external APIs.** Nothing in the analysis pipeline (parsing, NLP
  cleaning, sentiment prediction, analytics) calls OpenAI, Anthropic,
  or any other external NLP/analytics service — sentiment prediction
  uses a local `joblib`-saved scikit-learn model, not a hosted API.
  `tests/test_sentiment_analysis.py` explicitly asserts that
  `src/sentiment.py` never imports `requests`, `urllib`,
  `http.client`, `socket`, `openai`, or `anthropic`.
- **Nothing is stored by default.** An uploaded chat is only kept in
  memory for the current session.
- **Local history is opt-in.** The Streamlit UI has a *"Save this
  analysis to local history"* checkbox, off by default. Only when you
  turn it on does the app write parsed messages + a summary to a local
  SQLite file, `chat_analysis.db`, in the project directory. A
  *"Clear saved history"* button in the sidebar wipes it on demand,
  and `db.py` exposes `clear_all_history()` for the same purpose from
  a script or shell.
- **No accidental logging of message content.** The codebase does not
  configure a `logging` file handler, and no `print()`/`st.write()`
  call anywhere in the app or backend modules dumps raw chat text —
  only aggregate stats, tokens, and non-identifying summaries are
  ever printed (e.g. in the offline training scripts).
- **No secrets or API keys** are used or stored anywhere in this
  project — there's nothing to leak.
- **`.gitignore` keeps private data out of version control**: the
  local `chat_analysis.db` (and any `*.db`/`*.sqlite` file), real
  `WhatsApp Chat with *.txt` exports, `.env` files, and common
  credential file patterns are all ignored. Only the synthetic
  `sample_chat.txt` demo file is tracked.
- **"100% local" claim, scoped honestly**: the *analysis pipeline*
  (parsing → NLP → ML sentiment → analytics) never leaves your
  machine and never calls an external service. The *only* thing this
  app ever writes to disk is the opt-in local SQLite history file
  described above — and that file also never leaves your machine.

If you fork or deploy this app, review `db.py` and the sidebar code in
`app.py` before changing the storage behavior, and make sure any new
feature that would call an external service is clearly disclosed to
the user in the UI before it's enabled.

## Switching the Database to MySQL / MongoDB

By default, if you opt in to local history, the app uses a local SQLite
file (`chat_analysis.db`) — zero setup required. To use MySQL instead:

1. `pip install sqlalchemy pymysql`
2. Set the environment variable before running the app:
   ```bash
   export DATABASE_URL="mysql+pymysql://user:password@localhost/whatsapp_analyzer"
   streamlit run app.py
   ```
`db.py` already contains the logic to pick this up automatically — no
other file needs to change.

(For MongoDB, swap the functions in `db.py` to use `pymongo` — the rest
of the app only calls `db.save_analysis_run()` and `db.fetch_past_runs()`,
so the storage engine underneath can be changed independently.)

## Notes / Limitations

- The bundled sentiment training set is intentionally small (for a fast,
  fully offline demo).
- `src/preprocessing.py` is a rule-based normalization layer, not a true
  Hindi/Hinglish language-understanding system — see the dedicated
  section above for exactly what it does and doesn't do.
- The Phase 5/6 ML models are genuinely leakage-safe and their metrics
  are real, but the combined dataset they train on is demonstration-
  scale (242 rows, partly hand-written by the developer) — their
  accuracy numbers describe fit to that dataset, not a validated,
  general-purpose Hinglish sentiment benchmark. Sentiment predictions
  are estimates and can be wrong, especially on sarcasm and
  context-dependent slang — see the in-app disclaimer.
- The behavioral-analytics metrics (response latency, night-owl,
  conversation starters, streaks, interaction) are purely descriptive
  timing/structural statistics — they don't measure or imply interest,
  personality, emotional state, or relationship strength.
- Works best with English/Hinglish text; heavy use of other languages may
  reduce sentiment accuracy.
- Designed for single-file, offline batch analysis of an exported chat
  (not a live/real-time WhatsApp integration).

## Running the Tests

```bash
pip install pytest
pytest tests/ -v
```
