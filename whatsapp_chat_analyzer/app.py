"""
app.py
------
PHASE 8 -- Professional Streamlit Dashboard.

WhatsApp Chat Analysis -- a full analytics dashboard built entirely on
top of the existing, already-tested backend modules:

    parser.py           -> parses the raw .txt export
    analytics.py        -> chat & user analytics + activity figures
    behavioral.py       -> response latency / night-owl / streaks / interaction
    preprocessing.py    -> Hinglish-aware text cleaning pipeline
    sentiment.py        -> ML sentiment classification + charts
    helper.py           -> stats, timelines, word cloud
    emoji_analysis.py   -> emoji frequency analysis
    chat_search.py      -> keyword/sender/date/sentiment search + pagination
    db.py                -> SQLite persistence of past runs

This file is purely the presentation/orchestration layer: it does not
re-implement any analytics logic that already lives in the modules
above -- it wires their existing, pure functions up to a polished,
multi-tab Streamlit UI with filters, download buttons, loading
indicators and error handling.

Complete workflow demonstrated end-to-end on every upload:
    TXT upload -> parsing -> preprocessing -> analytics -> ML -> visualization

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.src import analytics
from scripts.src import behavioral
from scripts.src import chat_search
from scripts.src import db
from scripts.src import emoji_analysis
from scripts.src import helper
from scripts.src import parser
from scripts.src import preprocessing
from scripts.src import sentiment

# ===========================================================================
# Page configuration + light visual polish
# ===========================================================================
st.set_page_config(
    page_title="WhatsApp Chat Analytics",
    page_icon="\U0001F4AC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background-color: rgba(37, 211, 102, 0.07);
            border: 1px solid rgba(37, 211, 102, 0.25);
            border-radius: 10px;
            padding: 12px 14px 8px 14px;
        }
        div[data-testid="stMetricLabel"] {font-weight: 600;}
        .streamlit-expanderHeader {font-weight: 600;}
        section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,0.2);}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_REPORT_PATH = os.path.join(BASE_DIR, "models", "model_comparison_report.json")
SAMPLE_CHAT_PATH = os.path.join(BASE_DIR, "sample_chat.txt")

BRAND_GREEN = "#25D366"
BRAND_DARK = "#128C7E"


# ===========================================================================
# Cached wrappers around the (unmodified) backend functions.
# Caching lives here, in the presentation layer, purely for dashboard
# responsiveness -- none of the underlying analytics logic is touched.
# ===========================================================================
@st.cache_data(show_spinner=False)
def cached_preprocess(raw_text: str) -> pd.DataFrame:
    return parser.preprocess(raw_text)


@st.cache_data(show_spinner=False)
def cached_predict(messages: tuple, model_path: str) -> list:
    return sentiment.predict_messages(list(messages), model_path)


@st.cache_data(show_spinner=False)
def cached_clean_tokens(messages: tuple) -> list:
    """Runs every message through the default Hinglish-aware cleaning
    pipeline and returns the flat list of resulting tokens."""
    tokens: list = []
    for msg in messages:
        tokens.extend(preprocessing.tokenize_text(msg))
    return tokens


@st.cache_data(show_spinner=False)
def cached_ml_report(path: str, mtime: float):
    """`mtime` is only part of the cache key so the report is reloaded
    if the underlying JSON file ever changes on disk."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_ml_report():
    if not os.path.exists(ML_REPORT_PATH):
        return None
    return cached_ml_report(ML_REPORT_PATH, os.path.getmtime(ML_REPORT_PATH))


def df_download_button(dframe: pd.DataFrame, label: str, file_name: str, key: str):
    """Small helper so every tab can offer a CSV download the same way."""
    if dframe is None or dframe.empty:
        return
    st.download_button(
        label=label,
        data=dframe.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


# ===========================================================================
# Header
# ===========================================================================
st.title("\U0001F4AC WhatsApp Chat Analytics Dashboard")
st.caption(
    "Upload an exported WhatsApp chat (.txt) for statistics, behavioral "
    "insights, Hinglish-aware NLP, and ML-powered sentiment analysis -- "
    "every computation runs locally on this machine; nothing is uploaded "
    "anywhere."
)

with st.expander("\U0001F512 Privacy & data handling -- please read"):
    st.markdown(
        """
This app is designed to process your chat **entirely on this machine**:

- **No external calls.** Parsing, NLP cleaning, sentiment prediction and
  all analytics run with local, offline models (scikit-learn / joblib).
  Nothing in this codebase sends chat text to OpenAI, Anthropic, or any
  other external API or analytics service -- there are no network calls
  in the analysis pipeline at all.
- **Nothing is stored by default.** Your uploaded file is only kept in
  memory for this session. It is **not** written to disk unless you
  explicitly tick *"Save this analysis to local history"* in the
  sidebar below.
- **Optional local history is opt-in.** If you turn it on, a summary
  and the parsed messages are saved to a SQLite file
  (`chat_analysis.db`) **on this computer only** -- still never sent
  anywhere -- so you can revisit past runs. You can clear it at any
  time with the *"Clear saved history"* button in the sidebar.
- **You're in control of the source file.** The exported `.txt` file
  itself lives wherever you saved it on your device; this app does not
  copy or relocate it.

If you're running this app yourself, `.gitignore` is set up to keep
real chat exports, the local database, and any secrets out of version
control -- see the repo's Privacy & Security section in `README.md`.
        """
    )

# ===========================================================================
# Sidebar -- upload
# ===========================================================================
with st.sidebar:
    st.header("\U0001F4C1 Data")
    uploaded_file = st.file_uploader(
        "WhatsApp chat export (.txt)", type=["txt"],
        help="WhatsApp app -> Chat -> More options -> Export chat -> Without Media.",
    )

    use_sample = False
    if os.path.exists(SAMPLE_CHAT_PATH):
        use_sample = st.button("\U0001F4C4 Try it with the bundled sample chat")

    st.caption("\U0001F512 All processing happens locally on this machine. "
               "Your chat is never sent to an external server or API.")

    st.divider()
    st.subheader("\U0001F4BE Local history (optional)")
    save_to_history = st.checkbox(
        "Save this analysis to local history",
        value=False,
        help=(
            "Off by default. When enabled, the parsed messages and a "
            "summary are written to a local SQLite file (chat_analysis.db) "
            "on this machine so you can browse past runs. Nothing is ever "
            "sent off this device."
        ),
    )
    if st.button("\U0001F5D1\uFE0F Clear saved history"):
        deleted = db.clear_all_history()
        st.success(f"Cleared {deleted} saved run(s) from local history.")
        st.rerun()

# Resolve the raw text to analyze (uploaded file takes priority)
raw_text = None
source_name = None
if uploaded_file is not None:
    try:
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        source_name = uploaded_file.name
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
elif use_sample:
    try:
        with open(SAMPLE_CHAT_PATH, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        source_name = "sample_chat.txt"
    except OSError as exc:
        st.error(f"Could not read the bundled sample chat: {exc}")
        st.stop()

if raw_text is None:
    st.info(
        "\U0001F446 Upload a WhatsApp `.txt` export from the sidebar to get started "
        "(or try the bundled sample chat)."
    )
    st.markdown(
        "**What you'll get:** overview stats \u00b7 per-user analytics \u00b7 activity "
        "timelines \u00b7 response-latency & behavioral patterns \u00b7 Hinglish NLP "
        "preprocessing \u00b7 ML-based sentiment analysis \u00b7 emoji analysis \u00b7 "
        "word/phrase frequency \u00b7 ML model performance \u00b7 full-text chat search."
    )
    st.stop()

# ---------------------------------------------------------------------
# Stage 1-2: Parsing + preprocessing, with a visible pipeline status
# ---------------------------------------------------------------------
with st.status("Running chat-processing pipeline...", expanded=False) as status:
    st.write("\U0001F4E5 Reading uploaded file...")
    st.write("\U0001F50D Parsing WhatsApp export format...")
    raw_df = cached_preprocess(raw_text)

    if raw_df.empty:
        status.update(label="Parsing failed", state="error")
        st.error(
            "Couldn't find any recognizable WhatsApp messages in this file. "
            "Please make sure you uploaded an **unedited** WhatsApp chat "
            "export (.txt) -- exports with media attachments included, or "
            "files edited in a word processor, often break the expected "
            "line format."
        )
        st.stop()

    st.write("\U0001F9F9 Validating structure & building dataframe...")
    status.update(label=f"Chat processed successfully ({len(raw_df)} lines)", state="complete")

# ---------------------------------------------------------------------
# Sidebar -- filters that depend on the parsed data
# ---------------------------------------------------------------------
all_users = sorted(u for u in raw_df["user"].unique() if u != "group_notification")
min_date = raw_df["only_date"].min()
max_date = raw_df["only_date"].max()

with st.sidebar:
    st.divider()
    st.header("\U0001F39B\uFE0F Filters")

    date_range = st.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
        help="Restrict every chart/table below to this date range.",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    selected_users = st.multiselect(
        "Include users", options=all_users, default=all_users,
        help="Uncheck a name to exclude that participant from every analysis below.",
    )

    focus_choices = ["Overall"] + selected_users
    selected_user = st.selectbox(
        "Focus on", focus_choices,
        help="'Overall' analyzes the whole (filtered) group; picking a name "
             "drills into that person's messages specifically.",
    )

    st.divider()
    st.header("\U0001F9E0 Sentiment Model")
    available_sentiment_models = sentiment.available_models()
    model_display_name = st.selectbox(
        "Classifier", list(available_sentiment_models.keys()),
        help="'Hinglish-aware' models were trained on a combined English + "
             "Hinglish demo dataset. 'Legacy' is English-only. The keyword "
             "lexicon is a simple always-available fallback.",
    )
    sentiment_model_path = available_sentiment_models[model_display_name]

    st.divider()
    st.header("\U0001F553 Behavioral Thresholds")
    max_response_minutes = st.slider(
        "Max response window (min)", min_value=5, max_value=720,
        value=behavioral.DEFAULT_MAX_RESPONSE_MINUTES, step=5,
        help="A reply after a longer gap is treated as a new, unrelated "
             "exchange and excluded from response-time numbers.",
    )
    session_gap_minutes = st.slider(
        "Session gap (min)", min_value=5, max_value=720,
        value=behavioral.DEFAULT_SESSION_GAP_MINUTES, step=5,
        help="A message sent after at least this much silence starts a new "
             "conversation session.",
    )

# ---------------------------------------------------------------------
# Apply filters -> the single dataframe every tab below works from
# ---------------------------------------------------------------------
if not selected_users:
    st.warning("Select at least one user in the sidebar to see the analysis.")
    st.stop()

date_filtered = raw_df[
    (raw_df["only_date"] >= start_date) & (raw_df["only_date"] <= end_date)
]
df = date_filtered[
    date_filtered["user"].isin(selected_users + ["group_notification"])
].reset_index(drop=True)

if df.empty:
    st.warning("No messages match the current date/user filters. Try widening them.")
    st.stop()

analysis_df = df[df["user"] != "group_notification"].copy()
if selected_user != "Overall":
    analysis_df = analysis_df[analysis_df["user"] == selected_user]

st.caption(
    f"Analyzing **{len(analysis_df):,}** messages from **{len(selected_users)}** "
    f"user(s) between **{start_date}** and **{end_date}** "
    f"(focus: **{selected_user}**) \u2014 source: `{source_name}`"
)

# ===========================================================================
# Top-level tabs
# ===========================================================================
(
    tab_overview, tab_users, tab_activity, tab_latency, tab_nlp,
    tab_sentiment, tab_emoji, tab_words, tab_ml, tab_search,
) = st.tabs([
    "\U0001F3E0 Overview",
    "\U0001F465 User Analytics",
    "\U0001F4C8 Activity Analysis",
    "\u23F1\uFE0F Response Latency",
    "\U0001F5E3\uFE0F Hinglish NLP",
    "\U0001F642 Sentiment Analysis",
    "\U0001F600 Emoji Analysis",
    "\U0001F4DD Word & Phrase Analysis",
    "\U0001F916 ML Model Performance",
    "\U0001F50E Chat Search",
])

# ---------------------------------------------------------------------------
# TAB 1 -- OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    try:
        num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user, df)
        ov_stats = analytics.overall_stats(
            df if selected_user == "Overall" else df[df["user"] == selected_user]
        )
    except Exception as exc:
        st.error(f"Could not compute overview statistics: {exc}")
    else:
        st.subheader("Top Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", f"{num_messages:,}")
        c2.metric("Words", f"{num_words:,}")
        c3.metric("Media Shared", f"{num_media:,}")
        c4.metric("Links Shared", f"{num_links:,}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Participants", ov_stats["total_users"])
        c6.metric("Avg. Message Length", f'{ov_stats["avg_message_length"]} words')
        c7.metric("Most Active User", ov_stats["most_active_user"] or "\u2014")

        st.divider()
        oc1, oc2 = st.columns(2)
        with oc1:
            st.plotly_chart(analytics.fig_messages_by_date(df, selected_user),
                             use_container_width=True)
        with oc2:
            st.plotly_chart(analytics.fig_messages_by_user(df), use_container_width=True)

        with st.expander("\U0001F4C5 Busiest day / month at a glance"):
            busy_day = helper.week_activity_map(selected_user, df)
            busy_month = helper.month_activity_map(selected_user, df)
            bc1, bc2 = st.columns(2)
            bc1.metric("Busiest Day", busy_day.idxmax() if not busy_day.empty else "\u2014")
            bc2.metric("Busiest Month", busy_month.idxmax() if not busy_month.empty else "\u2014")

        df_download_button(
            analysis_df[["date", "user", "message", "message_type"]],
            "\u2B07\uFE0F Download filtered messages (CSV)",
            "filtered_chat.csv", "dl_overview_raw",
        )

# ---------------------------------------------------------------------------
# TAB 2 -- USER ANALYTICS
# ---------------------------------------------------------------------------
with tab_users:
    try:
        stats_table = analytics.user_stats(df)
    except Exception as exc:
        st.error(f"Could not compute per-user statistics: {exc}")
        stats_table = pd.DataFrame()

    if stats_table.empty:
        st.info("No per-user statistics available for the current filters.")
    else:
        st.markdown("**Per-user statistics** \u2014 message/word counts, average "
                    "message length, conversation share, and each user's single "
                    "most active hour/day.")
        st.dataframe(stats_table, use_container_width=True, hide_index=True)
        df_download_button(stats_table, "\u2B07\uFE0F Download user stats (CSV)",
                            "user_stats.csv", "dl_user_stats")

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Most Messages**")
            st.dataframe(analytics.rank_by_messages(df), use_container_width=True, hide_index=True)
            st.markdown("**Longest Average Messages**")
            st.dataframe(analytics.rank_by_avg_length(df), use_container_width=True, hide_index=True)
        with r2:
            st.markdown("**Most Words**")
            st.dataframe(analytics.rank_by_words(df), use_container_width=True, hide_index=True)
            st.markdown("**Peak-Hour Volume**")
            st.dataframe(analytics.rank_by_active_hours(df), use_container_width=True, hide_index=True)

        if selected_user == "Overall" and len(selected_users) > 1:
            st.divider()
            st.markdown("**Most Busy Users** (share of conversation)")
            counts, percent_df = helper.most_busy_users(df)
            bu1, bu2 = st.columns([2, 1])
            with bu1:
                fig = px.bar(x=counts.index, y=counts.values,
                             labels={"x": "User", "y": "Messages"},
                             title="Messages per User", color_discrete_sequence=[BRAND_DARK])
                st.plotly_chart(fig, use_container_width=True)
            with bu2:
                st.dataframe(percent_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 -- ACTIVITY ANALYSIS
# ---------------------------------------------------------------------------
with tab_activity:
    try:
        act1, act2 = st.columns(2)
        with act1:
            st.plotly_chart(analytics.fig_messages_by_month(df, selected_user), use_container_width=True)
        with act2:
            st.plotly_chart(analytics.fig_messages_by_dow(df, selected_user), use_container_width=True)
        act3, act4 = st.columns(2)
        with act3:
            st.plotly_chart(analytics.fig_messages_by_hour(df, selected_user), use_container_width=True)
        with act4:
            daily = helper.daily_timeline(selected_user, df)
            if daily.empty:
                st.plotly_chart(go.Figure(), use_container_width=True)
            else:
                fig = px.line(daily, x="only_date", y="message", markers=True,
                              title="Daily Message Timeline",
                              color_discrete_sequence=[BRAND_GREEN])
                fig.update_layout(xaxis_title="Date", yaxis_title="Messages")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Monthly Timeline**")
        monthly = helper.monthly_timeline(selected_user, df)
        if not monthly.empty:
            fig = px.line(monthly, x="time", y="message", markers=True,
                          title="Monthly Message Timeline", color_discrete_sequence=[BRAND_DARK])
            fig.update_layout(xaxis_title="Month", yaxis_title="Messages")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Weekly Activity Heatmap** (day of week \u00d7 hour of day)")
        heatmap = helper.activity_heatmap(selected_user, df)
        if not heatmap.empty:
            fig = px.imshow(
                heatmap, aspect="auto", color_continuous_scale="Greens",
                labels=dict(x="Hour", y="Day", color="Messages"),
                title="When Is This Chat Most Active?",
            )
            st.plotly_chart(fig, use_container_width=True)

        df_download_button(daily, "\u2B07\uFE0F Download daily timeline (CSV)",
                            "daily_timeline.csv", "dl_daily_timeline")
    except Exception as exc:
        st.error(f"Could not render activity charts: {exc}")

# ---------------------------------------------------------------------------
# TAB 4 -- RESPONSE LATENCY (+ nested behavioral analyses)
# ---------------------------------------------------------------------------
with tab_latency:
    st.caption(
        "These are purely timing/structural communication metrics computed "
        "from message timestamps. They do **not** measure or imply interest, "
        "personality, emotional state, or relationship strength \u2014 a slow "
        "reply can simply mean someone was asleep, busy, or offline."
    )

    lat_tab, night_tab, starters_tab, streaks_tab, interaction_tab = st.tabs(
        ["Response Latency", "Night Owl", "Conversation Starters",
         "Activity Streaks", "User Interaction"]
    )

    with lat_tab:
        try:
            lat_stats = behavioral.response_latency_stats(df, max_response_minutes)
        except Exception as exc:
            st.error(f"Could not compute response latency: {exc}")
        else:
            if lat_stats["sample_size"] == 0:
                st.info("Not enough consecutive different-user messages within the "
                        "configured response window to compute response latency.")
            else:
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("Avg. Response Time", f'{lat_stats["average_response_minutes"]} min')
                l2.metric("Median Response Time", f'{lat_stats["median_response_minutes"]} min')
                l3.metric("Fastest Response", f'{lat_stats["fastest_response_seconds"]} sec')
                l4.metric("Slowest Response", f'{round(lat_stats["slowest_response_seconds"] / 60, 2)} min')
                st.caption(f'Based on {lat_stats["sample_size"]} reply events within a '
                          f'{lat_stats["max_response_window_minutes"]}-minute window.')

                by_user = behavioral.response_latency_by_user(df, max_response_minutes)
                st.markdown("**Response time by user**")
                st.dataframe(by_user, use_container_width=True, hide_index=True)
                df_download_button(by_user, "\u2B07\uFE0F Download (CSV)", "latency_by_user.csv", "dl_lat_user")

                between = behavioral.response_latency_between_users(df, max_response_minutes)
                st.markdown("**Response time between users**")
                if between.empty:
                    st.caption("No user pair has enough reply events yet to show a meaningful average.")
                else:
                    st.dataframe(between, use_container_width=True, hide_index=True)
                    df_download_button(between, "\u2B07\uFE0F Download (CSV)", "latency_between_users.csv", "dl_lat_pair")

    with night_tab:
        try:
            n_stats = behavioral.night_owl_stats(df)
            n1, n2 = st.columns(2)
            n1.metric("Night Messages (12 AM\u20135 AM)", n_stats["total_night_messages"])
            n2.metric("% of All Messages", f'{n_stats["pct_of_total"]}%')
            nc1, nc2 = st.columns(2)
            with nc1:
                st.plotly_chart(behavioral.fig_night_activity_by_hour(df), use_container_width=True)
            with nc2:
                st.plotly_chart(behavioral.fig_night_owl_by_user(df), use_container_width=True)
            night_table = behavioral.night_owl_by_user(df)
            st.dataframe(night_table, use_container_width=True, hide_index=True)
            df_download_button(night_table, "\u2B07\uFE0F Download (CSV)", "night_owl.csv", "dl_night")
        except Exception as exc:
            st.error(f"Could not compute night-owl analysis: {exc}")

    with starters_tab:
        try:
            st.caption(
                "Methodology: a message starts a new conversation if it follows "
                "a silence gap longer than the configured session gap (sidebar), "
                "or is the very first message in the chat. A timing heuristic, "
                "not a claim about who 'really' initiated contact."
            )
            total_sessions = behavioral.count_conversation_sessions(df, session_gap_minutes)
            st.metric("Conversation Sessions Detected", total_sessions)
            starters_table = behavioral.conversation_starters(df, session_gap_minutes)
            st.dataframe(starters_table, use_container_width=True, hide_index=True)
            df_download_button(starters_table, "\u2B07\uFE0F Download (CSV)", "conversation_starters.csv", "dl_starters")
        except Exception as exc:
            st.error(f"Could not compute conversation starters: {exc}")

    with streaks_tab:
        try:
            chat_streaks = behavioral.chat_activity_streaks(df)
            s1, s2, s3 = st.columns(3)
            s1.metric("Longest Active Streak", f'{chat_streaks["longest_streak_days"]} days')
            s2.metric("Current Streak", f'{chat_streaks["current_streak_days"]} days')
            s3.metric("Total Active Days", chat_streaks["total_active_days"])
            if chat_streaks["longest_streak_start"] is not None:
                st.caption(f'Longest streak: {chat_streaks["longest_streak_start"]} \u2192 '
                          f'{chat_streaks["longest_streak_end"]}')
            streaks_table = behavioral.user_activity_streaks(df)
            st.dataframe(streaks_table, use_container_width=True, hide_index=True)
            df_download_button(streaks_table, "\u2B07\uFE0F Download (CSV)", "activity_streaks.csv", "dl_streaks")
        except Exception as exc:
            st.error(f"Could not compute activity streaks: {exc}")

    with interaction_tab:
        try:
            st.plotly_chart(behavioral.fig_interaction_heatmap(df, max_response_minutes),
                             use_container_width=True)
            pairs_table = behavioral.top_interaction_pairs(df, max_response_minutes)
            st.markdown("**Most frequent reply pairs**")
            st.dataframe(pairs_table, use_container_width=True, hide_index=True)
            df_download_button(pairs_table, "\u2B07\uFE0F Download (CSV)", "interaction_pairs.csv", "dl_pairs")
        except Exception as exc:
            st.error(f"Could not compute user interaction map: {exc}")

# ---------------------------------------------------------------------------
# TAB 5 -- HINGLISH NLP
# ---------------------------------------------------------------------------
with tab_nlp:
    st.markdown(
        "A reusable, rule-based text-cleaning pipeline for code-mixed "
        "English/Hinglish chat text. It normalizes surface noise "
        "(case, URLs, mentions, repeated letters, punctuation), expands "
        "common chat slang/abbreviations, and removes filler stopwords -- "
        "**not** a linguistic parser or a claim of true language understanding."
    )

    with st.expander("\u2728 Try the pipeline on your own text"):
        sample_text = st.text_input("Message", placeholder="e.g. bhai kya kr rha h?? check karo plzz")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            opt_lower = st.checkbox("Lowercase", value=True)
            opt_urls = st.checkbox("Remove URLs", value=True)
        with oc2:
            opt_mentions = st.checkbox("Remove mentions", value=True)
            opt_repeat = st.checkbox("Squeeze repeated chars", value=True)
        with oc3:
            opt_emoji = st.checkbox("Preserve emojis", value=True)
            opt_stop = st.checkbox("Remove stopwords", value=True)

        if sample_text and sample_text.strip():
            pre = preprocessing.HinglishTextPreprocessor(
                lowercase=opt_lower, remove_urls=opt_urls, remove_mentions=opt_mentions,
                normalize_repeated_chars=opt_repeat, preserve_emojis=opt_emoji,
                remove_stopwords=opt_stop,
            )
            result_rows = [
                ("Original", sample_text),
                ("Normalized", pre.normalize(sample_text)),
                ("Tokens", ", ".join(pre.tokenize(sample_text)) or "\u2014"),
                ("Cleaned (vectorizer-ready)", pre.clean(sample_text)),
                ("Emojis found", ", ".join(preprocessing.extract_emojis(sample_text)) or "\u2014"),
                ("URLs found", ", ".join(preprocessing.extract_urls(sample_text)) or "\u2014"),
                ("Mentions found", ", ".join(preprocessing.extract_mentions(sample_text)) or "\u2014"),
            ]
            st.table(pd.DataFrame(result_rows, columns=["Stage", "Result"]))

    st.divider()
    st.markdown("**Corpus-wide cleaning stats** (this chat, current filters)")
    try:
        messages_tuple = tuple(analysis_df["message"].tolist())
        raw_word_counts = [len(m.split()) for m in messages_tuple]
        cleaned_tokens = cached_clean_tokens(messages_tuple)

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Messages Analyzed", f"{len(messages_tuple):,}")
        cc2.metric("Avg. Raw Words/Message",
                   f"{(sum(raw_word_counts) / len(raw_word_counts)):.2f}" if raw_word_counts else "0")
        cc3.metric("Avg. Cleaned Tokens/Message",
                   f"{(len(cleaned_tokens) / len(messages_tuple)):.2f}" if messages_tuple else "0")

        if cleaned_tokens:
            top_tokens = pd.DataFrame(Counter(cleaned_tokens).most_common(20),
                                       columns=["token", "count"])
            wc1, wc2 = st.columns([2, 1])
            with wc1:
                fig = px.bar(top_tokens, x="token", y="count",
                             title="Most Frequent Cleaned Tokens",
                             color_discrete_sequence=[BRAND_GREEN])
                st.plotly_chart(fig, use_container_width=True)
            with wc2:
                st.dataframe(top_tokens, use_container_width=True, hide_index=True)
            df_download_button(top_tokens, "\u2B07\uFE0F Download tokens (CSV)",
                                "cleaned_tokens.csv", "dl_tokens")
        else:
            st.info("Not enough text left after cleaning to show a token frequency table.")

        st.markdown("**Slang/abbreviations detected in this chat**")
        raw_words_lower = set()
        for m in messages_tuple:
            raw_words_lower.update(re.findall(r"[a-zA-Z]+", m.lower()))
        detected_slang = [
            {"slang": k, "expands_to": v}
            for k, v in preprocessing.DEFAULT_SLANG_DICT.items()
            if k in raw_words_lower
        ]
        if detected_slang:
            st.dataframe(pd.DataFrame(detected_slang), use_container_width=True, hide_index=True)
        else:
            st.caption("No known slang/abbreviations from the built-in dictionary were detected.")
    except Exception as exc:
        st.error(f"Could not run the Hinglish NLP corpus analysis: {exc}")

# ---------------------------------------------------------------------------
# TAB 6 -- SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------
sentiment_labels = []
with tab_sentiment:
    st.warning(f"\u26A0\uFE0F **Disclaimer:** {sentiment.DISCLAIMER}")
    st.caption(
        "Every classifier was trained locally on labeled datasets bundled "
        "with this project. Your chat never leaves this machine -- no "
        "external API is used for prediction."
    )

    try:
        with st.spinner(f"Classifying messages with '{model_display_name}'..."):
            sentiment_labels = cached_predict(
                tuple(analysis_df["message"].tolist()), sentiment_model_path
            )
        analysis_df = analysis_df.copy()
        analysis_df["sentiment"] = sentiment_labels
    except Exception as exc:
        st.error(f"Sentiment classification failed: {exc}")
        sentiment_labels = []

    if sentiment_labels:
        overall_sent_tab, user_sent_tab, timeline_sent_tab, predict_tab = st.tabs(
            ["Overall Sentiment", "By User", "Sentiment Timeline", "Try a Message"]
        )

        with overall_sent_tab:
            stats = sentiment.overall_sentiment_stats(sentiment_labels)
            if stats["total"] == 0:
                st.info("No messages to classify for this selection.")
            else:
                metric_cols = st.columns(len(stats["label_order"]) + 1)
                metric_cols[0].metric("Total Classified", stats["total"])
                for i, label in enumerate(stats["label_order"], start=1):
                    metric_cols[i].metric(
                        label.capitalize(),
                        f'{stats["counts"][label]} ({stats["percentages"][label]}%)',
                    )
                st.plotly_chart(sentiment.fig_sentiment_distribution(sentiment_labels),
                                 use_container_width=True)

        with user_sent_tab:
            user_sent_table = sentiment.user_sentiment_stats(analysis_df)
            st.dataframe(user_sent_table, use_container_width=True, hide_index=True)
            st.plotly_chart(sentiment.fig_user_sentiment_comparison(analysis_df),
                             use_container_width=True)
            df_download_button(user_sent_table, "\u2B07\uFE0F Download (CSV)",
                                "sentiment_by_user.csv", "dl_sent_user")

        with timeline_sent_tab:
            st.plotly_chart(sentiment.fig_sentiment_timeline(analysis_df),
                             use_container_width=True)

        with predict_tab:
            st.caption("Type any message (English/Hinglish) to see what the "
                      "selected classifier predicts. Runs entirely locally.")
            user_text = st.text_area("Message", placeholder="e.g. mast hai bro")
            if st.button("Predict Sentiment"):
                if user_text and user_text.strip():
                    try:
                        pred_label, pred_confidence = sentiment.predict_single_with_confidence(
                            user_text, sentiment_model_path
                        )
                        st.success(f"Predicted sentiment: **{pred_label.capitalize()}**")
                        if pred_confidence:
                            conf_df = pd.DataFrame(
                                {"sentiment": list(pred_confidence.keys()),
                                 "confidence": list(pred_confidence.values())}
                            ).sort_values("confidence", ascending=False)
                            st.dataframe(conf_df, use_container_width=True, hide_index=True)
                        st.caption(sentiment.DISCLAIMER)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                else:
                    st.warning("Please type a message first.")

        df_download_button(
            analysis_df[["date", "user", "message", "sentiment"]],
            "\u2B07\uFE0F Download sentiment-labeled messages (CSV)",
            "sentiment_labeled.csv", "dl_sent_all",
        )

# ---------------------------------------------------------------------------
# TAB 7 -- EMOJI ANALYSIS
# ---------------------------------------------------------------------------
with tab_emoji:
    try:
        emoji_df = emoji_analysis.emoji_helper(selected_user, df)
    except Exception as exc:
        st.error(f"Could not compute emoji analysis: {exc}")
        emoji_df = pd.DataFrame()

    if emoji_df.empty:
        st.info("No emojis found for the current filters.")
    else:
        ec1, ec2 = st.columns([1, 2])
        with ec1:
            st.dataframe(emoji_df, use_container_width=True, hide_index=True)
            df_download_button(emoji_df, "\u2B07\uFE0F Download (CSV)", "emoji_counts.csv", "dl_emoji")
        with ec2:
            top = emoji_df.head(10)
            fig = px.pie(top, names="emoji", values="count", title="Top Emoji Usage")
            st.plotly_chart(fig, use_container_width=True)

        if selected_user == "Overall" and len(selected_users) > 1:
            with st.expander("\U0001F464 Top emoji per user"):
                rows = []
                for u in selected_users:
                    u_emoji = emoji_analysis.emoji_helper(u, df)
                    if not u_emoji.empty:
                        top3 = ", ".join(f'{r.emoji} ({r["count"]})' for _, r in u_emoji.head(3).iterrows())
                        rows.append({"user": u, "top_emojis": top3})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No per-user emoji usage to show.")

# ---------------------------------------------------------------------------
# TAB 8 -- WORD & PHRASE ANALYSIS
# ---------------------------------------------------------------------------
with tab_words:
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown("**Word Cloud**")
        try:
            with st.spinner("Generating word cloud..."):
                fig = helper.create_wordcloud(selected_user, df)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Not enough text to generate a word cloud.")
        except Exception as exc:
            st.error(f"Could not generate word cloud: {exc}")
    with wc2:
        st.markdown("**Most Common Words**")
        common_words = helper.most_common_words(selected_user, df)
        st.dataframe(common_words, use_container_width=True, hide_index=True)
        df_download_button(common_words, "\u2B07\uFE0F Download (CSV)", "common_words.csv", "dl_words")

    st.divider()
    st.markdown("**Most Common Phrases (bigrams)** \u2014 computed from the Hinglish-aware "
               "cleaning pipeline's tokens")
    try:
        messages_tuple = tuple(analysis_df["message"].tolist())
        cleaned_tokens = cached_clean_tokens(messages_tuple)
        bigrams = [f"{a} {b}" for a, b in zip(cleaned_tokens, cleaned_tokens[1:])]
        if bigrams:
            top_bigrams = pd.DataFrame(Counter(bigrams).most_common(20), columns=["phrase", "count"])
            pc1, pc2 = st.columns([2, 1])
            with pc1:
                fig = px.bar(top_bigrams, x="phrase", y="count", title="Top 20 Phrases",
                             color_discrete_sequence=[BRAND_DARK])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            with pc2:
                st.dataframe(top_bigrams, use_container_width=True, hide_index=True)
            df_download_button(top_bigrams, "\u2B07\uFE0F Download (CSV)", "top_phrases.csv", "dl_phrases")
        else:
            st.info("Not enough cleaned text to extract meaningful phrases.")
    except Exception as exc:
        st.error(f"Could not compute phrase analysis: {exc}")

# ---------------------------------------------------------------------------
# TAB 9 -- ML MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab_ml:
    report = load_ml_report()
    if report is None:
        st.info(
            "No model comparison report found. Run `python train_hinglish_pipeline.py` "
            "once to train the models and generate real accuracy/F1/confusion-matrix metrics."
        )
    else:
        ds = report.get("dataset", {})
        settings = report.get("settings", {})
        models = report.get("models", {})

        st.markdown("### \U0001F4CA Training Dataset")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Labeled Rows", ds.get("total_rows", "\u2014"))
        label_counts = ds.get("label_counts", {})
        d2.metric("Positive / Neutral / Negative",
                  f'{label_counts.get("positive", 0)} / {label_counts.get("neutral", 0)} / '
                  f'{label_counts.get("negative", 0)}')
        # n_train / n_test are identical across models (same split), so read
        # them once from whichever model report is present.
        any_model = next(iter(models.values()), {})
        n_train = any_model.get("n_train", "\u2014")
        n_test = any_model.get("n_test", "\u2014")
        d3.metric("Train / Test Split", f"{n_train} / {n_test}")
        test_size_pct = settings.get("test_size")
        d4.metric("Best Model", report.get("best_model", "\u2014").replace("_", " ").title())

        if ds.get("sources"):
            src_df = pd.DataFrame(
                [{"source file": k, "rows": v} for k, v in ds["sources"].items()]
            )
            with st.expander("\U0001F4C1 Dataset sources"):
                st.dataframe(src_df, use_container_width=True, hide_index=True)
        if ds.get("disclaimer"):
            st.caption(f'\u26A0\uFE0F {ds["disclaimer"]}')

        if label_counts:
            with st.expander("\U0001F4C8 Class distribution (full labeled dataset)"):
                dist_df = pd.DataFrame(
                    [{"class": k, "count": v} for k, v in label_counts.items()]
                )
                fig_dist = px.bar(
                    dist_df, x="class", y="count", color="class",
                    title="Label distribution across the full training dataset",
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        if settings:
            st.caption(
                f'TF-IDF settings: max_features={settings.get("max_features")}, '
                f'min_df={settings.get("min_df")}, ngram_range={tuple(settings.get("ngram_range", []))} \u2014 '
                f'split: test_size={test_size_pct} '
                f'({n_train} train rows / {n_test} held-out test rows), '
                f'random_state={settings.get("random_state")}'
            )
        st.caption(
            "The TF-IDF vectorizer is fit on the training split **only**, then "
            "applied to the held-out test split \u2014 the test messages are never "
            "seen during vectorizer fitting or model training, so the metrics "
            "below reflect genuine held-out performance, not memorization."
        )

        st.divider()
        st.markdown("### \U0001F3C6 Held-out Test Set Performance")
        st.caption("Real metrics computed by `train_hinglish_pipeline.py` / `ml_pipeline.py` "
                   "and saved to `models/model_comparison_report.json` at training time -- "
                   "nothing on this tab is hardcoded.")
        if not models:
            st.info("The report does not contain any model results.")
        else:
            summary_rows = [
                {
                    "model": name.replace("_", " ").title(),
                    "accuracy": round(m.get("accuracy", 0.0), 4),
                    "precision_macro": round(m.get("precision_macro", 0.0), 4),
                    "recall_macro": round(m.get("recall_macro", 0.0), 4),
                    "f1_macro": round(m.get("f1_macro", 0.0), 4),
                    "n_train": m.get("n_train", "\u2014"),
                    "n_test": m.get("n_test", "\u2014"),
                }
                for name, m in models.items()
            ]
            summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)

            mcols = st.columns(len(summary_rows))
            for col, row in zip(mcols, summary_df.to_dict("records")):
                col.metric(row["model"], f'{row["accuracy"]:.1%} acc',
                           f'F1 {row["f1_macro"]:.3f}')

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            metrics_long = summary_df.melt(
                id_vars="model",
                value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                var_name="metric", value_name="score",
            )
            fig = px.bar(metrics_long, x="model", y="score", color="metric", barmode="group",
                         title="Model Comparison \u2014 Accuracy / Precision / Recall / F1 (macro-averaged, test set)")
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Confusion matrix per model**")
            cm_tabs = st.tabs([name.replace("_", " ").title() for name in models.keys()])
            for cm_tab, (name, model_report) in zip(cm_tabs, models.items()):
                with cm_tab:
                    cm = model_report.get("confusion_matrix")
                    labels = model_report.get("labels", [])
                    if cm and labels:
                        fig_cm = px.imshow(
                            cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Greens",
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            title=f'Confusion Matrix \u2014 {name.replace("_", " ").title()} '
                                  f'({model_report.get("n_test", "?")} test samples)',
                        )
                        st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{name}")
                    else:
                        st.info("No confusion matrix stored for this model.")

                    cls_report = model_report.get("classification_report")
                    if cls_report:
                        cls_df = (
                            pd.DataFrame(cls_report).T.reset_index()
                            .rename(columns={"index": "class"})
                        )
                        st.dataframe(cls_df, use_container_width=True, hide_index=True)

            st.download_button(
                "\u2B07\uFE0F Download full report (JSON)",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name="model_comparison_report.json", mime="application/json",
                key="dl_ml_report",
            )

        with st.expander("\U0001F9E0 Why do Logistic Regression, Naive Bayes, and Random "
                         "Forest perform differently on the same TF-IDF features?"):
            st.markdown(
                "All three models see the **same input**: a sparse, high-dimensional "
                "TF-IDF matrix (unigrams + bigrams, weighted by term frequency and "
                "inverse document frequency). They differ in *how* they use it, "
                "which is why their accuracy/F1 numbers above aren't identical:\n\n"
                "- **Logistic Regression** fits a linear decision boundary directly "
                "over the TF-IDF weights. TF-IDF vectors are exactly the kind of "
                "sparse, (roughly) linearly-separable, high-dimensional input linear "
                "models tend to do well on -- each word/bigram gets its own weight, "
                "and the model doesn't need many examples per feature to learn a "
                "useful coefficient. It also outputs calibrated-ish probabilities, "
                "which is why it's used for the 'confidence' shown elsewhere in the app.\n\n"
                "- **Multinomial Naive Bayes** explicitly assumes every feature "
                "(word/bigram) is conditionally independent given the class -- an "
                "assumption text data violates constantly (words co-occur for a "
                "reason). It's fast and a solid baseline, but that independence "
                "assumption is usually the reason it trails Logistic Regression on "
                "the same features, especially once bigrams start correlating with "
                "the unigrams they contain.\n\n"
                "- **Random Forest** builds many decision trees that split on "
                "individual TF-IDF columns. Trees can capture non-linear "
                "interactions between features that a linear model can't -- but "
                "each tree only ever looks at a handful of the thousands of sparse "
                "TF-IDF columns per split, so on very high-dimensional, mostly-zero "
                "text features it can be less sample-efficient than a linear model, "
                "and more prone to overfitting small training sets. On a small, "
                "demonstration-scale dataset like this one, its relative ranking "
                "against Logistic Regression can go either way -- both are visible "
                "in the table above.\n\n"
                "None of this says one algorithm is universally 'better' for text "
                "classification -- it's a function of dataset size, class balance, "
                "and how linearly separable this particular vocabulary happens to "
                "be. That's exactly why this tab compares them empirically instead "
                "of asserting it."
            )

        st.divider()
        st.markdown("**Prediction distribution across models, on *your* uploaded chat**")
        st.caption(
            "This is NOT an accuracy measurement (your chat has no ground-truth "
            "sentiment labels) -- it simply shows how each trained model classifies "
            "the same messages, so you can see where they agree or disagree."
        )
        try:
            comparison_rows = []
            messages_list = analysis_df["message"].tolist()
            for name, path in available_sentiment_models.items():
                if not messages_list:
                    continue
                preds = cached_predict(tuple(messages_list), path)
                counts = Counter(preds)
                for label, count in counts.items():
                    comparison_rows.append({"model": name, "sentiment": label, "count": count})
            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                fig = px.bar(comp_df, x="model", y="count", color="sentiment", barmode="group",
                             title="Predicted Sentiment Distribution by Model")
                fig.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No messages available under the current filters to compare.")
        except Exception as exc:
            st.error(f"Could not compute the live model comparison: {exc}")

# ---------------------------------------------------------------------------
# TAB 10 -- CHAT SEARCH  (PHASE 9)
# ---------------------------------------------------------------------------
with tab_search:
    st.markdown(
        "Search within the currently filtered chat (respects the sidebar's "
        "user filter, but ignores the 'Focus on' selection). Everything "
        "below runs locally in this session -- your search text and chat "
        "content are never sent to an external API."
    )

    sc1, sc2, sc3 = st.columns([3, 1, 1])
    with sc1:
        query = st.text_input(
            "Keyword", placeholder="e.g. birthday, exam, movie...",
            help="Leave blank to browse/filter without a text query.",
        )
    with sc2:
        use_regex = st.checkbox("Regex", value=False)
    with sc3:
        case_sensitive = st.checkbox("Case-sensitive", value=False)

    fc1, fc2 = st.columns(2)
    with fc1:
        search_users = st.multiselect(
            "Sender", options=selected_users, default=selected_users,
        )
        type_options = sorted(df["message_type"].dropna().unique().tolist())
        search_types = st.multiselect(
            "Message type", options=type_options, default=type_options,
        )
    with fc2:
        search_date_range = st.date_input(
            "Date range", value=(min_date, max_date),
            min_value=min_date, max_value=max_date,
            help="Independent of the sidebar's date filter -- narrows "
                 "search results to this range specifically.",
            key="search_date_range",
        )
        if isinstance(search_date_range, tuple) and len(search_date_range) == 2:
            search_start_date, search_end_date = search_date_range
        else:
            search_start_date, search_end_date = min_date, max_date

        sentiment_series = None
        sentiment_filter_values = None
        if sentiment_labels:
            sentiment_series = chat_search.build_sentiment_series(
                analysis_df.index, sentiment_labels
            )
            available_sentiments = sorted(set(sentiment_labels))
            sentiment_filter_values = st.multiselect(
                "Sentiment", options=available_sentiments,
                default=available_sentiments,
                help="Only messages the current classifier has scored "
                     "(under the 'Sentiment Analysis' tab) can match this.",
            )
        else:
            st.caption(
                "\u2139\uFE0F Sentiment filter unavailable -- open the "
                "'Sentiment Analysis' tab first to classify messages."
            )

    try:
        results = chat_search.search_messages(
            df,
            keyword=query,
            use_regex=use_regex,
            case_sensitive=case_sensitive,
            senders=search_users,
            message_types=search_types,
            start_date=search_start_date,
            end_date=search_end_date,
            sentiment=sentiment_series,
            sentiment_filter=sentiment_filter_values,
        )
        search_error = None
    except re.error as exc:
        results = df.iloc[0:0]
        search_error = f"Invalid regular expression: {exc}"
    except Exception as exc:
        results = df.iloc[0:0]
        search_error = f"Search failed: {exc}"

    if search_error:
        st.error(search_error)
    elif not search_users:
        st.warning("Select at least one sender above to search.")
    else:
        if sentiment_series is not None:
            results = results.assign(
                sentiment=sentiment_series.reindex(results.index).fillna("\u2014")
            )
        results = results.sort_values("date")
        st.metric("Matches Found", f"{len(results):,}")

        if results.empty:
            st.caption(
                "No messages matched your search."
                if query or (sentiment_filter_values is not None
                             and len(sentiment_filter_values) < len(set(sentiment_labels or [])))
                else "No messages match the current filters."
            )
        else:
            # ---- Pagination: only ever render one page, however large
            # the full match set is, so a huge chat can't freeze the UI.
            pc1, pc2 = st.columns([1, 3])
            with pc1:
                page_size = st.selectbox(
                    "Results per page", chat_search.PAGE_SIZE_OPTIONS,
                    index=chat_search.PAGE_SIZE_OPTIONS.index(chat_search.DEFAULT_PAGE_SIZE),
                    key="search_page_size",
                )

            filters_signature = (
                query, use_regex, case_sensitive, tuple(sorted(search_users)),
                tuple(sorted(search_types)), search_start_date, search_end_date,
                tuple(sorted(sentiment_filter_values)) if sentiment_filter_values else None,
                page_size,
            )
            if st.session_state.get("_search_sig") != filters_signature:
                st.session_state["_search_sig"] = filters_signature
                st.session_state["search_page"] = 1

            page_df, total_pages, total_results = chat_search.paginate(
                results, st.session_state.get("search_page", 1), page_size
            )

            with pc2:
                nav1, nav2, nav3 = st.columns([1, 2, 1])
                with nav1:
                    if st.button("\u2190 Prev", disabled=st.session_state["search_page"] <= 1):
                        st.session_state["search_page"] -= 1
                        st.rerun()
                with nav2:
                    st.markdown(
                        f"<div style='text-align:center'>Page "
                        f"<b>{st.session_state['search_page']}</b> of "
                        f"<b>{total_pages}</b> &nbsp;({total_results:,} results)</div>",
                        unsafe_allow_html=True,
                    )
                with nav3:
                    if st.button("Next \u2192", disabled=st.session_state["search_page"] >= total_pages):
                        st.session_state["search_page"] += 1
                        st.rerun()

            display_cols = chat_search.DISPLAY_COLUMNS.copy()
            if "sentiment" in results.columns:
                display_cols.append("sentiment")
            st.dataframe(
                page_df[display_cols], use_container_width=True, hide_index=True,
            )
            df_download_button(
                results[display_cols], "\u2B07\uFE0F Download all matches (CSV)",
                "search_results.csv", "dl_search",
            )

# ===========================================================================
# Persist this run to the database -- opt-in only (see sidebar checkbox).
# Nothing is written to disk unless the user explicitly asked for it.
# ===========================================================================
st.divider()
if save_to_history:
    try:
        if sentiment_labels:
            run_id = db.save_analysis_run(source_name, analysis_df, sentiment_labels)
            st.success(
                f"\u2705 Analysis saved to local history (run #{run_id}). "
                "Stored only in chat_analysis.db on this machine."
            )
    except Exception as exc:
        st.warning(f"Could not save this run to local history: {exc}")
else:
    st.caption(
        "\U0001F512 Local history is off, so this analysis was not written "
        "to disk. Enable \"Save this analysis to local history\" in the "
        "sidebar if you'd like to keep it for next time."
    )

with st.sidebar:
    st.divider()
    st.subheader("\U0001F4DC Past Analysis Runs")
    try:
        past_runs = db.fetch_past_runs()
    except Exception:
        past_runs = []
    if past_runs:
        past_df = pd.DataFrame(
            past_runs,
            columns=["id", "file", "run_at", "messages", "users", "positive", "neutral", "negative"],
        )
        st.dataframe(past_df.set_index("id"), use_container_width=True)
    else:
        st.caption("No analysis runs saved yet.")
