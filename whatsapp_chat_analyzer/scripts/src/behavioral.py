"""
behavioral.py
-------------
Phase 3: Advanced Behavioral Analytics.

Everything in this module measures *timing and structural patterns* in
how messages were exchanged: who replied to whom, how quickly, at what
hour, how consistently across days, and who tends to send the
session-opening message. These are descriptive communication/behavioral
metrics computed purely from message timestamps and senders.

IMPORTANT -- what these numbers do NOT mean:
    A response-latency, night-owl, "conversation starter", streak, or
    interaction number is NOT evidence of interest, personality,
    emotional state, or the strength/quality of a relationship between
    users. A slow reply can mean someone was asleep, at work, or had no
    signal -- not that they "care less". A user who starts more sessions
    isn't necessarily more invested -- they might just have a schedule
    that puts them online first. Treat every metric here as neutral
    communication-cadence data and nothing more. UI/report text built on
    top of this module should describe results the same way.

Design mirrors analytics.py: every function is pure (DataFrame in ->
DataFrame/dict/Figure out), takes no Streamlit dependency, is defensive
against empty/single-user/single-message chats, and never raises.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import common

__all__ = [
    "compute_response_events",
    "response_latency_stats",
    "response_latency_by_user",
    "response_latency_between_users",
    "night_owl_stats",
    "night_owl_by_user",
    "night_activity_by_hour",
    "fig_night_activity_by_hour",
    "fig_night_owl_by_user",
    "conversation_starters",
    "count_conversation_sessions",
    "chat_activity_streaks",
    "user_activity_streaks",
    "interaction_matrix",
    "top_interaction_pairs",
    "fig_interaction_heatmap",
    "DEFAULT_MAX_RESPONSE_MINUTES",
    "DEFAULT_SESSION_GAP_MINUTES",
    "NIGHT_HOURS",
]

# Configurable defaults -- all exposed as function parameters so the UI
# (or a caller) can tune them per chat rather than being locked in.
DEFAULT_MAX_RESPONSE_MINUTES = 180   # a reply after >3h silence isn't a "timed response"
DEFAULT_SESSION_GAP_MINUTES = 60     # silence longer than this starts a new "conversation"
DEFAULT_MIN_SAMPLES_FOR_PAIR = 3     # minimum reply events before a user-pair average is shown
NIGHT_HOURS = range(0, 5)            # 12:00 AM - 4:59 AM


# ---------------------------------------------------------------------------
# Internal helpers
# (Shared with analytics.py / helper.py via common.py -- kept as thin
# local aliases so the rest of this module doesn't need to change.)
# ---------------------------------------------------------------------------

_is_empty = common.is_empty
_real_messages = common.exclude_system_messages


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


def _longest_streak(active_days: list):
    """Given a sorted list of unique `date` objects, returns
    (longest_streak_length, streak_start_date, streak_end_date) for the
    longest run of consecutive calendar days."""
    if not active_days:
        return 0, None, None

    longest_len = 1
    longest_start = active_days[0]
    longest_end = active_days[0]

    run_start = active_days[0]
    run_len = 1
    for i in range(1, len(active_days)):
        if (active_days[i] - active_days[i - 1]).days == 1:
            run_len += 1
        else:
            if run_len > longest_len:
                longest_len, longest_start, longest_end = run_len, run_start, active_days[i - 1]
            run_start = active_days[i]
            run_len = 1
    if run_len > longest_len:
        longest_len, longest_start, longest_end = run_len, run_start, active_days[-1]

    return longest_len, longest_start, longest_end


def _current_streak(active_days: list) -> int:
    """Length of the consecutive-day run ending at the last active day."""
    if not active_days:
        return 0
    streak = 1
    for i in range(len(active_days) - 1, 0, -1):
        if (active_days[i] - active_days[i - 1]).days == 1:
            streak += 1
        else:
            break
    return streak


# ---------------------------------------------------------------------------
# 1. Response Latency
# ---------------------------------------------------------------------------

def compute_response_events(
    df: pd.DataFrame,
    max_response_minutes: int = DEFAULT_MAX_RESPONSE_MINUTES,
) -> pd.DataFrame:
    """Builds one row per "reply event": a message sent by a different
    user than the immediately preceding real message, within
    `max_response_minutes` of it.

    Design choices (why this is safe on messy real chats):
      - Consecutive messages from the SAME user are ignored -- those are
        follow-ups/continuations, not replies to someone else.
      - A gap larger than `max_response_minutes` is treated as the start
        of an unrelated new conversation rather than a slow "response",
        so an overnight silence never gets counted as a multi-hour
        "response time" data point -- it's simply excluded.
      - `max_response_minutes` is fully configurable by the caller.

    Returns
    -------
    pd.DataFrame with columns:
        original_sender, replier, original_time, reply_time,
        response_seconds, response_minutes
    """
    cols = ["original_sender", "replier", "original_time", "reply_time",
            "response_seconds", "response_minutes"]
    data = _real_messages(df)
    if _is_empty(data) or len(data) < 2:
        return pd.DataFrame(columns=cols)

    data = data.sort_values("date", kind="mergesort").reset_index(drop=True)

    prev_user = data["user"].shift(1)
    prev_time = data["date"].shift(1)
    curr_user = data["user"]
    curr_time = data["date"]

    delta_seconds = (curr_time - prev_time).dt.total_seconds()

    different_user = prev_user.notna() & (prev_user != curr_user)
    within_window = delta_seconds <= (max_response_minutes * 60)
    non_negative = delta_seconds >= 0  # guards against any out-of-order timestamps
    mask = different_user & within_window & non_negative

    events = pd.DataFrame({
        "original_sender": prev_user[mask].values,
        "replier": curr_user[mask].values,
        "original_time": prev_time[mask].values,
        "reply_time": curr_time[mask].values,
        "response_seconds": delta_seconds[mask].values,
    })
    if events.empty:
        return pd.DataFrame(columns=cols)

    events["response_minutes"] = (events["response_seconds"] / 60).round(2)
    return events.reset_index(drop=True)


def response_latency_stats(
    df: pd.DataFrame,
    max_response_minutes: int = DEFAULT_MAX_RESPONSE_MINUTES,
) -> dict:
    """Chat-wide response-time summary. Purely a timing metric -- see
    module docstring: does not indicate interest/personality/emotion.

    Returns dict with: sample_size, average/median/fastest/slowest
    response time (in both seconds and minutes), and the
    max_response_window_minutes used to compute it.
    """
    events = compute_response_events(df, max_response_minutes)
    if events.empty:
        return {
            "sample_size": 0,
            "average_response_seconds": None,
            "average_response_minutes": None,
            "median_response_seconds": None,
            "median_response_minutes": None,
            "fastest_response_seconds": None,
            "slowest_response_seconds": None,
            "max_response_window_minutes": max_response_minutes,
        }

    secs = events["response_seconds"]
    return {
        "sample_size": int(len(events)),
        "average_response_seconds": round(float(secs.mean()), 2),
        "average_response_minutes": round(float(secs.mean()) / 60, 2),
        "median_response_seconds": round(float(secs.median()), 2),
        "median_response_minutes": round(float(secs.median()) / 60, 2),
        "fastest_response_seconds": round(float(secs.min()), 2),
        "slowest_response_seconds": round(float(secs.max()), 2),
        "max_response_window_minutes": max_response_minutes,
    }


def response_latency_by_user(
    df: pd.DataFrame,
    max_response_minutes: int = DEFAULT_MAX_RESPONSE_MINUTES,
) -> pd.DataFrame:
    """Per-user response-time summary, for the events where THAT user is
    the one replying. A purely descriptive timing metric (see module
    docstring).

    Columns: user, replies_count, avg/median/fastest/slowest_response_minutes.
    """
    cols = ["user", "replies_count", "avg_response_minutes",
            "median_response_minutes", "fastest_response_minutes",
            "slowest_response_minutes"]
    events = compute_response_events(df, max_response_minutes)
    if events.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for user, g in events.groupby("replier"):
        secs = g["response_seconds"]
        rows.append({
            "user": user,
            "replies_count": int(len(g)),
            "avg_response_minutes": round(float(secs.mean()) / 60, 2),
            "median_response_minutes": round(float(secs.median()) / 60, 2),
            "fastest_response_minutes": round(float(secs.min()) / 60, 2),
            "slowest_response_minutes": round(float(secs.max()) / 60, 2),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(
        "avg_response_minutes"
    ).reset_index(drop=True)


def response_latency_between_users(
    df: pd.DataFrame,
    max_response_minutes: int = DEFAULT_MAX_RESPONSE_MINUTES,
    min_samples: int = DEFAULT_MIN_SAMPLES_FOR_PAIR,
) -> pd.DataFrame:
    """Directional (original_sender -> replier) response-time averages,
    restricted to pairs with at least `min_samples` observed reply
    events -- an average built from 1-2 data points isn't a meaningful
    per-pair number, so thinner pairs are omitted entirely rather than
    shown with false precision.

    Columns: original_sender, replier, replies_count,
    avg_response_minutes, median_response_minutes.
    """
    cols = ["original_sender", "replier", "replies_count",
            "avg_response_minutes", "median_response_minutes"]
    events = compute_response_events(df, max_response_minutes)
    if events.empty:
        return pd.DataFrame(columns=cols)

    grouped = (
        events.groupby(["original_sender", "replier"])["response_seconds"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    grouped = grouped[grouped["count"] >= min_samples].copy()
    if grouped.empty:
        return pd.DataFrame(columns=cols)

    grouped["avg_response_minutes"] = (grouped["mean"] / 60).round(2)
    grouped["median_response_minutes"] = (grouped["median"] / 60).round(2)
    grouped = grouped.rename(columns={"count": "replies_count"})
    return grouped[cols].sort_values("replies_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Night Owl Analysis (12:00 AM - 5:00 AM)
# ---------------------------------------------------------------------------

def night_owl_stats(df: pd.DataFrame) -> dict:
    """Chat-wide night-time (12:00 AM-4:59 AM) messaging volume."""
    data = _real_messages(df)
    if _is_empty(data):
        return {"total_night_messages": 0, "total_messages": 0, "pct_of_total": 0.0}

    total_messages = int(len(data))
    total_night = int(data["hour"].isin(NIGHT_HOURS).sum())
    pct = round(total_night / total_messages * 100, 2) if total_messages else 0.0
    return {
        "total_night_messages": total_night,
        "total_messages": total_messages,
        "pct_of_total": pct,
    }


def night_owl_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user night-time messaging breakdown.

    Columns: user, night_messages, total_messages,
    pct_of_user_messages (this user's own night share),
    pct_of_all_night_messages (this user's share of ALL night messages).
    """
    cols = ["user", "night_messages", "total_messages",
            "pct_of_user_messages", "pct_of_all_night_messages"]
    data = _real_messages(df)
    if _is_empty(data):
        return pd.DataFrame(columns=cols)

    total_night = int(data["hour"].isin(NIGHT_HOURS).sum())
    rows = []
    for user, g in data.groupby("user"):
        night_count = int(g["hour"].isin(NIGHT_HOURS).sum())
        total = int(len(g))
        rows.append({
            "user": user,
            "night_messages": night_count,
            "total_messages": total,
            "pct_of_user_messages": round(night_count / total * 100, 2) if total else 0.0,
            "pct_of_all_night_messages": round(night_count / total_night * 100, 2) if total_night else 0.0,
        })
    return pd.DataFrame(rows, columns=cols).sort_values(
        "night_messages", ascending=False
    ).reset_index(drop=True)


def night_activity_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Message counts for each night hour (0-4). Columns: hour, messages."""
    hours = list(NIGHT_HOURS)
    data = _real_messages(df)
    if _is_empty(data):
        return pd.DataFrame({"hour": hours, "messages": [0] * len(hours)})

    counts = data["hour"].value_counts().reindex(hours).fillna(0).astype(int).reset_index()
    counts.columns = ["hour", "messages"]
    return counts


def fig_night_activity_by_hour(df: pd.DataFrame) -> go.Figure:
    """Bar chart of message volume for each night hour (12 AM-4 AM)."""
    counts = night_activity_by_hour(df)
    if counts["messages"].sum() == 0:
        return _empty_figure("No night-time messages (12 AM\u20135 AM) found")

    fig = px.bar(counts, x="hour", y="messages",
                 title="Night-Owl Activity by Hour (12 AM \u2013 5 AM)")
    fig.update_layout(xaxis_title="Hour", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    fig.update_xaxes(dtick=1)
    return fig


def fig_night_owl_by_user(df: pd.DataFrame) -> go.Figure:
    """Bar chart of night-time message counts per user."""
    data = night_owl_by_user(df)
    if data.empty or data["night_messages"].sum() == 0:
        return _empty_figure("No night-time messages (12 AM\u20135 AM) found")

    fig = px.bar(data, x="user", y="night_messages", title="Night-Owl Messages by User")
    fig.update_layout(xaxis_title="User", yaxis_title="Night Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    return fig


# ---------------------------------------------------------------------------
# 3. Conversation Starters
# ---------------------------------------------------------------------------

def conversation_starters(
    df: pd.DataFrame,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> pd.DataFrame:
    """Estimates which users most frequently send the message that opens
    a new conversation "session".

    Methodology (silence-gap heuristic, fully documented since this is
    an estimate, not a ground truth):
      1. Real messages are sorted chronologically.
      2. The gap since the previous message (from ANY user) is computed
         for every message.
      3. A message is considered the start of a new conversation session
         if that gap exceeds `session_gap_minutes`, OR it is the very
         first message in the chat.
      4. The sender of each session-starting message is credited with
         having "started" that session.
    This is a structural/timing heuristic based on silence gaps -- it
    does NOT determine who raised a topic, who "really" initiated
    contact outside the chat, or anything about intent. It only measures
    who happened to send the first message after a period of chat
    inactivity.

    Columns: user, conversations_started, pct_of_conversations.
    """
    cols = ["user", "conversations_started", "pct_of_conversations"]
    data = _real_messages(df)
    if _is_empty(data):
        return pd.DataFrame(columns=cols)

    data = data.sort_values("date", kind="mergesort").reset_index(drop=True)
    gap_seconds = data["date"].diff().dt.total_seconds()
    is_start = gap_seconds.isna() | (gap_seconds > session_gap_minutes * 60)

    total_sessions = int(is_start.sum())
    if total_sessions == 0:
        return pd.DataFrame(columns=cols)

    counts = data.loc[is_start, "user"].value_counts().reset_index()
    counts.columns = ["user", "conversations_started"]
    counts["pct_of_conversations"] = round(
        counts["conversations_started"] / total_sessions * 100, 2
    )
    return counts.sort_values("conversations_started", ascending=False).reset_index(drop=True)


def count_conversation_sessions(
    df: pd.DataFrame,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> int:
    """Total number of conversation sessions detected (see
    `conversation_starters` for the exact methodology)."""
    data = _real_messages(df)
    if _is_empty(data):
        return 0
    data = data.sort_values("date", kind="mergesort").reset_index(drop=True)
    gap_seconds = data["date"].diff().dt.total_seconds()
    is_start = gap_seconds.isna() | (gap_seconds > session_gap_minutes * 60)
    return int(is_start.sum())


# ---------------------------------------------------------------------------
# 4. Activity Streaks
# ---------------------------------------------------------------------------

def chat_activity_streaks(df: pd.DataFrame) -> dict:
    """Chat-wide consecutive-day activity streaks -- how many calendar
    days in a row had at least one message.

    Returns dict with: longest_streak_days, longest_streak_start,
    longest_streak_end, current_streak_days (the run ending on the last
    active day in the data), total_active_days.
    """
    data = _real_messages(df)
    if _is_empty(data):
        return {
            "longest_streak_days": 0, "longest_streak_start": None,
            "longest_streak_end": None, "current_streak_days": 0,
            "total_active_days": 0,
        }

    active_days = sorted(set(data["only_date"]))
    longest_len, longest_start, longest_end = _longest_streak(active_days)
    return {
        "longest_streak_days": int(longest_len),
        "longest_streak_start": longest_start,
        "longest_streak_end": longest_end,
        "current_streak_days": int(_current_streak(active_days)),
        "total_active_days": int(len(active_days)),
    }


def user_activity_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user consecutive-day activity streaks.

    Columns: user, longest_streak_days, longest_streak_start,
    longest_streak_end, total_active_days.
    """
    cols = ["user", "longest_streak_days", "longest_streak_start",
            "longest_streak_end", "total_active_days"]
    data = _real_messages(df)
    if _is_empty(data):
        return pd.DataFrame(columns=cols)

    rows = []
    for user, g in data.groupby("user"):
        active_days = sorted(set(g["only_date"]))
        longest_len, longest_start, longest_end = _longest_streak(active_days)
        rows.append({
            "user": user,
            "longest_streak_days": int(longest_len),
            "longest_streak_start": longest_start,
            "longest_streak_end": longest_end,
            "total_active_days": int(len(active_days)),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(
        "longest_streak_days", ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. User Interaction
# ---------------------------------------------------------------------------

def interaction_matrix(
    df: pd.DataFrame,
    max_response_minutes: int = DEFAULT_MAX_RESPONSE_MINUTES,
) -> pd.DataFrame:
    """user x user reply-count matrix: rows = replier, columns =
    original_sender, values = how many times the row-user replied to the
    column-user's message (using the same reply-event definition as
    `compute_response_events`). Purely a count of reply patterns -- not a
    measure of closeness or relationship strength."""
    events = compute_response_events(df, max_response_minutes)
    if events.empty:
        return pd.DataFrame()
    return pd.crosstab(events["replier"], events["original_sender"])


def top_interaction_pairs(
    df: pd.DataFrame,
    max_response_minutes: int = DEFAULT_MAX_RESPONSE_MINUTES,
    top_n: int = 10,
) -> pd.DataFrame:
    """Most frequent (original_sender -> replier) reply pairs.

    Columns: original_sender, replier, reply_count.
    """
    cols = ["original_sender", "replier", "reply_count"]
    events = compute_response_events(df, max_response_minutes)
    if events.empty:
        return pd.DataFrame(columns=cols)

    counts = (
        events.groupby(["original_sender", "replier"]).size()
        .reset_index(name="reply_count")
    )
    return counts.sort_values("reply_count", ascending=False).head(top_n).reset_index(drop=True)


def fig_interaction_heatmap(
    df: pd.DataFrame,
    max_response_minutes: int = DEFAULT_MAX_RESPONSE_MINUTES,
) -> go.Figure:
    """Heatmap of the user-interaction matrix (who replies to whom, and
    how often)."""
    matrix = interaction_matrix(df, max_response_minutes)
    if matrix.empty:
        return _empty_figure("Not enough reply data to build an interaction map")

    fig = px.imshow(
        matrix, text_auto=True, aspect="auto", color_continuous_scale="Greens",
        labels=dict(x="Original Sender", y="Replier", color="Reply Count"),
        title="User Interaction Map (who replies to whom)",
    )
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    return fig
