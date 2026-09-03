"""
analytics.py
------------
Phase 2: Chat & User Analytics.

Turns the DataFrame produced by preprocessor.preprocess() into:
  - Overall chat statistics (overall_stats)
  - Per-user statistics (user_stats)
  - Per-user hour/day activity distributions (user_hour_distribution,
    user_day_distribution)
  - Interactive Plotly figures for activity analysis (fig_messages_by_*)
  - User rankings (rank_by_messages, rank_by_words, rank_by_avg_length,
    rank_by_active_hours)

Every function is pure (DataFrame in -> DataFrame/dict/Figure out), takes
no Streamlit dependency, and is defensive against the edge cases a real
chat export can produce: a completely empty chat, a chat with only one
user, a chat where every message is a system notification, etc. Nothing
here is hardcoded -- every number is derived from the `df` passed in.

No sentiment/ML logic lives in this module (that's a separate phase);
this module only touches structural fields the parser guarantees:
`user`, `message`, `message_type`, `date`, `only_date`, `year`,
`month_num`, `month`, `day_name`, `hour`.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import common

__all__ = [
    "overall_stats",
    "user_stats",
    "user_hour_distribution",
    "user_day_distribution",
    "fig_messages_by_date",
    "fig_messages_by_month",
    "fig_messages_by_dow",
    "fig_messages_by_hour",
    "fig_messages_by_user",
    "rank_by_messages",
    "rank_by_words",
    "rank_by_avg_length",
    "rank_by_active_hours",
]

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]

_USER_STATS_COLUMNS = [
    "user", "message_count", "word_count", "avg_message_length",
    "pct_of_total", "most_active_hour", "most_active_day",
]


# ---------------------------------------------------------------------------
# Internal helpers
# (Shared with behavioral.py / helper.py via common.py -- kept as thin
# local aliases so the rest of this module doesn't need to change.)
# ---------------------------------------------------------------------------

_is_empty = common.is_empty
_real_messages = common.exclude_system_messages
_filter_user = common.filter_by_user


def _word_counts(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().apply(len)


def _count_links(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).apply(lambda m: len(_URL_RE.findall(m))).sum())


def _count_media(df: pd.DataFrame) -> int:
    if "message_type" in df.columns:
        return int((df["message_type"] == "media").sum())
    return int(df["message"].fillna("").str.contains("<Media omitted>", na=False).sum())


def _empty_figure(message: str) -> go.Figure:
    """A styled placeholder figure so the UI never breaks on empty/edge
    case data -- callers can always `st.plotly_chart()` the result."""
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
# Overall Statistics
# ---------------------------------------------------------------------------

def overall_stats(df: pd.DataFrame) -> dict:
    """Real, computed-from-data overall statistics for the whole chat
    (or a single user's slice, if pre-filtered by the caller).

    Returns
    -------
    dict with keys:
        total_messages, total_users, total_words, avg_message_length,
        media_messages, links, most_active_user
    """
    data = _real_messages(df)

    if _is_empty(data):
        return {
            "total_messages": 0,
            "total_users": 0,
            "total_words": 0,
            "avg_message_length": 0.0,
            "media_messages": 0,
            "links": 0,
            "most_active_user": None,
        }

    total_messages = int(len(data))
    total_users = int(data["user"].nunique())
    total_words = int(_word_counts(data["message"]).sum())
    avg_message_length = round(total_words / total_messages, 2) if total_messages else 0.0
    media_messages = _count_media(data)
    links = _count_links(data["message"])
    most_active_user = data["user"].value_counts().idxmax()

    return {
        "total_messages": total_messages,
        "total_users": total_users,
        "total_words": total_words,
        "avg_message_length": avg_message_length,
        "media_messages": media_messages,
        "links": links,
        "most_active_user": most_active_user,
    }


# ---------------------------------------------------------------------------
# Per-User Analytics
# ---------------------------------------------------------------------------

def user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user statistics table, one row per real (non-system) sender.

    Columns: user, message_count, word_count, avg_message_length,
    pct_of_total, most_active_hour, most_active_day.

    For the fine-grained distributions behind `most_active_hour` /
    `most_active_day`, see `user_hour_distribution` / `user_day_distribution`.
    """
    data = _real_messages(df)
    if _is_empty(data):
        return pd.DataFrame(columns=_USER_STATS_COLUMNS)

    total_messages = len(data)
    rows = []
    for user, group in data.groupby("user"):
        message_count = int(len(group))
        word_count = int(_word_counts(group["message"]).sum())
        avg_len = round(word_count / message_count, 2) if message_count else 0.0
        pct = round(message_count / total_messages * 100, 2) if total_messages else 0.0

        most_active_hour = (
            int(group["hour"].value_counts().idxmax())
            if "hour" in group.columns and not group["hour"].dropna().empty
            else None
        )
        most_active_day = (
            group["day_name"].value_counts().idxmax()
            if "day_name" in group.columns and not group["day_name"].dropna().empty
            else None
        )

        rows.append({
            "user": user,
            "message_count": message_count,
            "word_count": word_count,
            "avg_message_length": avg_len,
            "pct_of_total": pct,
            "most_active_hour": most_active_hour,
            "most_active_day": most_active_day,
        })

    out = pd.DataFrame(rows, columns=_USER_STATS_COLUMNS)
    return out.sort_values("message_count", ascending=False).reset_index(drop=True)


def user_hour_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """user x hour(0-23) message-count matrix -- the full 'active hours'
    picture per user, useful for heatmaps/detailed views."""
    data = _real_messages(df)
    if _is_empty(data):
        return pd.DataFrame(columns=list(range(24)))
    pivot = data.pivot_table(index="user", columns="hour", values="message",
                              aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    return pivot


def user_day_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """user x day-of-week message-count matrix -- the full 'active days'
    picture per user."""
    data = _real_messages(df)
    if _is_empty(data):
        return pd.DataFrame(columns=_DAY_ORDER)
    pivot = data.pivot_table(index="user", columns="day_name", values="message",
                              aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=_DAY_ORDER, fill_value=0)
    return pivot


# ---------------------------------------------------------------------------
# Activity Analysis -- interactive Plotly figures
# ---------------------------------------------------------------------------

def fig_messages_by_date(df: pd.DataFrame, selected_user: str = "Overall") -> go.Figure:
    """Daily message-volume line chart."""
    data = _filter_user(df, selected_user)
    if _is_empty(data):
        return _empty_figure("No messages to display")

    daily = (
        data.groupby("only_date").size().reset_index(name="messages")
        .sort_values("only_date")
    )
    fig = px.line(daily, x="only_date", y="messages", markers=True,
                  title="Messages by Date")
    fig.update_layout(xaxis_title="Date", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    return fig


def fig_messages_by_month(df: pd.DataFrame, selected_user: str = "Overall") -> go.Figure:
    """Monthly message-volume bar chart (chronologically ordered)."""
    data = _filter_user(df, selected_user)
    if _is_empty(data):
        return _empty_figure("No messages to display")

    monthly = (
        data.groupby(["year", "month_num", "month"]).size()
        .reset_index(name="messages")
        .sort_values(["year", "month_num"])
    )
    monthly["label"] = monthly["month"] + " " + monthly["year"].astype(str)
    fig = px.bar(monthly, x="label", y="messages", title="Messages by Month")
    fig.update_layout(xaxis_title="Month", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    return fig


def fig_messages_by_dow(df: pd.DataFrame, selected_user: str = "Overall") -> go.Figure:
    """Message-volume bar chart by day of week (Monday-Sunday order)."""
    data = _filter_user(df, selected_user)
    if _is_empty(data):
        return _empty_figure("No messages to display")

    counts = (
        data["day_name"].value_counts().reindex(_DAY_ORDER).fillna(0)
        .reset_index()
    )
    counts.columns = ["day", "messages"]
    fig = px.bar(counts, x="day", y="messages", title="Messages by Day of Week")
    fig.update_layout(xaxis_title="Day", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    return fig


def fig_messages_by_hour(df: pd.DataFrame, selected_user: str = "Overall") -> go.Figure:
    """Message-volume bar chart by hour of day (0-23)."""
    data = _filter_user(df, selected_user)
    if _is_empty(data):
        return _empty_figure("No messages to display")

    counts = (
        data["hour"].value_counts().reindex(range(24)).fillna(0)
        .reset_index()
    )
    counts.columns = ["hour", "messages"]
    fig = px.bar(counts, x="hour", y="messages", title="Messages by Hour")
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    fig.update_xaxes(dtick=1)
    return fig


def fig_messages_by_user(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Message-count bar chart across all real users (not filtered to a
    single selected user -- ranking users only makes sense unfiltered)."""
    data = _real_messages(df)
    if _is_empty(data):
        return _empty_figure("No messages to display")

    counts = data["user"].value_counts().head(top_n).reset_index()
    counts.columns = ["user", "messages"]
    fig = px.bar(counts, x="user", y="messages", title="Messages by User")
    fig.update_layout(xaxis_title="User", yaxis_title="Messages",
                       margin=dict(t=50, l=10, r=10, b=10))
    return fig


# ---------------------------------------------------------------------------
# User Ranking
# ---------------------------------------------------------------------------

def rank_by_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Users ranked by total message count, descending."""
    stats = user_stats(df)
    if stats.empty:
        return stats
    return (
        stats[["user", "message_count"]]
        .sort_values("message_count", ascending=False)
        .reset_index(drop=True)
    )


def rank_by_words(df: pd.DataFrame) -> pd.DataFrame:
    """Users ranked by total word count, descending."""
    stats = user_stats(df)
    if stats.empty:
        return stats
    return (
        stats[["user", "word_count"]]
        .sort_values("word_count", ascending=False)
        .reset_index(drop=True)
    )


def rank_by_avg_length(df: pd.DataFrame) -> pd.DataFrame:
    """Users ranked by average message length (words/message), descending."""
    stats = user_stats(df)
    if stats.empty:
        return stats
    return (
        stats[["user", "avg_message_length"]]
        .sort_values("avg_message_length", ascending=False)
        .reset_index(drop=True)
    )


def rank_by_active_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Users ranked by 'most active hours' -- i.e. how many messages they
    send during their single busiest hour of the day. A user who floods
    one specific hour ranks above one whose activity is spread thin,
    which is what "most active hours" means at a per-user level.

    Columns: user, peak_hour, peak_hour_messages.
    """
    dist = user_hour_distribution(df)
    if dist.empty:
        return pd.DataFrame(columns=["user", "peak_hour", "peak_hour_messages"])

    peak_hour = dist.idxmax(axis=1)
    peak_hour_messages = dist.max(axis=1)

    out = pd.DataFrame({
        "user": dist.index,
        "peak_hour": peak_hour.values,
        "peak_hour_messages": peak_hour_messages.values.astype(int),
    })
    return out.sort_values("peak_hour_messages", ascending=False).reset_index(drop=True)
