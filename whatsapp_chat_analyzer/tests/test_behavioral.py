"""
tests/test_behavioral.py
--------------------------
Unit tests for behavioral.py (Phase 3: Advanced Behavioral Analytics).

We build DataFrames directly (matching the schema parser.preprocess()
produces) rather than only going through raw .txt text, so that response
timing, night-hour boundaries, session gaps, and streak boundaries can be
controlled to the second/day for precise assertions. An end-to-end test
against sample_chat.txt (via the real parser) is included at the bottom
to confirm the integration path also works.

Run with:
    pytest tests/test_behavioral.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import pytest

from whatsapp_chat_analyzer.scripts.src import behavioral
from whatsapp_chat_analyzer.scripts.src import parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(entries):
    """entries: list of (datetime_str 'YYYY-MM-DD HH:MM:SS', user, message)
    Builds a DataFrame with the same schema parser.preprocess()
    produces, so behavioral.py functions can be tested with exact,
    hand-picked timestamps."""
    rows = []
    for date_str, user, message in entries:
        d = pd.to_datetime(date_str)
        rows.append({
            "date": d,
            "datetime": d,
            "only_date": d.date(),
            "time": d.strftime("%H:%M"),
            "year": d.year,
            "month_num": d.month,
            "month": d.month_name(),
            "day": d.day,
            "day_name": d.day_name(),
            "hour": d.hour,
            "minute": d.minute,
            "user": user,
            "sender": user,
            "message": message,
            "message_type": "text",
        })
    return pd.DataFrame(rows)


def _empty_schema_df():
    return parser.preprocess("")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def latency_df():
    """
    Aditi -> Rohan reply after 60s
    Rohan -> Aditi reply after 120s
    Aditi -> Aditi (same user, consecutive - must be ignored)
    Aditi -> Rohan reply after 300s (5 min)
    (large gap, 5 hours later) Rohan -> Aditi: excluded by default 180min window
    """
    return _make_df([
        ("2023-05-12 09:00:00", "Aditi", "hey"),
        ("2023-05-12 09:01:00", "Rohan", "hi"),                # +60s reply
        ("2023-05-12 09:03:00", "Aditi", "how are you"),       # +120s reply
        ("2023-05-12 09:03:30", "Aditi", "u there?"),           # same user, ignored
        ("2023-05-12 09:08:30", "Rohan", "yes all good"),      # +300s reply (from 09:03:30)
        ("2023-05-12 14:08:30", "Aditi", "ok talk later"),     # 5h gap -> excluded (default 180min)
    ])


@pytest.fixture
def night_df():
    """3 night messages (hours 0, 2, 4) + 2 day messages (hour 10, 14)."""
    return _make_df([
        ("2023-05-12 00:30:00", "Aditi", "can't sleep"),
        ("2023-05-12 02:15:00", "Rohan", "me neither"),
        ("2023-05-12 04:59:00", "Aditi", "almost dawn"),
        ("2023-05-12 05:00:00", "Rohan", "5am exactly - not night"),  # boundary: NOT night
        ("2023-05-12 10:00:00", "Aditi", "morning"),
        ("2023-05-12 14:00:00", "Rohan", "afternoon"),
    ])


@pytest.fixture
def conversation_df():
    """Two sessions using a 60-minute default gap:
    Session 1: Aditi starts, Rohan replies within the session.
    (2 hour gap)
    Session 2: Rohan starts, Aditi replies.
    """
    return _make_df([
        ("2023-05-12 09:00:00", "Aditi", "session1 start"),
        ("2023-05-12 09:05:00", "Rohan", "session1 reply"),
        ("2023-05-12 11:30:00", "Rohan", "session2 start (2h15m gap)"),
        ("2023-05-12 11:32:00", "Aditi", "session2 reply"),
    ])


@pytest.fixture
def streak_df():
    """Aditi active on 3 consecutive days (May 10-12), then a gap, then May 15.
    Rohan only active on May 11."""
    return _make_df([
        ("2023-05-10 09:00:00", "Aditi", "day1"),
        ("2023-05-11 09:00:00", "Aditi", "day2"),
        ("2023-05-11 10:00:00", "Rohan", "rohan day"),
        ("2023-05-12 09:00:00", "Aditi", "day3"),
        ("2023-05-15 09:00:00", "Aditi", "day6, after gap"),
    ])


# ---------------------------------------------------------------------------
# 1. Response Latency
# ---------------------------------------------------------------------------

def test_compute_response_events_ignores_same_user_and_large_gap(latency_df):
    events = behavioral.compute_response_events(latency_df, max_response_minutes=180)
    # Expect 3 events: 60s, 120s, 300s. The same-user pair and the 5h gap
    # pair must be excluded.
    assert len(events) == 3
    assert set(events["response_seconds"]) == {60.0, 120.0, 300.0}


def test_compute_response_events_configurable_window(latency_df):
    # With a tiny window, only the 60s reply survives
    events = behavioral.compute_response_events(latency_df, max_response_minutes=1)
    assert len(events) == 1
    assert events.iloc[0]["response_seconds"] == 60.0

    # With a huge window, the 5-hour gap reply is now included too
    events_wide = behavioral.compute_response_events(latency_df, max_response_minutes=1000)
    assert len(events_wide) == 4


def test_response_latency_stats_values(latency_df):
    stats = behavioral.response_latency_stats(latency_df, max_response_minutes=180)
    assert stats["sample_size"] == 3
    assert stats["fastest_response_seconds"] == 60.0
    assert stats["slowest_response_seconds"] == 300.0
    assert stats["average_response_seconds"] == pytest.approx((60 + 120 + 300) / 3, rel=1e-3)
    assert stats["median_response_seconds"] == 120.0
    assert stats["max_response_window_minutes"] == 180


def test_response_latency_stats_empty():
    stats = behavioral.response_latency_stats(_empty_schema_df())
    assert stats["sample_size"] == 0
    assert stats["average_response_seconds"] is None
    assert stats["fastest_response_seconds"] is None


def test_response_latency_by_user(latency_df):
    by_user = behavioral.response_latency_by_user(latency_df, max_response_minutes=180)
    # Rohan replied twice (60s, 300s), Aditi replied once (120s)
    rohan = by_user[by_user["user"] == "Rohan"].iloc[0]
    aditi = by_user[by_user["user"] == "Aditi"].iloc[0]
    assert rohan["replies_count"] == 2
    assert aditi["replies_count"] == 1
    assert aditi["avg_response_minutes"] == round(120 / 60, 2)


def test_response_latency_between_users_min_samples_filter(latency_df):
    # Only 1 sample per direction here, default min_samples=3 -> nothing meaningful
    between = behavioral.response_latency_between_users(latency_df, max_response_minutes=180)
    assert between.empty

    # Lower the threshold to 1 -> now they should show up
    between_relaxed = behavioral.response_latency_between_users(
        latency_df, max_response_minutes=180, min_samples=1
    )
    assert not between_relaxed.empty
    assert set(between_relaxed.columns) == {
        "original_sender", "replier", "replies_count",
        "avg_response_minutes", "median_response_minutes",
    }


def test_response_latency_no_crossuser_replies_single_user():
    df = _make_df([
        ("2023-05-12 09:00:00", "Aditi", "note 1"),
        ("2023-05-12 09:01:00", "Aditi", "note 2"),
    ])
    events = behavioral.compute_response_events(df)
    assert events.empty
    stats = behavioral.response_latency_stats(df)
    assert stats["sample_size"] == 0


# ---------------------------------------------------------------------------
# 2. Night Owl Analysis
# ---------------------------------------------------------------------------

def test_night_owl_stats(night_df):
    stats = behavioral.night_owl_stats(night_df)
    # hours 0, 2, 4 are night (3 messages); 5:00 and later are NOT night
    assert stats["total_night_messages"] == 3
    assert stats["total_messages"] == 6
    assert stats["pct_of_total"] == round(3 / 6 * 100, 2)


def test_night_owl_boundary_5am_excluded(night_df):
    # The 05:00:00 message must NOT be counted as night
    night_hours_present = night_df[night_df["hour"].isin(behavioral.NIGHT_HOURS)]["hour"].tolist()
    assert 5 not in night_hours_present
    assert set(night_hours_present) == {0, 2, 4}


def test_night_owl_by_user(night_df):
    by_user = behavioral.night_owl_by_user(night_df)
    aditi = by_user[by_user["user"] == "Aditi"].iloc[0]
    rohan = by_user[by_user["user"] == "Rohan"].iloc[0]
    # Aditi: night msgs at 00:30 and 04:59 -> 2 night messages out of 3 total
    assert aditi["night_messages"] == 2
    assert aditi["total_messages"] == 3
    # Rohan: night msg at 02:15 only (05:00 excluded) -> 1 out of 3
    assert rohan["night_messages"] == 1
    assert rohan["total_messages"] == 3


def test_night_activity_by_hour(night_df):
    counts = behavioral.night_activity_by_hour(night_df)
    assert list(counts["hour"]) == [0, 1, 2, 3, 4]
    lookup = dict(zip(counts["hour"], counts["messages"]))
    assert lookup[0] == 1
    assert lookup[2] == 1
    assert lookup[4] == 1
    assert lookup[1] == 0
    assert lookup[3] == 0


def test_night_owl_empty_chat():
    empty = _empty_schema_df()
    stats = behavioral.night_owl_stats(empty)
    assert stats == {"total_night_messages": 0, "total_messages": 0, "pct_of_total": 0.0}
    assert behavioral.night_owl_by_user(empty).empty


def test_fig_night_activity_returns_figure(night_df):
    fig = behavioral.fig_night_activity_by_hour(night_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_fig_night_activity_placeholder_when_no_night_messages():
    df = _make_df([("2023-05-12 10:00:00", "Aditi", "daytime only")])
    fig = behavioral.fig_night_activity_by_hour(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 1


# ---------------------------------------------------------------------------
# 3. Conversation Starters
# ---------------------------------------------------------------------------

def test_conversation_starters_methodology(conversation_df):
    starters = behavioral.conversation_starters(conversation_df, session_gap_minutes=60)
    # Session 1 started by Aditi (first message ever), session 2 started
    # by Rohan (after a 2h15m > 60min gap).
    total_sessions = behavioral.count_conversation_sessions(conversation_df, session_gap_minutes=60)
    assert total_sessions == 2

    lookup = dict(zip(starters["user"], starters["conversations_started"]))
    assert lookup["Aditi"] == 1
    assert lookup["Rohan"] == 1
    assert set(starters["pct_of_conversations"]) == {50.0}


def test_conversation_starters_configurable_gap(conversation_df):
    # With a huge session gap, the whole chat is one session -> only Aditi
    # (first message) is credited as the starter.
    starters = behavioral.conversation_starters(conversation_df, session_gap_minutes=100000)
    assert len(starters) == 1
    assert starters.iloc[0]["user"] == "Aditi"
    assert starters.iloc[0]["conversations_started"] == 1


def test_conversation_starters_empty_chat():
    starters = behavioral.conversation_starters(_empty_schema_df())
    assert starters.empty
    assert behavioral.count_conversation_sessions(_empty_schema_df()) == 0


def test_conversation_starters_single_user():
    df = _make_df([
        ("2023-05-12 09:00:00", "Aditi", "msg1"),
        ("2023-05-12 09:05:00", "Aditi", "msg2"),
    ])
    starters = behavioral.conversation_starters(df, session_gap_minutes=60)
    assert len(starters) == 1
    assert starters.iloc[0]["user"] == "Aditi"
    assert starters.iloc[0]["pct_of_conversations"] == 100.0


# ---------------------------------------------------------------------------
# 4. Activity Streaks
# ---------------------------------------------------------------------------

def test_chat_activity_streaks(streak_df):
    streaks = behavioral.chat_activity_streaks(streak_df)
    # Active days: May 10, 11, 12 (consecutive = 3), then May 15 (isolated)
    assert streaks["longest_streak_days"] == 3
    assert streaks["longest_streak_start"] == dt.date(2023, 5, 10)
    assert streaks["longest_streak_end"] == dt.date(2023, 5, 12)
    assert streaks["total_active_days"] == 4
    # current streak = run ending at the LAST active day (May 15) = 1 day
    assert streaks["current_streak_days"] == 1


def test_user_activity_streaks(streak_df):
    streaks = behavioral.user_activity_streaks(streak_df)
    aditi = streaks[streaks["user"] == "Aditi"].iloc[0]
    rohan = streaks[streaks["user"] == "Rohan"].iloc[0]
    assert aditi["longest_streak_days"] == 3  # May 10-12
    assert aditi["total_active_days"] == 4    # 10, 11, 12, 15
    assert rohan["longest_streak_days"] == 1  # only May 11
    assert rohan["total_active_days"] == 1


def test_streaks_empty_chat():
    streaks = behavioral.chat_activity_streaks(_empty_schema_df())
    assert streaks["longest_streak_days"] == 0
    assert streaks["total_active_days"] == 0
    assert behavioral.user_activity_streaks(_empty_schema_df()).empty


def test_streaks_single_day():
    df = _make_df([("2023-05-12 09:00:00", "Aditi", "only message")])
    streaks = behavioral.chat_activity_streaks(df)
    assert streaks["longest_streak_days"] == 1
    assert streaks["current_streak_days"] == 1
    assert streaks["total_active_days"] == 1


# ---------------------------------------------------------------------------
# 5. User Interaction
# ---------------------------------------------------------------------------

def test_interaction_matrix(latency_df):
    matrix = behavioral.interaction_matrix(latency_df, max_response_minutes=180)
    # Rohan replied to Aditi twice; Aditi replied to Rohan once
    assert matrix.loc["Rohan", "Aditi"] == 2
    assert matrix.loc["Aditi", "Rohan"] == 1


def test_top_interaction_pairs(latency_df):
    pairs = behavioral.top_interaction_pairs(latency_df, max_response_minutes=180)
    top = pairs.iloc[0]
    assert top["original_sender"] == "Aditi"
    assert top["replier"] == "Rohan"
    assert top["reply_count"] == 2


def test_interaction_matrix_empty():
    matrix = behavioral.interaction_matrix(_empty_schema_df())
    assert matrix.empty


def test_fig_interaction_heatmap(latency_df):
    fig = behavioral.fig_interaction_heatmap(latency_df, max_response_minutes=180)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_fig_interaction_heatmap_empty_placeholder():
    fig = behavioral.fig_interaction_heatmap(_empty_schema_df())
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 1


def test_interaction_single_user_no_replies():
    df = _make_df([
        ("2023-05-12 09:00:00", "Aditi", "msg1"),
        ("2023-05-12 09:05:00", "Aditi", "msg2"),
    ])
    matrix = behavioral.interaction_matrix(df)
    assert matrix.empty


# ---------------------------------------------------------------------------
# Integration: real parser + sample_chat.txt
# ---------------------------------------------------------------------------

def test_sample_chat_full_behavioral_pipeline():
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_chat.txt"
    )
    with open(sample_path, encoding="utf-8") as f:
        data = f.read()
    df = parser.preprocess(data)
    assert not df.empty

    stats = behavioral.response_latency_stats(df)
    assert isinstance(stats, dict)

    by_user = behavioral.response_latency_by_user(df)
    assert isinstance(by_user, pd.DataFrame)

    night_stats = behavioral.night_owl_stats(df)
    assert night_stats["total_messages"] > 0

    starters = behavioral.conversation_starters(df)
    assert isinstance(starters, pd.DataFrame)

    streaks = behavioral.chat_activity_streaks(df)
    assert streaks["total_active_days"] >= 1

    matrix = behavioral.interaction_matrix(df)
    assert isinstance(matrix, pd.DataFrame)

    for fn_name in ["fig_night_activity_by_hour", "fig_night_owl_by_user"]:
        fig = getattr(behavioral, fn_name)(df)
        assert isinstance(fig, go.Figure)

    heatmap = behavioral.fig_interaction_heatmap(df)
    assert isinstance(heatmap, go.Figure)
