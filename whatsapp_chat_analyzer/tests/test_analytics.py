"""
tests/test_analytics.py
------------------------
Unit tests for analytics.py (Phase 2: Chat & User Analytics).

Run with:
    pytest tests/test_analytics.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import pytest

from whatsapp_chat_analyzer.scripts.src import analytics
from whatsapp_chat_analyzer.scripts.src import parser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MULTI_USER_CHAT = (
    "12/05/23, 9:15 pm - Messages and calls are end-to-end encrypted.\n"
    "12/05/23, 9:15 pm - Aditi: Hey everyone! How's it going?\n"
    "12/05/23, 9:16 pm - Rohan: All good yaar, just finished my exam\n"
    "12/05/23, 9:16 pm - Rohan: Feeling super relieved\n"
    "12/05/23, 9:17 pm - Meera: Congrats Rohan!! that's amazing news\n"
    "12/05/23, 9:18 pm - Karan: Nice one bro, party time?\n"
    "12/05/23, 9:20 pm - Aditi: <Media omitted>\n"
    "12/05/23, 9:21 pm - Rohan: haha thanks guys, feeling great\n"
    "13/05/23, 8:05 am - Rohan: morning! check this out https://example.com/article\n"
    "13/05/23, 9:00 am - Karan: anyone free for lunch today?\n"
    "13/05/23, 9:02 am - Rohan: I'm in!\n"
    "14/06/23, 11:30 am - Aditi: a message from a different month entirely\n"
)

ONE_TO_ONE_CHAT = (
    "12/05/23, 9:15 pm - Aditi: hi there\n"
    "12/05/23, 9:16 pm - Rohan: hey! good to hear from you\n"
    "12/05/23, 9:17 pm - Aditi: how have you been doing lately\n"
)

SINGLE_USER_CHAT = (
    "12/05/23, 9:15 pm - Aditi: message one\n"
    "12/05/23, 10:15 pm - Aditi: message two here\n"
    "13/05/23, 11:15 am - Aditi: message three is a bit longer than the rest\n"
)


@pytest.fixture
def multi_user_df():
    return parser.preprocess(MULTI_USER_CHAT)


@pytest.fixture
def one_to_one_df():
    return parser.preprocess(ONE_TO_ONE_CHAT)


@pytest.fixture
def single_user_df():
    return parser.preprocess(SINGLE_USER_CHAT)


@pytest.fixture
def empty_df():
    return parser.preprocess("")


# ---------------------------------------------------------------------------
# overall_stats
# ---------------------------------------------------------------------------

def test_overall_stats_real_numbers(multi_user_df):
    stats = analytics.overall_stats(multi_user_df)
    # 11 real messages (system notification excluded)
    assert stats["total_messages"] == 11
    assert stats["total_users"] == 4
    assert stats["media_messages"] == 1
    assert stats["links"] == 1
    assert stats["most_active_user"] == "Rohan"  # Rohan sent 5 messages
    assert stats["total_words"] > 0
    assert stats["avg_message_length"] == round(stats["total_words"] / stats["total_messages"], 2)


def test_overall_stats_not_hardcoded_changes_with_data(multi_user_df, one_to_one_df):
    stats_multi = analytics.overall_stats(multi_user_df)
    stats_pair = analytics.overall_stats(one_to_one_df)
    assert stats_multi["total_messages"] != stats_pair["total_messages"]
    assert stats_multi["total_users"] != stats_pair["total_users"]


def test_overall_stats_empty_chat(empty_df):
    stats = analytics.overall_stats(empty_df)
    assert stats == {
        "total_messages": 0,
        "total_users": 0,
        "total_words": 0,
        "avg_message_length": 0.0,
        "media_messages": 0,
        "links": 0,
        "most_active_user": None,
    }


def test_overall_stats_single_user_chat(single_user_df):
    stats = analytics.overall_stats(single_user_df)
    assert stats["total_messages"] == 3
    assert stats["total_users"] == 1
    assert stats["most_active_user"] == "Aditi"


# ---------------------------------------------------------------------------
# user_stats
# ---------------------------------------------------------------------------

def test_user_stats_columns_and_totals(multi_user_df):
    stats = analytics.user_stats(multi_user_df)
    assert list(stats.columns) == [
        "user", "message_count", "word_count", "avg_message_length",
        "pct_of_total", "most_active_hour", "most_active_day",
    ]
    assert stats["message_count"].sum() == 11
    # percentages should sum to (approximately) 100
    assert abs(stats["pct_of_total"].sum() - 100.0) < 0.1


def test_user_stats_rohan_details(multi_user_df):
    stats = analytics.user_stats(multi_user_df)
    rohan = stats[stats["user"] == "Rohan"].iloc[0]
    assert rohan["message_count"] == 5
    assert rohan["word_count"] > 0
    assert rohan["avg_message_length"] == round(rohan["word_count"] / rohan["message_count"], 2)


def test_user_stats_empty_chat(empty_df):
    stats = analytics.user_stats(empty_df)
    assert stats.empty
    assert list(stats.columns) == [
        "user", "message_count", "word_count", "avg_message_length",
        "pct_of_total", "most_active_hour", "most_active_day",
    ]


def test_user_stats_single_user_chat(single_user_df):
    stats = analytics.user_stats(single_user_df)
    assert len(stats) == 1
    assert stats.iloc[0]["user"] == "Aditi"
    assert stats.iloc[0]["message_count"] == 3
    assert stats.iloc[0]["pct_of_total"] == 100.0


# ---------------------------------------------------------------------------
# user_hour_distribution / user_day_distribution
# ---------------------------------------------------------------------------

def test_user_hour_distribution_shape(multi_user_df):
    dist = analytics.user_hour_distribution(multi_user_df)
    assert list(dist.columns) == list(range(24))
    # total across the matrix should equal total real messages
    assert dist.values.sum() == 11


def test_user_day_distribution_shape(multi_user_df):
    dist = analytics.user_day_distribution(multi_user_df)
    assert list(dist.columns) == analytics._DAY_ORDER
    assert dist.values.sum() == 11


def test_distributions_empty_chat(empty_df):
    assert analytics.user_hour_distribution(empty_df).empty
    assert analytics.user_day_distribution(empty_df).empty


# ---------------------------------------------------------------------------
# Plotly activity figures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn_name", [
    "fig_messages_by_date", "fig_messages_by_month",
    "fig_messages_by_dow", "fig_messages_by_hour",
])
def test_activity_figures_return_plotly_figure(multi_user_df, fn_name):
    fn = getattr(analytics, fn_name)
    fig = fn(multi_user_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # has actual traces, not just an empty placeholder


@pytest.mark.parametrize("fn_name", [
    "fig_messages_by_date", "fig_messages_by_month",
    "fig_messages_by_dow", "fig_messages_by_hour",
])
def test_activity_figures_empty_chat_returns_placeholder(empty_df, fn_name):
    fn = getattr(analytics, fn_name)
    fig = fn(empty_df)
    assert isinstance(fig, go.Figure)
    # placeholder figure has an annotation, no bar/line trace
    assert len(fig.layout.annotations) == 1


def test_fig_messages_by_hour_respects_selected_user(multi_user_df):
    fig_overall = analytics.fig_messages_by_hour(multi_user_df, "Overall")
    fig_rohan = analytics.fig_messages_by_hour(multi_user_df, "Rohan")
    total_overall = sum(fig_overall.data[0].y)
    total_rohan = sum(fig_rohan.data[0].y)
    assert total_rohan < total_overall
    assert total_rohan == 5  # Rohan sent 5 messages


def test_fig_messages_by_user_ranks_all_users(multi_user_df):
    fig = analytics.fig_messages_by_user(multi_user_df)
    assert isinstance(fig, go.Figure)
    users = list(fig.data[0].x)
    assert set(users) == {"Aditi", "Rohan", "Meera", "Karan"}


def test_fig_messages_by_user_empty_chat(empty_df):
    fig = analytics.fig_messages_by_user(empty_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 1


def test_fig_messages_by_user_single_user(single_user_df):
    fig = analytics.fig_messages_by_user(single_user_df)
    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].x) == ["Aditi"]
    assert list(fig.data[0].y) == [3]


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def test_rank_by_messages_order(multi_user_df):
    ranking = analytics.rank_by_messages(multi_user_df)
    assert list(ranking["user"]) == list(ranking.sort_values("message_count", ascending=False)["user"])
    assert ranking.iloc[0]["user"] == "Rohan"
    assert ranking.iloc[0]["message_count"] == 5


def test_rank_by_words_order(multi_user_df):
    ranking = analytics.rank_by_words(multi_user_df)
    assert ranking["word_count"].is_monotonic_decreasing


def test_rank_by_avg_length_order(multi_user_df):
    ranking = analytics.rank_by_avg_length(multi_user_df)
    assert ranking["avg_message_length"].is_monotonic_decreasing


def test_rank_by_active_hours(multi_user_df):
    ranking = analytics.rank_by_active_hours(multi_user_df)
    assert set(ranking.columns) == {"user", "peak_hour", "peak_hour_messages"}
    assert ranking["peak_hour_messages"].is_monotonic_decreasing
    # every peak hour must be a valid hour of day
    assert ranking["peak_hour"].between(0, 23).all()


def test_rankings_empty_chat_do_not_raise(empty_df):
    assert analytics.rank_by_messages(empty_df).empty
    assert analytics.rank_by_words(empty_df).empty
    assert analytics.rank_by_avg_length(empty_df).empty
    assert analytics.rank_by_active_hours(empty_df).empty


def test_rankings_single_user_chat(single_user_df):
    assert len(analytics.rank_by_messages(single_user_df)) == 1
    assert len(analytics.rank_by_active_hours(single_user_df)) == 1


# ---------------------------------------------------------------------------
# Integration with the bundled sample_chat.txt
# ---------------------------------------------------------------------------

def test_sample_chat_full_analytics_pipeline():
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_chat.txt"
    )
    with open(sample_path, encoding="utf-8") as f:
        data = f.read()
    df = parser.preprocess(data)

    stats = analytics.overall_stats(df)
    assert stats["total_messages"] > 0
    assert stats["most_active_user"] is not None

    users = analytics.user_stats(df)
    assert not users.empty

    for fn_name in ["fig_messages_by_date", "fig_messages_by_month",
                     "fig_messages_by_dow", "fig_messages_by_hour",
                     "fig_messages_by_user"]:
        fig = getattr(analytics, fn_name)(df)
        assert isinstance(fig, go.Figure)

    for fn_name in ["rank_by_messages", "rank_by_words",
                     "rank_by_avg_length", "rank_by_active_hours"]:
        ranking = getattr(analytics, fn_name)(df)
        assert not ranking.empty
