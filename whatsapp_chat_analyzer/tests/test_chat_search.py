"""Tests for chat_search.py (PHASE 9 -- Chat Search & Filtering)."""

import datetime

import pandas as pd
import pytest

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whatsapp_chat_analyzer.scripts.src import chat_search


@pytest.fixture
def sample_df():
    rows = [
        # date,        user,     message,                          message_type
        ("2023-05-01", "Aditi", "Good morning everyone!",           "text"),
        ("2023-05-01", "Rohan", "Good morning, ready for the exam?", "text"),
        ("2023-05-02", "Meera", "I loved that movie last night",    "text"),
        ("2023-05-02", "Karan", "<Media omitted>",                  "media"),
        ("2023-05-03", "Aditi", "Exam went okay I guess",            "text"),
        ("2023-05-03", "Rohan", "Meera added Karan",                 "system"),
        ("2023-05-04", "Meera", "Ugh, I failed the exam :(",         "text"),
    ]
    df = pd.DataFrame(rows, columns=["date", "user", "message", "message_type"])
    df["only_date"] = pd.to_datetime(df["date"]).dt.date
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------
def test_keyword_search_case_insensitive(sample_df):
    results = chat_search.search_messages(sample_df, keyword="EXAM")
    assert len(results) == 3
    assert set(results["user"]) == {"Rohan", "Aditi", "Meera"}


def test_keyword_search_case_sensitive(sample_df):
    results = chat_search.search_messages(sample_df, keyword="EXAM", case_sensitive=True)
    assert len(results) == 0


def test_keyword_search_regex(sample_df):
    results = chat_search.search_messages(sample_df, keyword=r"^Good", use_regex=True)
    assert len(results) == 2


def test_keyword_search_invalid_regex_raises(sample_df):
    with pytest.raises(Exception):
        chat_search.search_messages(sample_df, keyword="[unclosed", use_regex=True)


def test_blank_keyword_is_noop(sample_df):
    results = chat_search.search_messages(sample_df, keyword="   ")
    assert len(results) == len(sample_df)


# ---------------------------------------------------------------------------
# Sender / user search
# ---------------------------------------------------------------------------
def test_search_by_single_sender(sample_df):
    results = chat_search.search_messages(sample_df, senders=["Meera"])
    assert len(results) == 2
    assert set(results["user"]) == {"Meera"}


def test_search_by_multiple_senders(sample_df):
    results = chat_search.search_messages(sample_df, senders=["Meera", "Karan"])
    assert set(results["user"]) == {"Meera", "Karan"}


def test_no_senders_is_noop(sample_df):
    results = chat_search.search_messages(sample_df, senders=[])
    assert len(results) == len(sample_df)


# ---------------------------------------------------------------------------
# Date / date-range filtering
# ---------------------------------------------------------------------------
def test_search_by_single_date(sample_df):
    d = datetime.date(2023, 5, 2)
    results = chat_search.search_messages(sample_df, start_date=d, end_date=d)
    assert len(results) == 2
    assert set(results["only_date"]) == {d}


def test_search_by_date_range(sample_df):
    results = chat_search.search_messages(
        sample_df,
        start_date=datetime.date(2023, 5, 2),
        end_date=datetime.date(2023, 5, 3),
    )
    assert len(results) == 4


def test_date_range_outside_data_returns_empty(sample_df):
    results = chat_search.search_messages(
        sample_df,
        start_date=datetime.date(2023, 6, 1),
        end_date=datetime.date(2023, 6, 30),
    )
    assert results.empty


# ---------------------------------------------------------------------------
# Sentiment filtering
# ---------------------------------------------------------------------------
def test_sentiment_filter(sample_df):
    # Pretend only the "text" rows (indices 0,1,2,4,6) were classified.
    text_idx = sample_df[sample_df["message_type"] == "text"].index
    labels = ["positive", "neutral", "positive", "negative", "negative"]
    sentiment = chat_search.build_sentiment_series(text_idx, labels)

    results = chat_search.search_messages(
        sample_df, sentiment=sentiment, sentiment_filter=["negative"]
    )
    assert len(results) == 2

    # Rows never classified (media/system) never match a sentiment filter.
    assert "media" not in results["message_type"].values
    assert "system" not in results["message_type"].values


def test_sentiment_filter_none_is_noop(sample_df):
    results = chat_search.search_messages(sample_df, sentiment=None, sentiment_filter=["positive"])
    assert len(results) == len(sample_df)


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------
def test_combined_keyword_sender_and_date(sample_df):
    results = chat_search.search_messages(
        sample_df,
        keyword="exam",
        senders=["Aditi"],
        start_date=datetime.date(2023, 5, 3),
        end_date=datetime.date(2023, 5, 3),
    )
    assert len(results) == 1
    assert results.iloc[0]["message"] == "Exam went okay I guess"


def test_combined_filters_narrow_to_empty(sample_df):
    results = chat_search.search_messages(
        sample_df, keyword="exam", senders=["Karan"]  # Karan never mentions "exam"
    )
    assert results.empty


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------
def test_no_matches_returns_empty_dataframe(sample_df):
    results = chat_search.search_messages(sample_df, keyword="xyzzyunfindable")
    assert isinstance(results, pd.DataFrame)
    assert results.empty


def test_empty_input_dataframe():
    empty = pd.DataFrame(columns=["date", "user", "message", "message_type", "only_date"])
    results = chat_search.search_messages(empty, keyword="anything")
    assert results.empty


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
def test_pagination_basic(sample_df):
    page_df, total_pages, total_results = chat_search.paginate(sample_df, page=1, page_size=3)
    assert total_results == 7
    assert total_pages == 3
    assert len(page_df) == 3


def test_pagination_last_page_partial(sample_df):
    page_df, total_pages, total_results = chat_search.paginate(sample_df, page=3, page_size=3)
    assert total_pages == 3
    assert len(page_df) == 1  # 7 rows, page size 3 -> last page has 1


def test_pagination_page_clamped_when_out_of_range(sample_df):
    page_df, total_pages, _ = chat_search.paginate(sample_df, page=99, page_size=3)
    assert total_pages == 3
    assert len(page_df) == 1  # clamped to last page


def test_pagination_page_clamped_below_one(sample_df):
    page_df, total_pages, total_results = chat_search.paginate(sample_df, page=-5, page_size=3)
    assert len(page_df) == 3  # clamped to page 1


def test_pagination_on_empty_dataframe(sample_df):
    empty = sample_df.iloc[0:0]
    page_df, total_pages, total_results = chat_search.paginate(empty, page=1, page_size=50)
    assert total_results == 0
    assert total_pages == 1
    assert page_df.empty


def test_pagination_large_chat_never_materializes_more_than_a_page():
    # Simulate a "huge chat" -- 50,000 messages -- and make sure a page
    # request only ever returns page_size rows, regardless of how many
    # total rows exist.
    n = 50_000
    big_df = pd.DataFrame({
        "date": ["2023-01-01"] * n,
        "user": ["User"] * n,
        "message": [f"message {i}" for i in range(n)],
        "message_type": ["text"] * n,
        "only_date": [datetime.date(2023, 1, 1)] * n,
    })
    page_df, total_pages, total_results = chat_search.paginate(big_df, page=5, page_size=50)
    assert total_results == n
    assert len(page_df) == 50
    assert total_pages == n // 50
