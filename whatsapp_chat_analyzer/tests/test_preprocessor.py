"""
tests/test_preprocessor.py
---------------------------
Unit tests for preprocessor.py covering the formats and edge cases
required for Phase 1 (robust WhatsApp parser).

Run with:
    pytest tests/test_preprocessor.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from whatsapp_chat_analyzer.scripts.src import parser as preprocessor


# ---------------------------------------------------------------------------
# Basic format coverage
# ---------------------------------------------------------------------------

def test_android_12h_2digit_year():
    text = "12/05/23, 9:15 pm - Aditi: Hey everyone!"
    df = preprocessor.preprocess(text)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["user"] == "Aditi"
    assert row["message"] == "Hey everyone!"
    assert row["message_type"] == "text"
    assert row["date"].year == 2023
    assert row["date"].month == 5
    assert row["date"].day == 12
    assert row["hour"] == 21  # 9pm -> 21:00


def test_android_24h_4digit_year():
    text = "12/05/2023, 21:15 - Aditi: Hey everyone!"
    df = preprocessor.preprocess(text)
    assert len(df) == 1
    assert df.iloc[0]["hour"] == 21
    assert df.iloc[0]["year"] == 2023


def test_ios_bracket_format_with_seconds():
    text = "[12/05/23, 9:15:00 PM] Aditi: Hey everyone!"
    df = preprocessor.preprocess(text)
    assert len(df) == 1
    assert df.iloc[0]["user"] == "Aditi"
    assert df.iloc[0]["message"] == "Hey everyone!"
    assert df.iloc[0]["hour"] == 21


def test_dot_separated_date():
    text = "12.05.2023, 21:15 - Aditi: dot separated date"
    df = preprocessor.preprocess(text)
    assert len(df) == 1
    assert df.iloc[0]["user"] == "Aditi"


def test_en_dash_separator():
    text = "12/05/23, 9:15 pm \u2013 Aditi: en dash instead of hyphen"
    df = preprocessor.preprocess(text)
    assert len(df) == 1
    assert df.iloc[0]["user"] == "Aditi"


def test_narrow_no_break_space_ios():
    # iOS sometimes uses a narrow no-break space before AM/PM
    text = "12/05/23, 9:15\u202fPM - Aditi: narrow space before PM"
    df = preprocessor.preprocess(text)
    assert len(df) == 1
    assert df.iloc[0]["hour"] == 21


# ---------------------------------------------------------------------------
# Multi-line messages
# ---------------------------------------------------------------------------

def test_multiline_message_is_grouped_into_one_row():
    text = (
        "12/05/23, 9:15 pm - Aditi: Hey everyone\n"
        "this is a second line\n"
        "and a third line\n"
        "12/05/23, 9:16 pm - Rohan: ok cool"
    )
    df = preprocessor.preprocess(text)
    assert len(df) == 2
    assert df.iloc[0]["user"] == "Aditi"
    assert df.iloc[0]["message"] == "Hey everyone\nthis is a second line\nand a third line"
    assert df.iloc[1]["user"] == "Rohan"
    assert df.iloc[1]["message"] == "ok cool"


# ---------------------------------------------------------------------------
# System / group-notification messages
# ---------------------------------------------------------------------------

def test_system_message_has_no_sender():
    text = (
        "12/05/23, 9:15 pm - Messages and calls are end-to-end encrypted.\n"
        "12/05/23, 9:16 pm - Aditi added Rohan\n"
        "12/05/23, 9:17 pm - Rohan: hi"
    )
    df = preprocessor.preprocess(text)
    assert len(df) == 3
    assert (df["user"] == "group_notification").sum() == 2
    assert (df["message_type"] == "system").sum() == 2
    assert df.iloc[2]["user"] == "Rohan"
    assert df.iloc[2]["message_type"] == "text"


# ---------------------------------------------------------------------------
# Media / deleted classification
# ---------------------------------------------------------------------------

def test_media_omitted_legacy_bracket_style():
    text = "12/05/23, 9:15 pm - Aditi: <Media omitted>"
    df = preprocessor.preprocess(text)
    assert df.iloc[0]["message_type"] == "media"


@pytest.mark.parametrize("phrase", [
    "image omitted", "video omitted", "audio omitted",
    "sticker omitted", "GIF omitted", "document omitted",
])
def test_media_omitted_modern_style(phrase):
    text = f"12/05/23, 9:15 pm - Aditi: {phrase}"
    df = preprocessor.preprocess(text)
    assert df.iloc[0]["message_type"] == "media"


def test_deleted_message_classification():
    text = (
        "12/05/23, 9:15 pm - Aditi: This message was deleted\n"
        "12/05/23, 9:16 pm - Rohan: You deleted this message"
    )
    df = preprocessor.preprocess(text)
    assert (df["message_type"] == "deleted").all()


# ---------------------------------------------------------------------------
# Unicode / emoji / URL content preservation
# ---------------------------------------------------------------------------

def test_emoji_and_unicode_preserved():
    text = "12/05/23, 9:15 pm - Aditi: Hey \U0001F600 \u0928\u092e\u0938\u094d\u0924\u0947"
    df = preprocessor.preprocess(text)
    assert "\U0001F600" in df.iloc[0]["message"]
    assert "\u0928\u092e\u0938\u094d\u0924\u0947" in df.iloc[0]["message"]


def test_url_preserved():
    text = "13/05/23, 8:05 am - Rohan: check this out https://example.com/article"
    df = preprocessor.preprocess(text)
    assert "https://example.com/article" in df.iloc[0]["message"]


# ---------------------------------------------------------------------------
# Group chats vs one-to-one chats
# ---------------------------------------------------------------------------

def test_group_chat_multiple_senders():
    text = (
        "12/05/23, 9:15 pm - Aditi: hi\n"
        "12/05/23, 9:16 pm - Rohan: hi\n"
        "12/05/23, 9:17 pm - Meera: hi\n"
        "12/05/23, 9:18 pm - Karan: hi"
    )
    df = preprocessor.preprocess(text)
    senders = set(df["user"]) - {"group_notification"}
    assert senders == {"Aditi", "Rohan", "Meera", "Karan"}


def test_one_to_one_chat_two_senders():
    text = (
        "12/05/23, 9:15 pm - Aditi: hi\n"
        "12/05/23, 9:16 pm - Rohan: hey there"
    )
    df = preprocessor.preprocess(text)
    senders = set(df["user"])
    assert senders == {"Aditi", "Rohan"}


# ---------------------------------------------------------------------------
# Robustness: empty / malformed input must never raise
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty_dataframe():
    df = preprocessor.preprocess("")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == preprocessor.EMPTY_COLUMNS


def test_none_input_returns_empty_dataframe():
    df = preprocessor.preprocess(None)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_whitespace_only_returns_empty_dataframe():
    df = preprocessor.preprocess("   \n\n   \t  \n")
    assert df.empty


def test_pure_garbage_does_not_raise():
    text = "asdkjaslkdj\n#### not a chat export ####\n1234 random text"
    df = preprocessor.preprocess(text)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_garbage_preamble_before_valid_messages_is_skipped():
    text = (
        "corrupted header junk before real export starts\n"
        "12/05/23, 9:15 pm - Aditi: real message"
    )
    df = preprocessor.preprocess(text)
    assert len(df) == 1
    assert df.iloc[0]["message"] == "real message"


# ---------------------------------------------------------------------------
# Schema / derived columns
# ---------------------------------------------------------------------------

def test_schema_has_all_expected_columns():
    text = "12/05/23, 9:15 pm - Aditi: hi"
    df = preprocessor.preprocess(text)
    for col in preprocessor.EMPTY_COLUMNS:
        assert col in df.columns


def test_datetime_and_date_are_aligned_aliases():
    text = "12/05/23, 9:15 pm - Aditi: hi"
    df = preprocessor.preprocess(text)
    assert (df["date"] == df["datetime"]).all()


def test_sender_is_alias_of_user():
    text = "12/05/23, 9:15 pm - Aditi: hi"
    df = preprocessor.preprocess(text)
    assert (df["sender"] == df["user"]).all()


def test_derived_time_fields():
    text = "12/05/23, 9:15 pm - Aditi: hi"
    df = preprocessor.preprocess(text)
    row = df.iloc[0]
    assert row["day"] == 12
    assert row["month"] == "May"
    assert row["month_num"] == 5
    assert row["day_name"] == "Friday"
    assert row["time"] == "21:15"


# ---------------------------------------------------------------------------
# classify_message_type unit-level tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg,expected", [
    ("", "text"),
    ("hello there", "text"),
    ("<Media omitted>", "media"),
    ("image omitted", "media"),
    ("This message was deleted", "deleted"),
    ("You deleted this message", "deleted"),
])
def test_classify_message_type(msg, expected):
    assert preprocessor.classify_message_type(msg) == expected


# ---------------------------------------------------------------------------
# Integration with the bundled sample_chat.txt
# ---------------------------------------------------------------------------

def test_sample_chat_file_parses_without_error():
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_chat.txt"
    )
    with open(sample_path, encoding="utf-8") as f:
        data = f.read()
    df = preprocessor.preprocess(data)
    assert not df.empty
    assert "group_notification" in set(df["user"])
    assert (df["message_type"] == "media").any()
