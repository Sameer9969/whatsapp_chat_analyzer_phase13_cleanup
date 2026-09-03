"""
parser.py
----------------
Robust, modular parser that converts a raw WhatsApp exported .txt chat
into a clean, structured pandas DataFrame.

Design notes (why this is more robust than a "detect one global regex,
then re.split the whole file" approach):

  * WhatsApp export formats vary quite a bit across Android/iOS, phone
    locale, and app version -- separators (/, -, .), 12h/24h time,
    with/without seconds, with/without the iOS "[...]" bracket, and
    with/without a literal " - " between the timestamp and the sender.
    A single global regex + one strftime format (the old approach)
    breaks the moment a file mixes conventions or uses a format that
    wasn't explicitly enumerated.

  * Instead, this module scans the file **line by line**. Each line is
    tested against one flexible "message header" pattern. If it matches,
    it starts a new message. If it doesn't, the line is treated as a
    continuation of the previous message (this is how WhatsApp represents
    multi-line messages -- there is no per-line marker for continuations).

  * Once headers + raw bodies are extracted, actual datetime parsing is
    delegated to `pandas.to_datetime` (which itself wraps dateutil), with
    a `dayfirst=True` attempt first (the convention WhatsApp uses almost
    everywhere outside the US) and a `dayfirst=False` fallback for rows
    that still fail -- rather than a fixed list of strptime formats.

  * Every stage is defensive: malformed lines, stray pre-amble text,
    completely empty files, and undated garbage never raise -- they are
    simply skipped or dropped, and the function always returns a
    DataFrame (possibly empty) with a stable, documented schema.
"""

from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

__all__ = [
    "preprocess",
    "classify_message_type",
    "EMPTY_COLUMNS",
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Columns kept for backward compatibility with helper.py / app.py / db.py:
#   date, user, message, only_date, year, month_num, month, day, day_name,
#   hour, minute
# New columns added in this phase:
#   datetime   -> alias of `date` (explicit full timestamp, same dtype)
#   time       -> "HH:MM" string extracted straight from the export
#   sender     -> alias of `user` (clearer name for new/external code)
#   message_type -> one of: 'text', 'media', 'deleted', 'system'
EMPTY_COLUMNS = [
    "date", "datetime", "only_date", "time",
    "year", "month_num", "month", "day", "day_name", "hour", "minute",
    "user", "sender", "message", "message_type",
]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches the *start* of a new WhatsApp message line, e.g.:
#   12/05/23, 9:15 pm - Aditi: Hey everyone!
#   12/05/2023, 21:15 - Aditi: Hey everyone!
#   [12/05/23, 9:15:00 PM] Aditi: Hey everyone!
#   2023-05-12, 21:15 - Aditi: Hey everyone!
#   May 12, 2023, 9:15 PM - Aditi: Hey everyone!
# The optional leading '[' / trailing ']' handles iOS exports; the
# optional '-'/'–' handles Android exports; both are tolerated even if
# absent so a variety of real-world exports match.
_HEADER_RE = re.compile(
    r"""^\[?
        (?P<date>\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}
                 |[A-Za-z]{3,9}\.?\s\d{1,2},?\s\d{2,4})
        ,?\s*
        (?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:[APap]\.?[Mm]\.?)?)
        \]?
        \s*[-\u2013]?\s*
        (?P<rest>.*)$
    """,
    re.VERBOSE,
)

# First line of a message body: "Sender Name: actual message text"
# Sender names practically never contain a colon, so splitting on the
# first ": " is safe and matches what WhatsApp itself does.
_SENDER_RE = re.compile(r"^(?P<sender>[^:\n]{1,64}?):\s(?P<msg>.*)$")

# Media placeholders across export locales/app versions (with and
# without the legacy angle-bracket wrapper).
_MEDIA_RE = re.compile(
    r"(<Media omitted>|image omitted|video omitted|audio omitted|"
    r"sticker omitted|gif omitted|document omitted|contact card omitted)",
    re.IGNORECASE,
)

_DELETED_RE = re.compile(
    r"(this message was deleted|you deleted this message)",
    re.IGNORECASE,
)

# Characters WhatsApp sometimes uses instead of a normal space around
# AM/PM (iOS narrow no-break space) or as separators; normalize upfront.
_WHITESPACE_FIX = {
    "\u202f": " ",  # narrow no-break space
    "\xa0": " ",    # no-break space
}


def _normalize_text(data: str) -> str:
    """Fix known unicode whitespace quirks without altering real content."""
    if not data:
        return ""
    for bad, good in _WHITESPACE_FIX.items():
        data = data.replace(bad, good)
    # Normalize Windows/old-Mac line endings to '\n'
    data = data.replace("\r\n", "\n").replace("\r", "\n")
    return data


def classify_message_type(message: str) -> str:
    """Classify a *sender-attributed* message body as 'media', 'deleted',
    or 'text'. System/group-notification lines are classified separately
    (as 'system') by the caller, since they have no sender at all."""
    if not message:
        return "text"
    if _DELETED_RE.search(message):
        return "deleted"
    if _MEDIA_RE.search(message):
        return "media"
    return "text"


def _split_into_entries(data: str) -> List[dict]:
    """Line-by-line pass: groups the raw text into a list of
    {date, time, rest} dicts, one per WhatsApp message, folding
    multi-line message bodies into a single `rest` string.

    Any content before the first recognizable header (stray preamble,
    corrupted bytes, etc.) is silently discarded rather than raising.
    """
    entries: List[dict] = []
    current: Optional[dict] = None

    for line in data.split("\n"):
        match = _HEADER_RE.match(line)
        if match:
            if current is not None:
                entries.append(current)
            current = {
                "date": match.group("date"),
                "time": match.group("time"),
                "rest": match.group("rest"),
            }
        else:
            if current is not None:
                # Continuation of a multi-line message.
                current["rest"] += "\n" + line
            # else: garbage before the first valid message -> skipped.

    if current is not None:
        entries.append(current)

    return entries


def _split_sender_message(rest: str):
    """Given the raw body captured after a timestamp, split it into
    (user, message, message_type). If no 'Name: ' prefix is found on the
    first line, the entry is treated as a system/group-notification
    message (e.g. 'Aditi added Rohan', 'Messages are end-to-end
    encrypted...')."""
    first_line, _, remainder = rest.partition("\n")
    m = _SENDER_RE.match(first_line)
    if m:
        user = m.group("sender").strip()
        message = m.group("msg")
        if remainder:
            message = message + "\n" + remainder
        message = message.strip()
        return user, message, classify_message_type(message)

    # No "Name: " prefix -> system / group notification line.
    message = rest.strip()
    return "group_notification", message, "system"


def _normalize_time_str(time_str: str) -> str:
    """Clean up an extracted time token so pandas can parse it reliably,
    e.g. '9:15  p.m.' -> '9:15 pm'."""
    t = time_str.strip()
    t = re.sub(r"\.", "", t)  # drop periods in "a.m." / "p.m."
    t = re.sub(r"\s+", " ", t)
    return t


def preprocess(data: Optional[str]) -> pd.DataFrame:
    """Parse raw WhatsApp export text into a structured DataFrame.

    Never raises on malformed/empty input -- worst case it returns an
    empty DataFrame with the documented schema (see EMPTY_COLUMNS).

    Returns
    -------
    pd.DataFrame with columns:
        date, datetime, only_date, time, year, month_num, month, day,
        day_name, hour, minute, user, sender, message, message_type
    """
    data = _normalize_text(data or "")

    if not data.strip():
        return _empty_frame()

    entries = _split_into_entries(data)
    if not entries:
        return _empty_frame()

    rows = []
    for entry in entries:
        user, message, message_type = _split_sender_message(entry["rest"])
        rows.append({
            "date_str": entry["date"],
            "time_str": _normalize_time_str(entry["time"]),
            "user": user,
            "message": message,
            "message_type": message_type,
        })

    if not rows:
        return _empty_frame()

    df = pd.DataFrame(rows)

    combined = df["date_str"].astype(str) + " " + df["time_str"].astype(str)

    # format="mixed" lets pandas infer each row's layout without falling
    # back to a slow, warning-noisy per-element dateutil pass -- important
    # once a chat export has thousands of lines.
    parsed = pd.to_datetime(combined, dayfirst=True, format="mixed", errors="coerce")
    still_missing = parsed.isna()
    if still_missing.any():
        # Retry the failed rows assuming month-first (US-style) dates.
        fallback = pd.to_datetime(
            combined[still_missing], dayfirst=False, format="mixed", errors="coerce"
        )
        parsed.loc[still_missing] = fallback

    df["date"] = parsed
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    if df.empty:
        return _empty_frame()

    df["datetime"] = df["date"]
    df["only_date"] = df["date"].dt.date
    df["time"] = df["date"].dt.strftime("%H:%M")
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["sender"] = df["user"]

    df = df.drop(columns=["date_str", "time_str"])

    return df[EMPTY_COLUMNS]


def _empty_frame() -> pd.DataFrame:
    """A schema-correct, zero-row DataFrame so downstream code (which may
    call .dt accessors, groupby, etc.) never blows up on empty input."""
    df = pd.DataFrame({col: pd.Series(dtype="object") for col in EMPTY_COLUMNS})
    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ("year", "month_num", "day", "hour", "minute"):
        df[col] = df[col].astype("Int64")
    return df
