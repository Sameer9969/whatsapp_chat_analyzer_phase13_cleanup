"""
chat_search.py
--------------
PHASE 9 -- Chat Search & Filtering.

Pure, UI-independent search/filter utilities over the parsed chat
DataFrame produced by `preprocessor.preprocess()`. This module has no
Streamlit imports and does not know about widgets -- `app.py`'s "Chat
Search" tab is a thin UI layer wired on top of the functions below,
the same pattern used by `analytics.py` / `behavioral.py`.

Supported filters (all combinable):
    - keyword       substring or regex match against the message text
    - sender(s)     one or more chat participants
    - date range    inclusive start/end date
    - sentiment     predicted sentiment label(s), when available

Privacy: every function here operates entirely in-process on the
DataFrame already loaded in memory. No network calls are made and
nothing -- queries or chat content -- is ever sent anywhere. Search
runs 100% locally, same as the rest of the app.

Pagination: `paginate()` slices results down to one page at a time so
that rendering a match list never depends on the total number of
messages -- a chat with 200k messages and a broad query is exactly as
fast to *display* as one with 20.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Pagination defaults
# ---------------------------------------------------------------------------
DEFAULT_PAGE_SIZE = 50
PAGE_SIZE_OPTIONS = [25, 50, 100, 200]

# Columns shown in the results table, in order.
DISPLAY_COLUMNS = ["date", "user", "message", "message_type"]


def search_messages(
    df: pd.DataFrame,
    *,
    keyword: str = "",
    use_regex: bool = False,
    case_sensitive: bool = False,
    senders: Optional[Sequence[str]] = None,
    message_types: Optional[Sequence[str]] = None,
    start_date=None,
    end_date=None,
    sentiment: Optional[pd.Series] = None,
    sentiment_filter: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filters `df` by every supplied criterion and returns matching rows.

    All filters are optional and combine with AND semantics. Passing
    none of them returns `df` unchanged (a no-op filter).

    Parameters
    ----------
    df:
        A DataFrame with at least `user`, `message`, `message_type`,
        and `only_date` columns (as produced by `preprocessor.preprocess`).
    keyword:
        Substring (or, if `use_regex`, a regular expression) to match
        against the `message` column. Blank/whitespace-only means "no
        keyword filter" -- useful for pure sender/date/sentiment
        searches with no text query.
    use_regex:
        Treat `keyword` as a regular expression instead of a literal
        substring.
    case_sensitive:
        Case-sensitive matching (default is case-insensitive).
    senders:
        Keep only messages from these users. `None` or empty = no filter.
    message_types:
        Keep only these `message_type` values (e.g. "text", "media",
        "system"). `None` or empty = no filter.
    start_date, end_date:
        Inclusive date range filter on the `only_date` column. Both
        must be provided to take effect.
    sentiment:
        A Series of predicted sentiment labels aligned to `df`'s index
        (e.g. built from the sentiment-analysis tab's predictions).
        Pass `None` if sentiment hasn't been computed for this chat.
    sentiment_filter:
        Keep only rows whose sentiment (from `sentiment`) is in this
        list. Ignored if `sentiment` is `None`.

    Returns
    -------
    A filtered (possibly empty) DataFrame, a view/copy of `df` --
    callers should not assume it shares or doesn't share memory with
    the input.

    Raises
    ------
    re.error
        If `use_regex=True` and `keyword` is not a valid regular
        expression. Callers (the Streamlit UI) should catch this and
        show a friendly message instead of a stack trace.
    """
    result = df

    if senders:
        result = result[result["user"].isin(senders)]

    if message_types:
        result = result[result["message_type"].isin(message_types)]

    if start_date is not None and end_date is not None and "only_date" in result.columns:
        result = result[
            (result["only_date"] >= start_date) & (result["only_date"] <= end_date)
        ]

    if sentiment is not None and sentiment_filter:
        aligned = sentiment.reindex(result.index)
        result = result[aligned.isin(sentiment_filter)]

    keyword = (keyword or "").strip()
    if keyword:
        flags = 0 if case_sensitive else re.IGNORECASE
        if use_regex:
            pattern = re.compile(keyword, flags)
            mask = result["message"].str.contains(pattern, na=False, regex=True)
        else:
            mask = result["message"].str.contains(
                re.escape(keyword), case=case_sensitive, na=False, regex=True
            )
        result = result[mask]

    return result


def paginate(
    df: pd.DataFrame, page: int, page_size: int = DEFAULT_PAGE_SIZE
) -> Tuple[pd.DataFrame, int, int]:
    """Slices `df` into a single page of results.

    `page` is 1-indexed and is clamped into the valid `[1, total_pages]`
    range, so callers never need to pre-validate it (e.g. a stale page
    number left over after a filter change just clamps to the last
    valid page instead of raising or returning nothing).

    This is the guard against a huge chat export freezing the UI: no
    matter how many rows match, only one page's worth is ever sliced
    out and handed to the caller to render.

    Returns
    -------
    (page_df, total_pages, total_results)
    """
    total_results = len(df)
    if page_size <= 0:
        page_size = DEFAULT_PAGE_SIZE

    total_pages = max(1, -(-total_results // page_size))  # ceil division
    page = min(max(1, page), total_pages)

    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end], total_pages, total_results


def build_sentiment_series(
    analysis_index: Sequence, sentiment_labels: Sequence[str]
) -> pd.Series:
    """Builds an index-aligned sentiment Series for use as `search_messages`'s
    `sentiment` argument.

    `analysis_index` is the index of the DataFrame the sentiment labels
    were predicted on (e.g. `analysis_df.index`), and `sentiment_labels`
    is the list/array of predicted labels in the same order. Messages
    outside that index (e.g. media/system rows that were never
    classified, or rows excluded by a user-focus filter at prediction
    time) simply won't be present in the returned Series and will not
    match any sentiment filter -- which is the correct behavior, since
    they were never classified.
    """
    return pd.Series(list(sentiment_labels), index=list(analysis_index))
