"""
src/common.py
--------------
Small, shared utilities used across the analytics modules
(analytics.py, behavioral.py, helper.py, emoji_analysis.py).

These were previously copy-pasted with identical implementations in
multiple files (`_is_empty` / `_real_messages` in both analytics.py and
behavioral.py, plus an equivalent `_filter` in helper.py). Centralizing
them here removes that duplication -- every module now shares one
definition of "what counts as a real, user-authored message" and "how
do we narrow a DataFrame to a single selected user".
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

SYSTEM_USER = "group_notification"


def is_empty(df: Optional[pd.DataFrame]) -> bool:
    """True if `df` is None or has no rows."""
    return df is None or len(df) == 0


def exclude_system_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Drop system/group-notification rows -- analytics and behavioral
    metrics should only ever operate on actual user-authored messages."""
    if is_empty(df):
        return df
    return df[df["user"] != SYSTEM_USER]


def filter_by_user(df: pd.DataFrame, selected_user: str = "Overall") -> pd.DataFrame:
    """Real (non-system) messages, optionally narrowed to a single user --
    mirrors the `selected_user` convention used across the whole app."""
    data = exclude_system_messages(df)
    if is_empty(data):
        return data
    if selected_user and selected_user != "Overall":
        data = data[data["user"] == selected_user]
    return data
