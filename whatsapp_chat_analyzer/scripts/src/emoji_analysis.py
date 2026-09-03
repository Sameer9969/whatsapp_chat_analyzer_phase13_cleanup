"""
src/emoji_analysis.py
----------------------
Emoji usage analysis, split out of helper.py into its own module.

`emoji_helper(selected_user, df)` returns a DataFrame of every emoji used
(by the selected user, or everyone if "Overall") ranked by frequency.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from . import common

# Broad emoji code-point ranges (covers the emoji blocks WhatsApp messages use)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    flags=re.UNICODE,
)


def emoji_helper(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame of (emoji, count) sorted by frequency, descending."""
    df = common.filter_by_user(df, selected_user)
    all_emojis = []
    for message in df["message"]:
        all_emojis.extend(_EMOJI_RE.findall(message))
    return pd.DataFrame(Counter(all_emojis).most_common(), columns=["emoji", "count"])
