"""
helper.py
---------
Core analysis "modules" referenced in the project report:
  - fetch_stats            -> top statistics (messages/words/media/links)
  - most_busy_users        -> busiest members + % share
  - create_wordcloud       -> word cloud image
  - most_common_words      -> top words (stop-words removed)
  - monthly_timeline / daily_timeline
  - week_activity_map / month_activity_map / activity_heatmap

Emoji analysis lives in its own module, emoji_analysis.py.

Each function takes `selected_user` ("Overall" or a specific member name)
and the parsed DataFrame from parser.py, and returns exactly what the
Streamlit UI needs to render (numbers, DataFrames, or matplotlib figures).
"""

import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import matplotlib.pyplot as plt

from . import common

try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except ImportError:
    _HAS_WORDCLOUD = False

# --- lightweight stopword list (English + common Hinglish chat words) ------
# Kept local (no NLTK download needed) so the app works fully offline.
STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves media omitted this message was deleted null
haan nahi hai ho toh bhi ka ki ke ko se me mein yeh woh hi bhi kya kyu kyun
kaise kab kaha wala wali abhi tha thi the aur ye
""".split())

_URL_RE = re.compile(r'(https?://\S+|www\.\S+)')


def _filter(df, selected_user):
    """Kept as a local alias (same behaviour as common.filter_by_user) so
    every existing call site below is unaffected."""
    return common.filter_by_user(df, selected_user)


def fetch_stats(selected_user, df):
    """Returns (num_messages, num_words, num_media, num_links)."""
    df = _filter(df, selected_user)

    num_messages = df.shape[0]
    num_words = sum(len(msg.split()) for msg in df['message'])
    num_media = df[df['message'].str.contains('<Media omitted>', na=False)].shape[0]
    num_links = sum(len(_URL_RE.findall(msg)) for msg in df['message'])

    return num_messages, num_words, num_media, num_links


def most_busy_users(df):
    """Overall-only: returns (counts_series, percent_dataframe)."""
    df = df[df['user'] != 'group_notification']
    counts = df['user'].value_counts()
    percent_df = (
        round(df['user'].value_counts() / df.shape[0] * 100, 2)
        .reset_index().rename(columns={'index': 'user', 'user': 'percent'})
    )
    percent_df.columns = ['user', 'percent']
    return counts, percent_df


def _clean_words(df):
    words = []
    for message in df['message']:
        for w in message.lower().split():
            w = re.sub(r'[^\w]', '', w)
            if w and w not in STOP_WORDS and '<media' not in w:
                words.append(w)
    return words


def create_wordcloud(selected_user, df):
    """Returns a matplotlib Figure with a word cloud, or None if the
    optional `wordcloud` package isn't installed."""
    df = _filter(df, selected_user)
    words = _clean_words(df)
    if not words:
        return None

    if _HAS_WORDCLOUD:
        wc = WordCloud(width=800, height=400, min_font_size=10, background_color='white')
        img = wc.generate(' '.join(words))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(img)
        ax.axis('off')
        return fig

    # Fallback: simple bar-based "cloud" so the feature still works
    # even without the wordcloud package installed.
    top = Counter(words).most_common(25)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    import random
    for word, count in top:
        ax.text(random.uniform(0, 1), random.uniform(0, 1), word,
                fontsize=8 + count, ha='center', va='center', alpha=0.8)
    return fig


def most_common_words(selected_user, df, top_n=20):
    df = _filter(df, selected_user)
    words = _clean_words(df)
    return pd.DataFrame(Counter(words).most_common(top_n), columns=['word', 'count'])


def monthly_timeline(selected_user, df):
    df = _filter(df, selected_user)
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    df = _filter(df, selected_user)
    return df.groupby('only_date').count()['message'].reset_index()


def week_activity_map(selected_user, df):
    df = _filter(df, selected_user)
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    df = _filter(df, selected_user)
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    df = _filter(df, selected_user)
    return df.pivot_table(index='day_name', columns='hour', values='message',
                           aggfunc='count').fillna(0)
