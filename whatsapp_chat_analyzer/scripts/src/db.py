"""
db.py
-----
Persistence layer for the app. Stores parsed chat messages (with their
predicted sentiment) and a summary log of every analysis run, so past
uploads don't have to be re-processed from scratch.

Uses Python's built-in `sqlite3` by default -- zero setup, works
immediately on any machine. The connection is isolated behind
`get_connection()`, so switching to MySQL later only means changing
that one function (e.g. using `mysql-connector-python` or SQLAlchemy
with a `mysql+pymysql://user:pass@host/db` URL) -- no other file needs
to change.

Set the environment variable DATABASE_URL to point elsewhere, e.g.:
    DATABASE_URL=mysql://user:pass@localhost/whatsapp_analyzer
(MySQL support requires `pip install sqlalchemy pymysql` -- see README.)
"""

import os
import sqlite3
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, 'chat_analysis.db')
DATABASE_URL = os.environ.get('DATABASE_URL', '').strip()


def get_connection():
    """Returns a DB connection. Defaults to a local SQLite file; if
    DATABASE_URL is set to a MySQL URL and sqlalchemy+pymysql are
    installed, that is used instead."""
    if DATABASE_URL.startswith('mysql'):
        import sqlalchemy
        engine = sqlalchemy.create_engine(DATABASE_URL)
        return engine.raw_connection()
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            run_at TEXT,
            total_messages INTEGER,
            total_users INTEGER,
            positive_count INTEGER,
            neutral_count INTEGER,
            negative_count INTEGER
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            user TEXT,
            message TEXT,
            message_date TEXT,
            sentiment TEXT,
            FOREIGN KEY(run_id) REFERENCES analysis_runs(id)
        )
    ''')
    conn.commit()
    conn.close()


def save_analysis_run(file_name, df, sentiment_labels):
    """Persists one uploaded-chat analysis: a summary row + every message
    with its predicted sentiment. Returns the new run_id."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    total_messages = len(df)
    total_users = df['user'].nunique()
    pos = sentiment_labels.count('positive')
    neu = sentiment_labels.count('neutral')
    neg = sentiment_labels.count('negative')

    cur.execute(
        'INSERT INTO analysis_runs (file_name, run_at, total_messages, total_users, '
        'positive_count, neutral_count, negative_count) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (file_name, datetime.now().isoformat(timespec='seconds'), total_messages,
         total_users, pos, neu, neg)
    )
    run_id = cur.lastrowid

    rows = [
        (run_id, row.user, row.message, str(row.date), label)
        for row, label in zip(df.itertuples(index=False), sentiment_labels)
    ]
    cur.executemany(
        'INSERT INTO messages (run_id, user, message, message_date, sentiment) '
        'VALUES (?, ?, ?, ?, ?)',
        rows
    )
    conn.commit()
    conn.close()
    return run_id


def clear_all_history():
    """Deletes every saved run and message from local history. Used by the
    UI's "Clear saved history" button so users can wipe previously-opted-in
    chat data on demand. Returns the number of analysis_runs rows removed."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM analysis_runs')
    count = cur.fetchone()[0]
    cur.execute('DELETE FROM messages')
    cur.execute('DELETE FROM analysis_runs')
    conn.commit()
    conn.close()
    return count


def fetch_past_runs(limit=10):
    """Returns the most recent analysis runs, newest first, as a list of tuples."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        'SELECT id, file_name, run_at, total_messages, total_users, '
        'positive_count, neutral_count, negative_count '
        'FROM analysis_runs ORDER BY id DESC LIMIT ?',
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows
