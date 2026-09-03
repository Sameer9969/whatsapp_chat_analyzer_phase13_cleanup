"""
preprocessing.py
---------------------
Phase 4: Hinglish-aware NLP preprocessing pipeline.

A reusable, rule-based text-cleaning pipeline for short, informal
WhatsApp-style messages that mix:
    - English
    - Hindi written in Roman script ("Hinglish")
    - common Indian chat slang and abbreviations
    - WhatsApp-specific noise (URLs, @mentions, elongated letters,
      punctuation spam, emojis)

IMPORTANT -- what this module is and is NOT:
    This is a lightweight, dictionary- and regex-based normalization
    layer, not a linguistic parser. It does NOT perform part-of-speech
    tagging, transliteration to Devanagari, language identification, or
    any kind of deep semantic understanding of Hindi or Hinglish. It
    cannot resolve ambiguous romanized spellings, disambiguate
    homographs, or understand grammar/sentence structure. It simply:
      1. Normalizes surface-level noise (case, URLs, mentions, repeated
         characters, punctuation, whitespace).
      2. Expands a configurable set of common chat abbreviations/slang
         to a more canonical spelling (e.g. "kr" -> "kar", "h" -> "hai").
      3. Removes a configurable list of high-frequency
         English/Hinglish filler words ("stopwords") to surface the more
         meaningful content words for downstream tasks like TF-IDF.

    Treat its output as "cleaned, code-mixed conversational text" -- a
    reasonable, explainable preprocessing step for a student ML project,
    not a claim of true Hindi/Hinglish language understanding.

Reusability:
    Every stage is exposed independently (normalize / tokenize / clean /
    extract_emojis / extract_urls / extract_mentions) so the future ML
    pipeline (legacy_sentiment_model.py / scripts/train_sentiment_model.py, in a later
    phase) can call `clean_text(message)` to get vectorizer-ready text,
    or use the individual pieces (e.g. extracted emojis as a separate
    feature) without needing to reimplement any of this logic.

Example
-------
>>> pre = HinglishTextPreprocessor()
>>> pre.normalize("bhai kya kr rha h??  check https://x.com/y @918888888")
'bhai kya kar raha hai'
>>> pre.tokenize("kya scene hai")
['scene']
>>> pre.clean("mast hai bro!!!")
'mast bro'
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set

__all__ = [
    "HinglishTextPreprocessor",
    "DEFAULT_HINGLISH_STOPWORDS",
    "DEFAULT_SLANG_DICT",
    "extract_emojis",
    "extract_urls",
    "extract_mentions",
    "normalize_text",
    "tokenize_text",
    "clean_text",
]

# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
# Apostrophes (straight and curly) are deleted outright rather than
# turned into whitespace, so contractions collapse cleanly:
# "don't" -> "dont", "How's" -> "hows" -- instead of leaving stray
# single-letter fragments like "don t" / "how s" behind.
_APOSTROPHE_RE = re.compile(r"[\u2019']")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ identical chars in a row

# Common emoji code-point ranges (emoticons, symbols, transport, flags,
# dingbats, supplemental symbols/pictographs). Not a complete Unicode
# emoji spec (e.g. skin-tone modifiers / ZWJ sequences are matched
# character-by-character rather than as a single grapheme), but covers
# the emoji that actually show up in everyday WhatsApp chats.
_EMOJI_RANGES = (
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"  # arrows (occasionally used decoratively)
)
_EMOJI_FINDALL_RE = re.compile(f"[{_EMOJI_RANGES}]")
_KEEP_WITH_EMOJI_RE = re.compile(rf"[^\w\s{_EMOJI_RANGES}]", re.UNICODE)
_KEEP_NO_EMOJI_RE = re.compile(r"[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------------------
# Default configurable resources
# ---------------------------------------------------------------------------

# Common English filler/function words relevant to short chat messages.
_ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "so", "if", "than",
    "then", "with", "as", "by", "from", "up", "down", "out", "no", "not", "do",
    "does", "did", "will", "would", "can", "could", "should", "just", "very",
    "there", "here", "what", "when", "where", "who", "which", "how",
}

# Common Hindi function words / particles written in Roman script.
_HINGLISH_STOPWORDS: Set[str] = {
    "hai", "hain", "ho", "hoon", "hu", "hun", "tha", "thi", "the",
    "raha", "rahi", "rahe", "rha", "rhi", "rhe",
    "kar", "krna", "karna", "karta", "karti", "karte", "kiya", "kiye",
    "ka", "ke", "ki", "ko", "se", "me", "mein", "mai", "main",
    "mujhe", "mje", "tum", "tumhe", "tumko", "aap", "aapko",
    "hum", "humein", "unko", "unhe", "uska", "uski", "uske",
    "iska", "iski", "iske", "yeh", "ye", "woh", "wo", "is", "us", "in", "un",
    "kya", "kyun", "kyu", "kaise", "kese", "kab", "kahan", "kaha",
    "jab", "agar", "magar", "lekin", "par", "pr", "aur", "ya", "toh", "to",
    "bhi", "hi", "na", "nahi", "nahin", "nai", "haan", "ha", "ji",
    "tak", "tk", "abhi", "abi", "phir", "fir", "wahan", "yahan",
    "idhar", "udhar", "apna", "apni", "apne", "kuch", "sab", "sabhi",
    "koi", "kisi", "ek", "do",
}

#: Default combined stopword set. Fully overridable/extendable via the
#: `stopwords=` / `extra_stopwords=` constructor arguments.
DEFAULT_HINGLISH_STOPWORDS: Set[str] = frozenset(_ENGLISH_STOPWORDS | _HINGLISH_STOPWORDS)

#: Default slang/abbreviation -> canonical-form mapping. Not exhaustive --
#: extend it via `slang_dict=` / `extra_slang=`. Covers common
#: WhatsApp/SMS-style English abbreviations and common romanized-Hindi
#: shorthand.
DEFAULT_SLANG_DICT: Dict[str, str] = {
    # Romanized-Hindi shorthand -> fuller spelling
    "kr": "kar", "krta": "karta", "krti": "karti", "krte": "karte",
    "rha": "raha", "rhi": "rahi", "rhe": "rahe",
    "h": "hai", "hu": "hoon", "hun": "hoon",
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi",
    "kyu": "kyun", "kese": "kaise",
    "acha": "accha", "achha": "accha",
    "thik": "theek", "thk": "theek",
    "abi": "abhi", "kl": "kal",
    "mje": "mujhe", "mjhe": "mujhe", "tje": "tujhe", "tjhe": "tujhe",
    "apka": "aapka", "apki": "aapki", "apke": "aapke",
    "plz": "please", "pls": "please", "plzz": "please",
    "u": "you", "ur": "your", "r": "are",
    "wat": "what", "wut": "what",
    "gud": "good", "gd": "good",
    "thnx": "thanks", "thx": "thanks", "tq": "thanks",
    "bcoz": "because", "bcz": "because", "cuz": "because", "coz": "because",
    "wid": "with", "ppl": "people", "msg": "message", "abt": "about",
    "rn": "right now", "asap": "as soon as possible", "np": "no problem",
    "ikr": "i know right", "brb": "be right back", "ttyl": "talk to you later",
    "gm": "good morning", "gn": "good night", "hbd": "happy birthday",
    "idk": "i dont know", "btw": "by the way", "omg": "oh my god",
    "gr8": "great", "l8r": "later", "2day": "today", "2moro": "tomorrow",
    "k": "okay", "kk": "okay", "okie": "okay",
}


# ---------------------------------------------------------------------------
# Standalone extraction helpers (operate on ORIGINAL/raw text, since
# normalize() strips URLs/mentions/etc. and they'd no longer be findable
# afterwards)
# ---------------------------------------------------------------------------

def extract_emojis(text: Optional[str]) -> List[str]:
    """Returns every individual emoji character found in `text`, in
    order (duplicates included). Matches common emoji Unicode ranges;
    does not merge multi-codepoint emoji sequences (skin tones, ZWJ
    combinations) into single logical emoji."""
    if not text:
        return []
    matches = _EMOJI_FINDALL_RE.findall(text)
    return list(matches)


def extract_urls(text: Optional[str]) -> List[str]:
    """Returns every URL found in `text`, in order."""
    if not text:
        return []
    return _URL_RE.findall(text)


def extract_mentions(text: Optional[str]) -> List[str]:
    """Returns every @mention token found in `text`, in order."""
    if not text:
        return []
    return _MENTION_RE.findall(text)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class HinglishTextPreprocessor:
    """Configurable text-cleaning pipeline for code-mixed WhatsApp text.

    Every stage can be toggled independently, and both the stopword list
    and the slang dictionary can be fully replaced or just extended, so
    downstream code (or a future ML training script) can tune this
    without editing this file.

    Parameters
    ----------
    lowercase : bool
        Lowercase the text first (default True).
    remove_urls : bool
        Strip URLs, replacing each with `url_placeholder` (default True).
    url_placeholder : str
        What to replace a URL with when `remove_urls=True` (default: '').
    remove_mentions : bool
        Strip @mentions (default True).
    normalize_repeated_chars : bool
        Squeeze runs of 3+ identical characters down to `max_repeat`
        (e.g. "sooooo" -> "soo"), a common fix for expressive elongation
        in informal chat text (default True).
    max_repeat : int
        How many repeats to collapse down to (default 2).
    preserve_emojis : bool
        If True, punctuation stripping leaves emoji characters intact so
        they survive into the normalized text (default True). If False,
        emojis are stripped along with other punctuation -- use
        `extract_emojis()` beforehand if you need them separately.
    remove_stopwords : bool
        Whether `tokenize()`/`clean()` drop stopword tokens (default True).
        `normalize()` itself never removes stopwords -- only
        `tokenize()`/`clean()` do, so callers who just want surface
        cleanup (case/URLs/punctuation/slang) without losing words can
        use `normalize()` directly.
    stopwords : Optional[Iterable[str]]
        Full replacement stopword set. If None, uses
        `DEFAULT_HINGLISH_STOPWORDS`.
    extra_stopwords : Optional[Iterable[str]]
        Additional stopwords unioned onto the base set (ignored if
        `stopwords` is given, since that already fully replaces it --
        pass your extras inside `stopwords` in that case).
    slang_dict : Optional[Dict[str, str]]
        Full replacement slang dictionary. If None, uses
        `DEFAULT_SLANG_DICT`.
    extra_slang : Optional[Dict[str, str]]
        Additional slang mappings merged onto the base dictionary
        (ignored if `slang_dict` is given).
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        remove_urls: bool = True,
        url_placeholder: str = "",
        remove_mentions: bool = True,
        normalize_repeated_chars: bool = True,
        max_repeat: int = 2,
        preserve_emojis: bool = True,
        remove_stopwords: bool = True,
        stopwords: Optional[Iterable[str]] = None,
        extra_stopwords: Optional[Iterable[str]] = None,
        slang_dict: Optional[Dict[str, str]] = None,
        extra_slang: Optional[Dict[str, str]] = None,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.url_placeholder = url_placeholder
        self.remove_mentions = remove_mentions
        self.normalize_repeated_chars = normalize_repeated_chars
        self.max_repeat = max(1, int(max_repeat))
        self.preserve_emojis = preserve_emojis
        self.remove_stopwords = remove_stopwords

        if stopwords is not None:
            self.stopwords: Set[str] = set(stopwords)
        else:
            self.stopwords = set(DEFAULT_HINGLISH_STOPWORDS)
            if extra_stopwords:
                self.stopwords |= set(extra_stopwords)

        if slang_dict is not None:
            self.slang_dict: Dict[str, str] = dict(slang_dict)
        else:
            self.slang_dict = dict(DEFAULT_SLANG_DICT)
            if extra_slang:
                self.slang_dict.update(extra_slang)

    # -- extraction (delegate to the standalone module functions) -------

    @staticmethod
    def extract_emojis(text: Optional[str]) -> List[str]:
        return extract_emojis(text)

    @staticmethod
    def extract_urls(text: Optional[str]) -> List[str]:
        return extract_urls(text)

    @staticmethod
    def extract_mentions(text: Optional[str]) -> List[str]:
        return extract_mentions(text)

    # -- core pipeline ----------------------------------------------------

    def normalize(self, text: Optional[str]) -> str:
        """Surface-level cleanup + slang expansion. Does NOT remove
        stopwords (use `tokenize()`/`clean()` for that). Always returns a
        string (possibly empty), never raises on None/empty input.

        Pipeline: lowercase -> strip URLs -> strip mentions -> squeeze
        repeated characters -> delete apostrophes (so contractions like
        "don't"/"how's" collapse to "dont"/"hows" instead of leaving a
        stray letter behind) -> strip remaining punctuation (emoji-aware)
        -> expand slang -> collapse whitespace.
        """
        if not text:
            return ""

        t = str(text)

        if self.lowercase:
            t = t.lower()

        if self.remove_urls:
            t = _URL_RE.sub(self.url_placeholder, t)

        if self.remove_mentions:
            t = _MENTION_RE.sub(" ", t)

        if self.normalize_repeated_chars:
            t = _REPEATED_CHAR_RE.sub(lambda m: m.group(1) * self.max_repeat, t)

        t = _APOSTROPHE_RE.sub("", t)

        punct_pattern = _KEEP_WITH_EMOJI_RE if self.preserve_emojis else _KEEP_NO_EMOJI_RE
        t = punct_pattern.sub(" ", t)

        if self.slang_dict:
            tokens = t.split()
            tokens = [self.slang_dict.get(tok, tok) for tok in tokens]
            t = " ".join(tokens)

        t = _WHITESPACE_RE.sub(" ", t).strip()
        return t

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Normalizes `text`, then (if `remove_stopwords=True`) drops
        stopword tokens, returning the remaining "meaningful" tokens."""
        normalized = self.normalize(text)
        if not normalized:
            return []
        tokens = normalized.split()
        if self.remove_stopwords:
            tokens = [tok for tok in tokens if tok not in self.stopwords]
        return tokens

    def clean(self, text: Optional[str]) -> str:
        """Meaningful-token string, ready to hand to a vectorizer (e.g.
        TF-IDF) -- equivalent to `' '.join(self.tokenize(text))`."""
        return " ".join(self.tokenize(text))


# ---------------------------------------------------------------------------
# Module-level convenience functions (bound to a default-configured
# instance) for simple one-off use / easy import by future ML code.
# ---------------------------------------------------------------------------

_default_preprocessor = HinglishTextPreprocessor()

normalize_text = _default_preprocessor.normalize
tokenize_text = _default_preprocessor.tokenize
clean_text = _default_preprocessor.clean
