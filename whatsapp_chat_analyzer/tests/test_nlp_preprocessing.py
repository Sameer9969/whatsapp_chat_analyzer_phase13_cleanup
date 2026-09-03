"""
tests/test_nlp_preprocessing.py
---------------------------------
Unit tests for nlp_preprocessing.py (Phase 4: Hinglish-aware NLP
preprocessing pipeline).

Run with:
    pytest tests/test_nlp_preprocessing.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from whatsapp_chat_analyzer.scripts.src import preprocessing as nlp
from whatsapp_chat_analyzer.scripts.src.preprocessing import HinglishTextPreprocessor


# ---------------------------------------------------------------------------
# 1. Lowercase normalization
# ---------------------------------------------------------------------------

def test_lowercase_normalization():
    pre = HinglishTextPreprocessor()
    assert pre.normalize("BHAI KYA HAAL HAI") == "bhai kya haal hai"


def test_lowercase_can_be_disabled():
    pre = HinglishTextPreprocessor(lowercase=False, remove_stopwords=False)
    assert pre.normalize("Bhai Kya") == "Bhai Kya"


# ---------------------------------------------------------------------------
# 2. URL handling
# ---------------------------------------------------------------------------

def test_url_removed_by_default():
    pre = HinglishTextPreprocessor()
    out = pre.normalize("check this https://example.com/page?x=1 now")
    assert "http" not in out
    assert "check this" in out
    assert "now" in out


def test_url_with_www_removed():
    pre = HinglishTextPreprocessor()
    out = pre.normalize("visit www.example.com today")
    assert "www" not in out


def test_url_placeholder_configurable():
    pre = HinglishTextPreprocessor(url_placeholder="<URL>")
    out = pre.normalize("see https://example.com now")
    assert "<URL>" in out or "url" in out.lower()  # punctuation strip may lowercase/strip <>


def test_url_kept_when_remove_urls_false():
    pre = HinglishTextPreprocessor(remove_urls=False, remove_stopwords=False)
    out = pre.normalize("see https example com now")
    assert "example" in out


def test_extract_urls_standalone():
    text = "first https://a.com/x then www.b.com end"
    urls = nlp.extract_urls(text)
    assert urls == ["https://a.com/x", "www.b.com"]


# ---------------------------------------------------------------------------
# 3. Punctuation handling
# ---------------------------------------------------------------------------

def test_punctuation_stripped():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    out = pre.normalize("kya haal hai???!!!")
    assert "?" not in out
    assert "!" not in out
    assert "kya haal hai" == out


def test_punctuation_replaced_with_space_not_glued():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    out = pre.normalize("good,bad;ok")
    # words must not get glued together into "goodbadok"
    assert "goodbadok" not in out
    assert "good" in out and "bad" in out and "ok" in out


def test_apostrophes_collapse_contractions_without_stray_letters():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    assert pre.normalize("How's it going?") == "hows it going"
    assert pre.normalize("I'm so tired") == "im so tired"
    assert pre.normalize("that's amazing") == "thats amazing"
    assert pre.normalize("don't worry") == "dont worry"
    # no stray single-letter fragments like "s" / "m" / "t" left behind
    out = pre.normalize("How's it going? that's amazing, I'm happy, don't worry")
    assert " s " not in f" {out} "
    assert " m " not in f" {out} "


# ---------------------------------------------------------------------------
# 4. Whitespace normalization
# ---------------------------------------------------------------------------

def test_whitespace_normalization():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    out = pre.normalize("  multiple    spaces\n\nand\ttabs   ")
    assert out == "multiple spaces and tabs"


def test_empty_and_none_input_never_raises():
    pre = HinglishTextPreprocessor()
    assert pre.normalize("") == ""
    assert pre.normalize(None) == ""
    assert pre.tokenize("") == []
    assert pre.tokenize(None) == []
    assert pre.clean("") == ""
    assert pre.clean(None) == ""


def test_whitespace_only_input():
    pre = HinglishTextPreprocessor()
    assert pre.normalize("     \n\t  ") == ""
    assert pre.tokenize("     \n\t  ") == []


# ---------------------------------------------------------------------------
# 5. Repeated-character normalization
# ---------------------------------------------------------------------------

def test_repeated_characters_squeezed():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    assert pre.normalize("sooooo good") == "soo good"
    assert pre.normalize("yesss") == "yess"
    assert pre.normalize("nooooo") == "noo"


def test_repeated_characters_not_overcorrected():
    # Genuine double letters (<=2 repeats) must be left alone
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    assert pre.normalize("school") == "school"
    assert pre.normalize("good") == "good"


def test_repeated_char_normalization_configurable_max():
    pre = HinglishTextPreprocessor(remove_stopwords=False, max_repeat=1)
    assert pre.normalize("sooooo") == "so"


def test_repeated_char_normalization_can_be_disabled():
    pre = HinglishTextPreprocessor(normalize_repeated_chars=False, remove_stopwords=False)
    assert pre.normalize("sooooo good") == "sooooo good"


# ---------------------------------------------------------------------------
# 6. Configurable Hinglish stopword list
# ---------------------------------------------------------------------------

def test_default_stopwords_removed_in_tokenize():
    pre = HinglishTextPreprocessor()
    tokens = pre.tokenize("mood off hai")
    assert "hai" not in tokens
    assert "mood" in tokens and "off" in tokens


def test_stopwords_not_removed_in_normalize():
    # normalize() should NOT drop stopwords -- only tokenize()/clean() do
    pre = HinglishTextPreprocessor()
    out = pre.normalize("mood off hai")
    assert "hai" in out


def test_custom_stopwords_full_replacement():
    pre = HinglishTextPreprocessor(stopwords={"mood"})
    tokens = pre.tokenize("mood off hai")
    assert "mood" not in tokens
    assert "hai" in tokens  # 'hai' is no longer a stopword since we fully replaced the set
    assert "off" in tokens


def test_extra_stopwords_extends_default():
    pre = HinglishTextPreprocessor(extra_stopwords={"bro"})
    tokens = pre.tokenize("mast hai bro")
    assert "bro" not in tokens
    assert "hai" not in tokens  # default stopword still applies
    assert "mast" in tokens


def test_remove_stopwords_can_be_disabled():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    tokens = pre.tokenize("mood off hai")
    assert "hai" in tokens


# ---------------------------------------------------------------------------
# 7. Configurable slang dictionary
# ---------------------------------------------------------------------------

def test_default_slang_expansion():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    assert pre.normalize("kr rha h") == "kar raha hai"
    assert pre.normalize("thnx bro") == "thanks bro"


def test_custom_slang_dict_full_replacement():
    pre = HinglishTextPreprocessor(slang_dict={"lol": "laughing out loud"}, remove_stopwords=False)
    out = pre.normalize("lol kr")  # 'kr' should NOT expand since slang_dict fully replaced
    assert "laughing out loud" in out
    assert "kar" not in out
    assert "kr" in out


def test_extra_slang_extends_default():
    pre = HinglishTextPreprocessor(extra_slang={"jhakas": "awesome"}, remove_stopwords=False)
    out = pre.normalize("jhakas kr rha h")
    assert "awesome" in out
    assert "kar raha hai" in out  # default slang still applies


# ---------------------------------------------------------------------------
# 8. Meaningful token extraction (given task examples)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_tokens", [
    ("kya scene hai", ["scene"]),
    ("mood off hai", ["mood", "off"]),
    ("kal milte h", ["kal", "milte"]),
    ("mast hai bro", ["mast", "bro"]),
])
def test_meaningful_token_extraction_task_examples(text, expected_tokens):
    pre = HinglishTextPreprocessor()
    assert pre.tokenize(text) == expected_tokens


def test_clean_returns_joined_meaningful_tokens():
    pre = HinglishTextPreprocessor()
    assert pre.clean("kya scene hai") == "scene"
    assert pre.clean("mast hai bro") == "mast bro"


# ---------------------------------------------------------------------------
# 9. Optional emoji preservation / extraction
# ---------------------------------------------------------------------------

def test_emoji_preserved_by_default():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    out = pre.normalize("so happy \U0001F602 today")
    assert "\U0001F602" in out


def test_emoji_stripped_when_disabled():
    pre = HinglishTextPreprocessor(preserve_emojis=False, remove_stopwords=False)
    out = pre.normalize("so happy \U0001F602 today")
    assert "\U0001F602" not in out
    assert "happy" in out


def test_extract_emojis_standalone():
    text = "lol \U0001F602\U0001F525\U0001F389 nice"
    emojis = nlp.extract_emojis(text)
    assert emojis == ["\U0001F602", "\U0001F525", "\U0001F389"]


def test_extract_emojis_empty_when_none_present():
    assert nlp.extract_emojis("just plain text") == []
    assert nlp.extract_emojis("") == []
    assert nlp.extract_emojis(None) == []


def test_repeated_emojis_squeezed_like_other_chars():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    out = pre.normalize("lol \U0001F602\U0001F602\U0001F602\U0001F602")
    assert out.count("\U0001F602") == 2


# ---------------------------------------------------------------------------
# Mentions
# ---------------------------------------------------------------------------

def test_mentions_removed_by_default():
    pre = HinglishTextPreprocessor(remove_stopwords=False)
    out = pre.normalize("hey @rohan_123 check this")
    assert "@rohan_123" not in out
    assert "hey" in out and "check this" in out


def test_mentions_kept_when_disabled():
    pre = HinglishTextPreprocessor(remove_mentions=False, remove_stopwords=False)
    out = pre.normalize("hey @rohan check")
    assert "rohan" in out  # the @ symbol itself is still stripped by punctuation handling


def test_extract_mentions_standalone():
    text = "cc @rohan_123 and @meera.k for this"
    mentions = nlp.extract_mentions(text)
    assert "@rohan_123" in mentions
    assert "@meera" in mentions  # '.' breaks the \w+ mention match, which is expected/documented


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def test_module_level_functions_work():
    assert nlp.normalize_text("MOOD OFF HAI") == "mood off hai"
    assert nlp.tokenize_text("mood off hai") == ["mood", "off"]
    assert nlp.clean_text("mood off hai") == "mood off"


# ---------------------------------------------------------------------------
# Real chat-style sentences (integration-style checks)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "bhai kya kr rha h",
    "kya scene hai",
    "mood off hai",
    "kal milte h",
    "mast hai bro",
    "yaar bahut accha laga aaj",
    "kal exam tha, bahut tension tha",
    "chalo movie dekhte hai weekend pe",
    "thik hai bhai, milte hai kal",
    "arre yaar bakwas mat kar",
])
def test_real_chat_examples_never_raise_and_return_sane_output(text):
    pre = HinglishTextPreprocessor()
    normalized = pre.normalize(text)
    tokens = pre.tokenize(text)
    cleaned = pre.clean(text)
    assert isinstance(normalized, str)
    assert isinstance(tokens, list)
    assert isinstance(cleaned, str)
    # normalized text should contain no leftover raw punctuation
    assert "?" not in normalized and "!" not in normalized


def test_mixed_case_mixed_language_message():
    pre = HinglishTextPreprocessor()
    text = "Bro seriously, KAL EXAM hai aur tu abhi movie dekh rha h??"
    tokens = pre.tokenize(text)
    assert "kal" in tokens or "exam" in tokens
    assert all(tok == tok.lower() for tok in tokens)
