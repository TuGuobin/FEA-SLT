"""Text normalization shared by training contexts and evaluation."""

import unicodedata


def normalize_text(text, language):
    if language.lower() in ("chinese", "zh", "zh_cn"):
        return "".join(text.split())
    text = text.lower()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return " ".join(text.split())


def context_key(text):
    # Context exclusion must also recognize duplicate targets with different punctuation.
    return "".join(normalize_text(text, "german").split())
