"""
src/preprocessing.py
HausaTaxBot — Hausa text preprocessing pipeline
COEN541 · Ahmadu Bello University, Zaria

Responsibilities: Eng. Nuhu (pipeline design)
"""

import re
import string
import unicodedata
from typing import List, Optional


# ─────────────────────────────────────────────
# Hausa stopwords (curated for tax domain)
# ─────────────────────────────────────────────
HAUSA_STOPWORDS = {
    # Pronouns
    "ni", "kai", "ita", "shi", "mu", "ku", "su",
    "na", "ta", "ya", "ka", "ki", "mun", "kun", "sun",
    # Common particles & prepositions
    "da", "a", "an", "ba", "ko", "don", "daga", "ga",
    "har", "idan", "in", "kuma", "kamar", "ne", "ce",
    "wai", "sai", "tare", "zuwa", "akan", "bayan",
    "tsakanin", "lokacin", "kafin", "bisa", "game",
    # Articles / definiteness
    "wannan", "wancan", "wadannan", "wadancan",
    # Common verbs (auxiliary)
    "ake", "ana", "zai", "za", "ta", "ya", "suke",
    "yake", "take", "muke", "kuke", "suna", "shine",
    # Question words (kept short — full forms stay for retrieval)
    "me", "wane", "wanda",
    # Filler / discourse
    "fa", "kuwa", "ke", "nan", "can",
}

# Tax-domain keywords to NEVER strip (override stopword removal)
TAX_PRESERVE = {
    "haraji", "biya", "kamfani", "albashi", "kudin",
    "shiga", "kaso", "adadin", "dokar", "firs", "vat",
    "wht", "cit", "pit", "cgt", "dst", "ribar", "kame",
    "dijital", "sabis", "kaya", "rijista", "gwamnati",
}


class HausaPreprocessor:
    """
    End-to-end Hausa text preprocessor for the tax law chatbot.
    Designed to run identically in training (corpus) and inference (query).

    Usage:
        pp = HausaPreprocessor()
        clean = pp.process("Menene kason VAT da ke aiki a Najeriya?")
        # → "menene kason vat aiki najeriya"
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        min_token_length: int = 2,
    ):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length

    # ── public API ──────────────────────────────

    def process(self, text: str) -> str:
        """Full pipeline: clean → tokenize → filter → rejoin."""
        tokens = self.tokenize(self.clean(text))
        tokens = self.filter_tokens(tokens)
        return " ".join(tokens)

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(t) for t in texts]

    # ── pipeline steps ───────────────────────────

    def clean(self, text: str) -> str:
        """Normalize unicode, strip HTML artifacts, lowercase."""
        # Normalize unicode (handles Hausa tone marks if any)
        text = unicodedata.normalize("NFKC", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove digits (tax amounts are in answers, not useful for retrieval)
        text = re.sub(r"\d+", "", text)
        # Remove punctuation (keep apostrophes for Hausa glottal)
        if self.remove_punctuation:
            text = text.translate(
                str.maketrans("", "", string.punctuation.replace("'", ""))
            )
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization (Hausa is space-delimited)."""
        return text.split()

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Remove stopwords and short tokens, preserving tax keywords."""
        filtered = []
        for token in tokens:
            # Always keep tax-domain keywords
            if token in TAX_PRESERVE:
                filtered.append(token)
                continue
            # Remove stopwords
            if self.remove_stopwords and token in HAUSA_STOPWORDS:
                continue
            # Remove very short tokens
            if len(token) < self.min_token_length:
                continue
            filtered.append(token)
        return filtered


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pp = HausaPreprocessor()

    samples = [
        "Menene kason VAT da ke aiki a Najeriya a yanzu?",
        "Kamfanoni masu karamin kasuwanci suna biyan nawa haraji?",
        "Ta yaya ake lissafa haraji akan albashi na wata-wata?",
    ]

    print("=" * 60)
    print("HausaPreprocessor — sample output")
    print("=" * 60)
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {pp.process(s)}")
        print()
