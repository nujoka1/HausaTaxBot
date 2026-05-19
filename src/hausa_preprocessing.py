"""
Hausa Language Preprocessing Pipeline
=====================================

This module provides comprehensive Hausa text normalization and preprocessing
to improve NLP tasks including retrieval and classification.

Academic Purpose:
- Demonstrates language-specific preprocessing for low-resource NLP
- Improves retrieval accuracy for Hausa tax QA
- Handles Hausa diacritics and special characters

Author: HausaTaxBot Research Team
Project: COEN541 - Advanced NLP (Ahmadu Bello University)
"""

import re
import unicodedata
from typing import List, Optional, Dict
import logging

logger = logging.getLogger("HausaTaxBot.Preprocessing")


class HausaPreprocessor:
    """
    Hausa text preprocessing and normalization.
    
    Handles:
    - Hausa diacritical marks (glottals: ƙ, ɓ, ɗ)
    - Unicode normalization
    - Punctuation and whitespace cleanup
    - Optional stemming and stopword removal
    """
    
    # Hausa character mappings for normalization
    HAUSA_DIACRITICS = {
        'ƙ': 'k',  # Voiceless velar click (but preserve for now if needed)
        'ɓ': 'b',  # Voiced bilabial implosive
        'ɗ': 'd',  # Voiced alveolar implosive
        'Ƙ': 'K',
        'Ɓ': 'B',
        'Ɗ': 'D',
    }
    
    # Common Hausa stopwords
    HAUSA_STOPWORDS = {
        'ne', 'ni', 'ya', 'na', 'a', 'an', 'da', 'ta', 'sa', 'su',
        'ba', 'ce', 'ga', 'ka', 'ki', 'kowa', 'wai', 'amma',
        'amma', 'domin', 'ko', 'ko', 'in', 'don', 'har', 'har',
        'jiya', 'gida', 'shuni', 'zai', 'za', 'dole', 'dah'
    }
    
    def __init__(self, 
                 normalize_diacritics: bool = False,
                 remove_stopwords: bool = False,
                 lowercase: bool = True):
        """
        Initialize Hausa preprocessor.
        
        Args:
            normalize_diacritics: If True, convert ƙ→k, ɓ→b, ɗ→d (affects semantics, use carefully)
            remove_stopwords: If True, remove Hausa stopwords
            lowercase: If True, convert to lowercase
        """
        self.normalize_diacritics = normalize_diacritics
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        logger.info(f"HausaPreprocessor initialized: "
                   f"normalize_diacritics={normalize_diacritics}, "
                   f"remove_stopwords={remove_stopwords}, "
                   f"lowercase={lowercase}")
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to Hausa text.
        
        Pipeline:
        1. Normalize Unicode
        2. Lowercase (optional)
        3. Remove URLs and emails
        4. Remove special characters (preserve Hausa diacritics by default)
        5. Normalize whitespace
        6. Remove stopwords (optional)
        
        Args:
            text: Raw Hausa text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Unicode normalization (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Step 3: Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Step 4: Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Step 5: Optionally normalize diacritics
        if self.normalize_diacritics:
            text = self._normalize_diacritics(text)
        
        # Step 6: Remove punctuation and special characters
        # Keep Hausa diacritics and Arabic numerals
        text = re.sub(r'[^\w\sƙɓɗ]', '', text, flags=re.UNICODE)
        
        # Step 7: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 8: Remove stopwords
        if self.remove_stopwords:
            text = self._remove_stopwords(text)
        
        return text
    
    def _normalize_diacritics(self, text: str) -> str:
        """Convert Hausa diacritical marks to ASCII equivalents."""
        for hausa_char, ascii_char in self.HAUSA_DIACRITICS.items():
            text = text.replace(hausa_char, ascii_char)
        return text
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove Hausa stopwords while preserving text structure."""
        tokens = text.split()
        filtered = [t for t in tokens if t.lower() not in self.HAUSA_STOPWORDS]
        return ' '.join(filtered)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts efficiently.
        
        Args:
            texts: List of Hausa texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    @staticmethod
    def tokenize_hausa(text: str) -> List[str]:
        """
        Simple Hausa tokenization.
        
        Splits on whitespace and common punctuation.
        Preserves hyphenated words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Split on whitespace and specific punctuation
        tokens = re.split(r'[\s]+', text.strip())
        return [t for t in tokens if t]
    
    @staticmethod
    def get_character_ngrams(text: str, n: int = 3) -> List[str]:
        """
        Generate character-level n-grams (useful for Hausa morphology).
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            List of character n-grams
        """
        text = text.replace(' ', '')
        return [text[i:i+n] for i in range(len(text)-n+1)]


def create_preprocessor(normalize_diacritics: bool = False,
                       remove_stopwords: bool = False,
                       lowercase: bool = True) -> HausaPreprocessor:
    """
    Factory function to create a configured Hausa preprocessor.
    
    Args:
        normalize_diacritics: Whether to convert special Hausa characters
        remove_stopwords: Whether to remove Hausa stopwords
        lowercase: Whether to convert to lowercase
        
    Returns:
        Configured HausaPreprocessor instance
    """
    return HausaPreprocessor(
        normalize_diacritics=normalize_diacritics,
        remove_stopwords=remove_stopwords,
        lowercase=lowercase
    )
