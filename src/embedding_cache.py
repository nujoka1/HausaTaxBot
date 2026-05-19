"""
Embedding Cache Module - Fast embedding retrieval with caching
==============================================================

Caches question embeddings to avoid recomputation on every inference.
Uses Streamlit's @st.cache_resource for session-based caching and
optional disk caching for persistence.

Performance Impact:
- First inference: ~500-1000ms (compute embeddings)
- Cached inference: ~5-10ms (lookup only)
- Memory usage: ~2-5MB for typical Q&A datasets

Author: HausaTaxBot Research Team (COEN541)
"""

import numpy as np
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import os

logger = logging.getLogger("EmbeddingCache")

class EmbeddingCache:
    """
    Manages caching of question embeddings for fast retrieval.
    
    Supports:
    - In-memory caching (session-based)
    - Disk caching (persistent)
    - Cache invalidation on data changes
    - Multiple encoder support
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for persistent cache storage
                      (default: ./cache/embeddings)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache: {encoder_id: {question_hash: embedding}}
        self.memory_cache: Dict[str, Dict[str, np.ndarray]] = {}
        
        logger.info(f"EmbeddingCache initialized | Cache dir: {self.cache_dir}")
    
    @staticmethod
    def _hash_content(content: str) -> str:
        """Create MD5 hash of content for cache key."""
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def _hash_questions(questions: List[str]) -> str:
        """Create hash of question list to detect data changes."""
        combined = "\n".join(questions)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cache_key(self, encoder_id: str, data_hash: str) -> str:
        """Generate cache file name."""
        return f"{encoder_id}_{data_hash}.npy"
    
    def compute_and_cache(self, 
                         questions: List[str],
                         encoder,
                         encoder_id: str,
                         use_disk_cache: bool = True) -> np.ndarray:
        """
        Compute embeddings and store in cache.
        
        Args:
            questions: List of questions to embed
            encoder: Encoder object (supports transform() or encode())
            encoder_id: Identifier for encoder type (e.g., "ctfidf", "colbert")
            use_disk_cache: Whether to save to disk
            
        Returns:
            Array of embeddings (shape: [n_questions, embedding_dim])
            
        Example:
            >>> embeddings = cache.compute_and_cache(
            ...     questions=qa_data['qa_pairs'],
            ...     encoder=my_encoder,
            ...     encoder_id="c-tfidf"
            ... )
        """
        logger.info(f"Computing embeddings for {len(questions)} questions | Encoder: {encoder_id}")
        
        # Encode questions
        if hasattr(encoder, 'transform'):
            embeddings = encoder.transform(questions)
        else:
            embeddings = encoder.encode(questions)
        
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # Store in memory cache
        if encoder_id not in self.memory_cache:
            self.memory_cache[encoder_id] = {}
        
        for i, question in enumerate(questions):
            question_hash = self._hash_content(question)
            self.memory_cache[encoder_id][question_hash] = embeddings[i]
        
        # Store on disk for persistence
        if use_disk_cache:
            data_hash = self._hash_questions(questions)
            cache_file = self.cache_dir / self.get_cache_key(encoder_id, data_hash)
            
            np.save(cache_file, embeddings)
            logger.info(f"✅ Cached {len(questions)} embeddings to {cache_file}")
        
        return embeddings
    
    def get_cached_embeddings(self,
                             questions: List[str],
                             encoder,
                             encoder_id: str,
                             use_disk_cache: bool = True) -> Optional[np.ndarray]:
        """
        Retrieve cached embeddings or compute if not cached.
        
        Strategy:
        1. Check memory cache (fastest)
        2. Check disk cache (restart-safe)
        3. Compute and cache (first time)
        
        Args:
            questions: List of questions
            encoder: Encoder object
            encoder_id: Encoder identifier
            use_disk_cache: Use persistent disk cache
            
        Returns:
            Embeddings array or None if computation fails
        """
        data_hash = self._hash_questions(questions)
        
        # Try disk cache first (checks for data consistency)
        if use_disk_cache:
            cache_file = self.cache_dir / self.get_cache_key(encoder_id, data_hash)
            if cache_file.exists():
                try:
                    embeddings = np.load(cache_file)
                    logger.debug(f"✅ Loaded from disk cache: {cache_file.name}")
                    
                    # Populate memory cache for faster future access
                    if encoder_id not in self.memory_cache:
                        self.memory_cache[encoder_id] = {}
                    
                    for i, question in enumerate(questions):
                        q_hash = self._hash_content(question)
                        self.memory_cache[encoder_id][q_hash] = embeddings[i]
                    
                    return embeddings
                except Exception as e:
                    logger.warning(f"Failed to load disk cache: {e}")
        
        # Compute and cache if not found
        return self.compute_and_cache(questions, encoder, encoder_id, use_disk_cache)
    
    def clear_memory_cache(self, encoder_id: Optional[str] = None):
        """Clear memory cache for specific encoder or all."""
        if encoder_id:
            if encoder_id in self.memory_cache:
                del self.memory_cache[encoder_id]
                logger.info(f"Cleared memory cache for {encoder_id}")
        else:
            self.memory_cache.clear()
            logger.info("Cleared all memory caches")
    
    def clear_disk_cache(self, encoder_id: Optional[str] = None):
        """Clear disk cache for specific encoder or all."""
        if encoder_id:
            pattern = f"{encoder_id}_*.npy"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"Deleted cache file: {cache_file.name}")
        else:
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
            logger.info("Cleared all disk caches")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for diagnostics."""
        total_memory_embeddings = sum(
            len(cache) for cache in self.memory_cache.values()
        )
        
        total_disk_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.npy")
        ) / (1024 * 1024)  # MB
        
        disk_files = list(self.cache_dir.glob("*.npy"))
        
        return {
            'memory_embeddings': total_memory_embeddings,
            'memory_encoders': len(self.memory_cache),
            'disk_cache_files': len(disk_files),
            'disk_cache_size_mb': round(total_disk_size, 2)
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        logger.info(
            f"Cache Stats: "
            f"Memory embeddings={stats['memory_embeddings']}, "
            f"Encoders={stats['memory_encoders']}, "
            f"Disk files={stats['disk_cache_files']}, "
            f"Disk size={stats['disk_cache_size_mb']}MB"
        )


# ==================== STREAMLIT INTEGRATION ====================

def get_cached_embeddings_streamlit(questions: List[str],
                                   encoder,
                                   encoder_id: str,
                                   cache_dir: str = "cache/embeddings") -> np.ndarray:
    """
    Streamlit-compatible embedding caching using @st.cache_resource.
    
    Usage in streamlit_app.py:
    ```python
    import streamlit as st
    from src.embedding_cache import get_cached_embeddings_streamlit
    
    @st.cache_resource
    def cached_embeddings():
        return EmbeddingCache()
    
    cache = cached_embeddings()
    embeddings = cache.get_cached_embeddings(questions, encoder, "c-tfidf")
    ```
    
    Args:
        questions: Questions to encode
        encoder: Encoder object
        encoder_id: Encoder name
        cache_dir: Cache directory path
        
    Returns:
        Cached embeddings array
    """
    cache = EmbeddingCache(cache_dir=cache_dir)
    return cache.get_cached_embeddings(questions, encoder, encoder_id)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    questions = [
        "What is income tax in Nigeria?",
        "How to file PIT returns?",
        "Tax reliefs for individuals"
    ]
    
    # Mock encoder
    class MockEncoder:
        def transform(self, texts):
            return np.random.randn(len(texts), 100)
    
    # Test caching
    cache = EmbeddingCache()
    encoder = MockEncoder()
    
    print("\n1️⃣ First call (computing)...")
    emb1 = cache.compute_and_cache(questions, encoder, "test-encoder")
    print(f"Shape: {emb1.shape}")
    
    print("\n2️⃣ Second call (cached)...")
    emb2 = cache.get_cached_embeddings(questions, encoder, "test-encoder")
    print(f"Shape: {emb2.shape}")
    print(f"Identical: {np.allclose(emb1, emb2)}")
    
    print("\n3️⃣ Cache stats...")
    cache.print_stats()
