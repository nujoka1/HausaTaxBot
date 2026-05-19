"""
Retrieval Pipeline with Confidence Calibration
==============================================

This module implements a stable, well-calibrated retrieval system that:

1. Encodes user queries into semantic embeddings
2. Searches Q&A database using cosine similarity
3. Optionally filters by intent using classifier
4. Returns answers with calibrated confidence scores
5. Falls back to keyword matching when ML confidence is low

Academic Purpose:
- Demonstrates proper confidence calibration in IR systems
- Shows multiple retrieval strategies (semantic + keyword)
- Includes detailed logging for retrieval diagnostics
- Supports model comparison through abstraction

Author: HausaTaxBot Research Team
Project: COEN541 - Advanced NLP
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("HausaTaxBot.Retrieval")


class ConfidenceLevel(Enum):
    """Confidence classification for retrieval results."""
    HIGH = "🟢 TABBATA (High)"      # >= 0.70
    MEDIUM = "🟡 WACCE (Medium)"    # 0.50 - 0.69
    LOW = "🔴 BA TABBATA (Low)"     # < 0.50


@dataclass
class RetrievalResult:
    """
    Result of a single retrieval attempt.
    
    Attributes:
        matched_pair: The matched Q&A pair (dict)
        score: Combined confidence score (0-1)
        confidence_level: Classification (HIGH/MEDIUM/LOW)
        similarity: Semantic similarity score (0-1)
        classifier_prob: Intent classification probability (0-1)
        encoder_name: Name of encoder used
        classifier_name: Name of classifier used (or fallback method)
        retrieval_method: Method used (ML_Semantic, ML_Intent_Filter, Keyword_Fallback)
        matched_question: Original matched question
        diagnostics: Dict with additional diagnostic information
    """
    matched_pair: Optional[Dict] = None
    score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    similarity: float = 0.0
    classifier_prob: float = 0.0
    encoder_name: str = "Unknown"
    classifier_name: str = "Unknown"
    retrieval_method: str = "Unknown"
    matched_question: str = ""
    diagnostics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.diagnostics is None:
            self.diagnostics = {}
    
    @property
    def is_confident(self) -> bool:
        """Return True if confidence >= 0.60 (recommended threshold)."""
        return self.score >= 0.60
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for logging/display."""
        return {
            'match': self.matched_pair,
            'score': float(self.score),
            'confidence_level': self.confidence_level.value,
            'similarity': float(self.similarity),
            'classifier_prob': float(self.classifier_prob),
            'encoder': self.encoder_name,
            'classifier': self.classifier_name,
            'method': self.retrieval_method,
            'matched_question': self.matched_question,
            'is_confident': self.is_confident,
            'diagnostics': self.diagnostics
        }


class SemanticRetriever:
    """
    Pure semantic retrieval using encoder embeddings and cosine similarity.
    
    This is the NEW preferred retrieval strategy: encode query → search corpus → 
    rank by similarity → return top match if confident.
    """
    
    def __init__(self, encoder, qa_data: Dict, encoder_name: str = "Unknown"):
        """
        Initialize semantic retriever.
        
        Args:
            encoder: Fitted encoder with transform/encode method
            qa_data: Q&A database with 'qa_pairs' key
            encoder_name: Display name of encoder
        """
        self.encoder = encoder
        self.qa_data = qa_data
        self.encoder_name = encoder_name
        self._embeddings_cache = None
        self._cache_questions = None
        logger.info(f"SemanticRetriever initialized with {encoder_name}")
    
    def _get_corpus_embeddings(self) -> np.ndarray:
        """
        Lazily compute and cache corpus embeddings.
        
        Returns:
            (n_questions, embedding_dim) embeddings matrix
        """
        if self._embeddings_cache is None:
            try:
                questions = [p.get('question', '') for p in self.qa_data.get('qa_pairs', [])]
                self._cache_questions = questions
                
                if hasattr(self.encoder, 'transform'):
                    embeddings = self.encoder.transform(questions)
                else:
                    embeddings = self.encoder.encode(questions)
                
                self._embeddings_cache = np.asarray(embeddings, dtype=np.float32)
                logger.debug(f"Cached embeddings for {len(questions)} questions. Shape: {self._embeddings_cache.shape}")
            except Exception as e:
                logger.error(f"Failed to cache embeddings: {e}")
                raise
        
        return self._embeddings_cache
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings for cosine similarity."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        return embeddings / norms
    
    def retrieve(self, query: str, top_k: int = 1, 
                 similarity_threshold: float = 0.30) -> List[RetrievalResult]:
        """
        Retrieve top-K matching Q&A pairs using semantic similarity.
        
        Args:
            query: User query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score to consider
            
        Returns:
            List of RetrievalResult objects (sorted by score)
        """
        results = []
        
        try:
            # Encode query
            if hasattr(self.encoder, 'transform'):
                query_emb = self.encoder.transform([query])
            else:
                query_emb = self.encoder.encode([query])
            
            query_emb = np.asarray(query_emb, dtype=np.float32)
            
            # Check for zero embeddings (broken encoder)
            if np.all(query_emb == 0):
                logger.warning("Zero embedding detected for query - encoder may be broken")
                return results
            
            # Normalize both query and corpus
            query_emb_norm = self._normalize_embeddings(query_emb)[0]
            corpus_emb = self._get_corpus_embeddings()
            corpus_emb_norm = self._normalize_embeddings(corpus_emb)
            
            # Compute cosine similarities
            similarities = corpus_emb_norm @ query_emb_norm
            similarities = np.asarray(similarities).flatten()
            
            # Get top-K indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            logger.debug(f"Top {top_k} similarities: {similarities[top_indices]}")
            
            # Create results
            qa_pairs = self.qa_data.get('qa_pairs', [])
            for rank, idx in enumerate(top_indices):
                sim_score = float(similarities[idx])
                
                if sim_score < similarity_threshold:
                    logger.debug(f"Skipping result {rank}: sim={sim_score} < threshold={similarity_threshold}")
                    continue
                
                pair = qa_pairs[idx]
                conf_level = self._get_confidence_level(sim_score)
                
                result = RetrievalResult(
                    matched_pair=pair,
                    score=sim_score,
                    confidence_level=conf_level,
                    similarity=sim_score,
                    classifier_prob=0.0,  # No classifier used
                    encoder_name=self.encoder_name,
                    classifier_name="None",
                    retrieval_method="Semantic_Similarity",
                    matched_question=pair.get('question', ''),
                    diagnostics={
                        'rank': rank + 1,
                        'num_corpus': len(qa_pairs),
                        'max_similarity': float(similarities.max()),
                        'mean_similarity': float(similarities.mean())
                    }
                )
                results.append(result)
            
            logger.info(f"Semantic retrieval: {len(results)} results above threshold")
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}", exc_info=True)
        
        return results
    
    @staticmethod
    def _get_confidence_level(score: float) -> ConfidenceLevel:
        """Map similarity score to confidence level."""
        if score >= 0.70:
            return ConfidenceLevel.HIGH
        elif score >= 0.50:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class IntentFilteredRetriever:
    """
    Retrieval with intent pre-filtering (optional secondary signal).
    
    This approach:
    1. Classifier predicts intent
    2. Filter Q&A corpus by matching intents
    3. Use semantic similarity within filtered set
    4. Combine signals for final score
    """
    
    def __init__(self, semantic_retriever: SemanticRetriever, 
                 classifier, classifier_name: str = "Unknown"):
        """
        Initialize intent-filtered retriever.
        
        Args:
            semantic_retriever: Configured SemanticRetriever
            classifier: Fitted classifier with predict_proba
            classifier_name: Display name
        """
        self.semantic_retriever = semantic_retriever
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.qa_data = semantic_retriever.qa_data
        logger.info(f"IntentFilteredRetriever initialized with {classifier_name}")
    
    def retrieve(self, query: str, top_k: int = 1,
                 semantic_weight: float = 0.7,
                 intent_weight: float = 0.3) -> Optional[RetrievalResult]:
        """
        Retrieve using intent filtering + semantic ranking.
        
        Args:
            query: User query
            top_k: Top-K results to return
            semantic_weight: Weight for semantic similarity (0-1)
            intent_weight: Weight for intent classifier (0-1)
            
        Returns:
            Best RetrievalResult or None if no confident match
        """
        try:
            # Get query encoding
            if hasattr(self.semantic_retriever.encoder, 'transform'):
                query_emb = self.semantic_retriever.encoder.transform([query])
            else:
                query_emb = self.semantic_retriever.encoder.encode([query])
            
            query_emb = np.asarray(query_emb, dtype=np.float32)
            
            # Predict intent
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(query_emb)[0]
                top_intent_idx = np.argmax(probs)
                top_intent_prob = float(probs[top_intent_idx])
                
                if hasattr(self.classifier, 'classes_'):
                    predicted_intent = self.classifier.classes_[top_intent_idx]
                else:
                    predicted_intent = str(top_intent_idx)
            else:
                predictions = self.classifier.predict(query_emb)
                predicted_intent = str(predictions[0])
                top_intent_prob = 0.5  # Default fallback
            
            logger.debug(f"Predicted intent: {predicted_intent} (prob={top_intent_prob:.3f})")
            
            # Filter Q&A pairs by intent
            qa_pairs = self.qa_data.get('qa_pairs', [])
            intent_matches = [
                (i, p) for i, p in enumerate(qa_pairs)
                if str(p.get('intent', '')).strip() == str(predicted_intent).strip()
            ]
            
            if not intent_matches:
                logger.debug(f"No Q&A pairs found for intent: {predicted_intent}")
                return None
            
            # Get semantic similarities for intent-matching pairs
            corpus_emb = self.semantic_retriever._get_corpus_embeddings()
            corpus_emb_norm = self.semantic_retriever._normalize_embeddings(corpus_emb)
            query_emb_norm = self.semantic_retriever._normalize_embeddings(query_emb)[0]
            
            similarities = corpus_emb_norm @ query_emb_norm
            
            # Score intent-matching pairs
            scored_pairs = []
            for orig_idx, pair in intent_matches:
                sim_score = float(similarities[orig_idx])
                combined_score = (semantic_weight * sim_score + 
                                intent_weight * top_intent_prob)
                scored_pairs.append((orig_idx, pair, sim_score, combined_score))
            
            # Sort by combined score
            scored_pairs.sort(key=lambda x: x[3], reverse=True)
            best_orig_idx, best_pair, best_sim, best_combined = scored_pairs[0]
            
            logger.info(f"Intent-filtered retrieval: combined_score={best_combined:.3f}, "
                       f"sim={best_sim:.3f}, intent_prob={top_intent_prob:.3f}")
            
            conf_level = self._get_confidence_level(best_combined)
            
            return RetrievalResult(
                matched_pair=best_pair,
                score=best_combined,
                confidence_level=conf_level,
                similarity=best_sim,
                classifier_prob=top_intent_prob,
                encoder_name=self.semantic_retriever.encoder_name,
                classifier_name=self.classifier_name,
                retrieval_method="Intent_Filtered_Semantic",
                matched_question=best_pair.get('question', ''),
                diagnostics={
                    'predicted_intent': predicted_intent,
                    'intent_prob': top_intent_prob,
                    'num_intent_matches': len(intent_matches),
                    'semantic_weight': semantic_weight,
                    'intent_weight': intent_weight
                }
            )
        
        except Exception as e:
            logger.error(f"Intent-filtered retrieval failed: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _get_confidence_level(score: float) -> ConfidenceLevel:
        """Map combined score to confidence level."""
        if score >= 0.70:
            return ConfidenceLevel.HIGH
        elif score >= 0.50:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class KeywordFallbackRetriever:
    """
    Fallback keyword-based retrieval when ML confidence is low.
    
    Uses:
    - Token overlap with questions
    - Keyword matching
    - TF-IDF weighting
    """
    
    def __init__(self, qa_data: Dict):
        """
        Initialize keyword retriever.
        
        Args:
            qa_data: Q&A database
        """
        self.qa_data = qa_data
        logger.info("KeywordFallbackRetriever initialized")
    
    def retrieve(self, query: str) -> Optional[RetrievalResult]:
        """
        Retrieve using keyword matching.
        
        Args:
            query: User query
            
        Returns:
            RetrievalResult or None
        """
        try:
            query_lower = query.lower().strip()
            if not query_lower:
                return None
            
            query_tokens = set(query_lower.split())
            qa_pairs = self.qa_data.get('qa_pairs', [])
            best_score = 0.0
            best_pair = None
            best_idx = -1
            
            for idx, pair in enumerate(qa_pairs):
                question = pair.get('question', '').lower()
                keywords_str = pair.get('keywords', '').lower()
                
                # Token overlap score
                question_tokens = set(question.split())
                overlap = len(query_tokens & question_tokens) / (len(query_tokens) + 0.001)
                
                # Phrase match (query substring in question)
                phrase_match = 1.0 if query_lower in question else 0.0
                
                # Keyword match
                keyword_score = 0.0
                if keywords_str:
                    keywords = [k.strip() for k in keywords_str.split(',')]
                    matched_keywords = sum(1 for k in keywords if k in query_lower)
                    keyword_score = matched_keywords / (len(keywords) + 0.001)
                
                # Combined score
                score = (0.5 * phrase_match + 
                        0.3 * overlap + 
                        0.2 * keyword_score)
                
                if score > best_score:
                    best_score = score
                    best_pair = pair
                    best_idx = idx
            
            logger.debug(f"Keyword retrieval best score: {best_score:.3f}")
            
            if best_score < 0.25:  # Very low threshold for fallback
                return None
            
            conf_level = self._get_confidence_level(best_score)
            
            return RetrievalResult(
                matched_pair=best_pair,
                score=best_score,
                confidence_level=conf_level,
                similarity=best_score,
                classifier_prob=0.0,
                encoder_name="Keyword",
                classifier_name="None",
                retrieval_method="Keyword_Fallback",
                matched_question=best_pair.get('question', '') if best_pair else "",
                diagnostics={'num_qa_pairs': len(qa_pairs)}
            )
        
        except Exception as e:
            logger.error(f"Keyword fallback retrieval failed: {e}")
            return None
    
    @staticmethod
    def _get_confidence_level(score: float) -> ConfidenceLevel:
        """Map keyword score to confidence."""
        if score >= 0.70:
            return ConfidenceLevel.HIGH
        elif score >= 0.50:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
