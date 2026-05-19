"""
Evaluation and Model Comparison Utilities
==========================================

This module provides comprehensive evaluation metrics and model comparison
for HausaTaxBot, supporting the academic requirement for experimental analysis.

Metrics Include:
- Accuracy, Precision, Recall, F1
- Confusion matrices
- Retrieval-specific metrics (MRR, NDCG)
- Model comparison tables

Author: HausaTaxBot Research Team
Project: COEN541 - Advanced NLP
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger("HausaTaxBot.Evaluation")


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics."""
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/display."""
        return {
            'accuracy': float(self.accuracy),
            'precision_macro': float(self.precision_macro),
            'precision_weighted': float(self.precision_weighted),
            'recall_macro': float(self.recall_macro),
            'recall_weighted': float(self.recall_weighted),
            'f1_macro': float(self.f1_macro),
            'f1_weighted': float(self.f1_weighted),
            'per_class_metrics': self.per_class_metrics
        }
    
    def __str__(self) -> str:
        """Pretty-print metrics."""
        lines = [
            "=== Classification Metrics ===",
            f"Accuracy:          {self.accuracy:.4f}",
            f"Precision (macro): {self.precision_macro:.4f}",
            f"Precision (wtd):   {self.precision_weighted:.4f}",
            f"Recall (macro):    {self.recall_macro:.4f}",
            f"Recall (wtd):      {self.recall_weighted:.4f}",
            f"F1 (macro):        {self.f1_macro:.4f}",
            f"F1 (weighted):     {self.f1_weighted:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    mean_reciprocal_rank: float  # MRR
    recall_at_1: float  # R@1
    recall_at_3: float  # R@3
    recall_at_5: float  # R@5
    mean_similarity: float
    median_similarity: float
    num_queries: int
    num_correct_at_1: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mrr': float(self.mean_reciprocal_rank),
            'recall@1': float(self.recall_at_1),
            'recall@3': float(self.recall_at_3),
            'recall@5': float(self.recall_at_5),
            'mean_sim': float(self.mean_similarity),
            'median_sim': float(self.median_similarity),
            'num_queries': self.num_queries,
            'num_correct@1': self.num_correct_at_1
        }
    
    def __str__(self) -> str:
        """Pretty-print metrics."""
        lines = [
            "=== Retrieval Metrics ===",
            f"Mean Reciprocal Rank: {self.mean_reciprocal_rank:.4f}",
            f"Recall@1:             {self.recall_at_1:.4f}",
            f"Recall@3:             {self.recall_at_3:.4f}",
            f"Recall@5:             {self.recall_at_5:.4f}",
            f"Mean Similarity:      {self.mean_similarity:.4f}",
            f"Median Similarity:    {self.median_similarity:.4f}",
            f"Queries Evaluated:    {self.num_queries}",
        ]
        return "\n".join(lines)


class ClassificationEvaluator:
    """Evaluate classification performance."""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            ClassificationMetrics object
        """
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class = {}
        classes = np.unique(y_true)
        for cls in classes:
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            per_class[str(cls)] = {
                'precision': float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                'recall': float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                'f1': float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
            }
        
        return ClassificationMetrics(
            accuracy=float(acc),
            precision_macro=float(prec_macro),
            precision_weighted=float(prec_weighted),
            recall_macro=float(recall_macro),
            recall_weighted=float(recall_weighted),
            f1_macro=float(f1_macro),
            f1_weighted=float(f1_weighted),
            confusion_matrix=cm,
            per_class_metrics=per_class
        )


class RetrievalEvaluator:
    """Evaluate retrieval/ranking performance."""
    
    @staticmethod
    def evaluate(retrieval_results: List[Dict]) -> RetrievalMetrics:
        """
        Evaluate retrieval results.
        
        Expected format of each result:
        {
            'query': str,
            'correct_id': int,  # Correct/relevant item ID
            'ranking': [(item_id, score, rank), ...],  # Ranked results
            'similarities': [float, ...]  # Top-K similarities
        }
        
        Args:
            retrieval_results: List of retrieval result dicts
            
        Returns:
            RetrievalMetrics object
        """
        if not retrieval_results:
            logger.warning("No retrieval results provided")
            return RetrievalMetrics(
                mean_reciprocal_rank=0.0,
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                mean_similarity=0.0,
                median_similarity=0.0,
                num_queries=0,
                num_correct_at_1=0
            )
        
        reciprocal_ranks = []
        recalls_at_1 = []
        recalls_at_3 = []
        recalls_at_5 = []
        all_similarities = []
        num_correct_at_1 = 0
        
        for result in retrieval_results:
            correct_id = result.get('correct_id')
            ranking = result.get('ranking', [])
            sims = result.get('similarities', [])
            
            if sims:
                all_similarities.extend(sims)
            
            # Find rank of correct item
            correct_rank = None
            for idx, (item_id, score, rank) in enumerate(ranking):
                if item_id == correct_id:
                    correct_rank = rank
                    break
            
            if correct_rank is not None:
                rr = 1.0 / correct_rank
                reciprocal_ranks.append(rr)
                
                if correct_rank == 1:
                    num_correct_at_1 += 1
                    recalls_at_1.append(1.0)
                else:
                    recalls_at_1.append(0.0)
                
                recalls_at_3.append(1.0 if correct_rank <= 3 else 0.0)
                recalls_at_5.append(1.0 if correct_rank <= 5 else 0.0)
            else:
                # Correct item not in ranking
                reciprocal_ranks.append(0.0)
                recalls_at_1.append(0.0)
                recalls_at_3.append(0.0)
                recalls_at_5.append(0.0)
        
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        r@1 = np.mean(recalls_at_1) if recalls_at_1 else 0.0
        r@3 = np.mean(recalls_at_3) if recalls_at_3 else 0.0
        r@5 = np.mean(recalls_at_5) if recalls_at_5 else 0.0
        
        mean_sim = np.mean(all_similarities) if all_similarities else 0.0
        median_sim = np.median(all_similarities) if all_similarities else 0.0
        
        return RetrievalMetrics(
            mean_reciprocal_rank=float(mrr),
            recall_at_1=float(r@1),
            recall_at_3=float(r@3),
            recall_at_5=float(r@5),
            mean_similarity=float(mean_sim),
            median_similarity=float(median_sim),
            num_queries=len(retrieval_results),
            num_correct_at_1=int(num_correct_at_1)
        )


class ModelComparator:
    """Compare multiple models on same evaluation set."""
    
    @staticmethod
    def compare_classifiers(models: Dict[str, object],
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> Dict[str, ClassificationMetrics]:
        """
        Compare multiple classifiers.
        
        Args:
            models: Dict of {model_name: model_instance}
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict of {model_name: ClassificationMetrics}
        """
        results = {}
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                metrics = ClassificationEvaluator.evaluate(y_test, y_pred)
                results[model_name] = metrics
                logger.info(f"Evaluated {model_name}: acc={metrics.accuracy:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return results
    
    @staticmethod
    def create_comparison_table(results: Dict[str, ClassificationMetrics]) -> str:
        """
        Create formatted comparison table.
        
        Args:
            results: Dict of {model_name: metrics}
            
        Returns:
            Formatted table string
        """
        lines = ["=" * 100]
        lines.append("MODEL COMPARISON TABLE")
        lines.append("=" * 100)
        
        # Header
        header = f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 (Macro)':<12}"
        lines.append(header)
        lines.append("-" * 100)
        
        # Rows
        for model_name, metrics in results.items():
            row = (f"{model_name:<20} "
                  f"{metrics.accuracy:<12.4f} "
                  f"{metrics.precision_macro:<12.4f} "
                  f"{metrics.recall_macro:<12.4f} "
                  f"{metrics.f1_macro:<12.4f}")
            lines.append(row)
        
        lines.append("=" * 100)
        return "\n".join(lines)
    
    @staticmethod
    def find_best_model(results: Dict[str, ClassificationMetrics],
                       metric: str = 'f1_macro') -> Tuple[str, float]:
        """
        Find best model by specified metric.
        
        Args:
            results: Dict of {model_name: metrics}
            metric: Metric to compare ('accuracy', 'f1_macro', 'f1_weighted', etc.)
            
        Returns:
            Tuple of (best_model_name, metric_value)
        """
        best_model = None
        best_value = 0.0
        
        for model_name, metrics in results.items():
            value = getattr(metrics, metric, 0.0)
            if value > best_value:
                best_value = value
                best_model = model_name
        
        return best_model, best_value
