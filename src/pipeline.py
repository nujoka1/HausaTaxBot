"""
src/pipeline.py
HausaTaxBot — End-to-End Training & Inference Pipeline
COEN541 · Ahmadu Bello University, Zaria · 2025/2026

Orchestrates:
  1. Data loading & preprocessing
  2. Train/val/test split
  3. Training all encoder + classifier pairs
  4. Evaluation & comparison
  5. Model caching & persistence
  6. Inference interface (for Streamlit)

Lead: Eng. Nuhu
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

from preprocessing import HausaPreprocessor
from feature_extraction import (
    CTFIDFEncoder, ColBERTEncoder, Model2VecEncoder
)
from classifiers import SVMClassifier, FastKANClassifier
from evaluation import EvaluationFramework


class HausaTaxBotPipeline:
    """
    End-to-end pipeline for training, evaluating, and deploying HausaTaxBot.
    
    Usage:
        pipeline = HausaTaxBotPipeline("data/raw/hausa_tax_qa.json")
        pipeline.train()
        pipeline.evaluate()
        pipeline.save_models("models/")
        
        # Inference
        intent, confidence, answer = pipeline.predict("Ni hanya ce kudi a kasuwa?")
    """

    def __init__(self, corpus_path: str, cache_dir: str = ".cache"):
        """
        Parameters:
            corpus_path: path to hausa_tax_qa.json
            cache_dir: where to store fitted models
        """
        self.corpus_path = Path(corpus_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.corpus: List[Dict] = []
        self.preprocessor = HausaPreprocessor()
        self.train_data: List[Dict] = []
        self.val_data: List[Dict] = []
        self.test_data: List[Dict] = []

        # Best models from training
        self.best_encoder = None
        self.best_classifier = None
        self.best_metrics: Dict = {}

        # All trained models (for comparison)
        self.all_models: Dict = {}

    def load_corpus(self) -> int:
        """Load corpus from JSON file."""
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
        print(f"✓ Loaded {len(self.corpus)} Q&A pairs from {self.corpus_path}")
        return len(self.corpus)

    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Split corpus into train/val/test.
        train_ratio + val_ratio = (1 - test_ratio)
        """
        from sklearn.model_selection import train_test_split

        test_ratio = 1.0 - train_ratio - val_ratio
        n_train = int(len(self.corpus) * train_ratio)
        n_val = int(len(self.corpus) * val_ratio)

        # Stratified split by intent
        intents = [r.get("intent", "unknown") for r in self.corpus]

        # Train + Val vs Test
        train_val, test = train_test_split(
            self.corpus,
            test_size=test_ratio,
            random_state=42,
            stratify=intents
        )

        # Train vs Val (from the remaining)
        val_ratio_adj = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adj,
            random_state=42,
            stratify=[r.get("intent", "unknown") for r in train_val]
        )

        self.train_data = train
        self.val_data = val
        self.test_data = test

        print(f"✓ Data split: train={len(train)} | val={len(val)} | test={len(test)}")
        return len(train), len(val), len(test)

    def train_best_model(self, encoder_name: str, classifier_name: str):
        """
        Train a specific encoder-classifier pair and cache models.
        
        Usage:
            pipeline.train_best_model("Model2Vec", "FastKAN")
        """
        from evaluation import EvaluationFramework

        encoder_class = {
            "c-TF-IDF": CTFIDFEncoder,
            "ColBERT": ColBERTEncoder,
            "Model2Vec": Model2VecEncoder,
        }[encoder_name]

        classifier_class = {
            "SVM": SVMClassifier,
            "FastKAN": FastKANClassifier,
        }[classifier_name]

        # Encode training data
        print(f"\n[1] Fitting {encoder_name} encoder...")
        self.best_encoder = encoder_class()
        self.best_encoder.fit(self.train_data)

        train_X = self.best_encoder.encode(
            [r["question_hausa"] for r in self.train_data]
        )
        val_X = self.best_encoder.encode(
            [r["question_hausa"] for r in self.val_data]
        )
        test_X = self.best_encoder.encode(
            [r["question_hausa"] for r in self.test_data]
        )

        train_y = [r.get("intent", "unknown") for r in self.train_data]
        val_y = [r.get("intent", "unknown") for r in self.val_data]
        test_y = [r.get("intent", "unknown") for r in self.test_data]

        # Train classifier
        print(f"[2] Fitting {classifier_name} classifier...")
        self.best_classifier = classifier_class()
        if classifier_name == "FastKAN":
            self.best_classifier.fit(train_X, train_y, X_val=val_X, y_val=val_y)
        else:
            self.best_classifier.fit(train_X, train_y)

        # Evaluate on test set
        print(f"[3] Evaluating on test set...")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = self.best_classifier.predict(test_X)

        self.best_metrics = {
            "encoder": encoder_name,
            "classifier": classifier_name,
            "accuracy": float(accuracy_score(test_y, y_pred)),
            "precision": float(precision_score(test_y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(test_y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(test_y, y_pred, average="weighted", zero_division=0)),
            "test_samples": len(test_y),
            "timestamp": datetime.now().isoformat(),
        }

        print(f"✓ Training complete!")
        print(f"  - Accuracy:  {self.best_metrics['accuracy']:.4f}")
        print(f"  - Precision: {self.best_metrics['precision']:.4f}")
        print(f"  - Recall:    {self.best_metrics['recall']:.4f}")
        print(f"  - F1-Score:  {self.best_metrics['f1']:.4f}")

        return self.best_metrics

    def train_all_models(self) -> Dict:
        """
        Train all 8 encoder-classifier combinations and evaluate.
        """
        evaluator = EvaluationFramework(self.corpus)
        results = evaluator.evaluate_all(test_ratio=0.2)

        # Save evaluation results
        eval_dir = self.cache_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        evaluator.save_report(str(eval_dir / "results.json"))
        evaluator.generate_markdown_report(str(eval_dir / "RESULTS.md"))
        evaluator.plot_accuracy_comparison(str(eval_dir / "accuracy_comparison.png"))
        evaluator.plot_confusion_matrices(str(eval_dir / "confusion_matrices/"))

        print(f"✓ Evaluation results saved to {eval_dir}/")
        return results

    def save_models(self, output_dir: str = "models"):
        """
        Save best encoder and classifier to disk.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.best_encoder is None or self.best_classifier is None:
            print("✗ No models to save. Run train_best_model() first.")
            return

        encoder_path = output_path / "encoder.pkl"
        classifier_path = output_path / "classifier.pkl"
        metrics_path = output_path / "metrics.json"

        with open(encoder_path, "wb") as f:
            pickle.dump(self.best_encoder, f)
            print(f"✓ Saved encoder: {encoder_path}")

        with open(classifier_path, "wb") as f:
            pickle.dump(self.best_classifier, f)
            print(f"✓ Saved classifier: {classifier_path}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.best_metrics, f, indent=2)
            print(f"✓ Saved metrics: {metrics_path}")

    def load_models(self, model_dir: str = "models"):
        """
        Load previously trained encoder and classifier from disk.
        """
        model_path = Path(model_dir)

        encoder_path = model_path / "encoder.pkl"
        classifier_path = model_path / "classifier.pkl"
        metrics_path = model_path / "metrics.json"

        with open(encoder_path, "rb") as f:
            self.best_encoder = pickle.load(f)
            print(f"✓ Loaded encoder: {encoder_path}")

        with open(classifier_path, "rb") as f:
            self.best_classifier = pickle.load(f)
            print(f"✓ Loaded classifier: {classifier_path}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            self.best_metrics = json.load(f)
            print(f"✓ Loaded metrics: {metrics_path}")

    def predict(self, question: str) -> Tuple[str, float, Optional[Dict]]:
        """
        Predict intent for a question and retrieve answer.
        
        Returns:
            (intent, confidence, answer_dict) or (None, 0, None) if failed
        
        Usage:
            intent, conf, answer = pipeline.predict("Ni hanya ce kudi a kasuwa?")
        """
        if self.best_encoder is None or self.best_classifier is None:
            raise RuntimeError("Load or train models first (load_models() or train_best_model())")

        # Preprocess question
        question_clean = self.preprocessor.process(question)

        # Encode
        X = self.best_encoder.encode([question_clean])

        # Predict intent + confidence
        intent, confidence = self.best_classifier.predict_single(X[0])

        # Find matching answer from corpus (highest relevance in that intent)
        matching_records = [
            r for r in self.corpus if r.get("intent", "unknown") == intent
        ]

        if matching_records:
            # Return first matching answer (ideally rank by relevance)
            answer_record = matching_records[0]
        else:
            answer_record = None

        return intent, confidence, answer_record

    def batch_predict(self, questions: List[str]) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Predict intents for multiple questions.
        """
        return [self.predict(q) for q in questions]


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = HausaTaxBotPipeline("../data/raw/hausa_tax_qa.json")

    # Load corpus and split
    pipeline.load_corpus()
    pipeline.split_data()

    # Train best model
    print("\n" + "="*70)
    print("TRAINING BEST MODEL")
    print("="*70)
    metrics = pipeline.train_best_model("Model2Vec", "FastKAN")

    # Save
    pipeline.save_models("../models/")

    # Test inference
    print("\n" + "="*70)
    print("INFERENCE TEST")
    print("="*70)
    test_questions = [
        "Me ne kudi a kasuwa?",
        "Ta yi wadi ne kasuwa?",
    ]
    for q in test_questions:
        intent, conf, answer = pipeline.predict(q)
        print(f"Q: {q}")
        print(f"  → Intent: {intent} (confidence: {conf:.4f})")
        if answer:
            print(f"  → Answer: {answer.get('answer_hausa', 'N/A')[:80]}...")
        print()

    print("✓ Pipeline test complete!")
