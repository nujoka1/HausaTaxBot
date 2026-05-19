"""
src/evaluation.py
HausaTaxBot — Model Evaluation & Comparison Framework
COEN541 · Ahmadu Bello University, Zaria · 2025/2026

Comprehensive evaluation suite for comparing:
  - 3 encoders: c-TF-IDF, ColBERT, Model2Vec + retrieval
  - 2 classifiers: SVM, FastKAN
  = 6 configurations evaluated

Metrics from scikit-learn + custom retrieval metrics:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
  - Per-intent metrics
  - Retrieval MRR, NDCG, MAP (for retrieval task)

Output: JSON report + visualization plots

Lead: Eng. Abdullateef (QA)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationFramework:
    """
    Unified evaluation pipeline for all encoder + classifier pairs.
    
    Usage:
        eval_fw = EvaluationFramework(corpus)
        results = eval_fw.evaluate_all(test_ratio=0.2)
        eval_fw.save_report("results.json")
        eval_fw.plot_confusion_matrices("plots/")
    """

    def __init__(self, corpus: List[Dict], random_state: int = 42):
        """
        Parameters:
            corpus: list of dicts with keys:
                    "id", "question_hausa", "intent", ...
            random_state: for train/test split reproducibility
        """
        self.corpus = corpus
        self.random_state = random_state
        self.results: Dict = {
            "timestamp": datetime.now().isoformat(),
            "corpus_size": len(corpus),
            "models": {},
            "summary": {},
        }

    def split_data(self, test_ratio: float = 0.2, val_ratio: float = 0.1):
        """
        Split corpus into train/val/test.
        
        Returns:
            (train_data, val_data, test_data)
        """
        # First split: 80% train, 20% test
        train_val, test = train_test_split(
            self.corpus,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=[r.get("intent", "unknown") for r in self.corpus]
        )

        # Second split: 90% train, 10% val (from the 80%)
        val_ratio_scaled = val_ratio / (1 - test_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_scaled,
            random_state=self.random_state,
            stratify=[r.get("intent", "unknown") for r in train_val]
        )

        return train, val, test

    def evaluate_encoder_classifier_pair(
        self,
        encoder_name: str,
        encoder,
        classifier_name: str,
        classifier_class,
        train_X, train_y,
        val_X, val_y,
        test_X, test_y,
    ) -> Dict:
        """
        Train and evaluate one encoder + classifier combination.
        
        Returns:
            metrics_dict with accuracy, precision, recall, f1, confusion_matrix etc.
        """
        try:
            # Fit encoder
            encoder.fit(
                [{"question_hausa": q, "intent": i}
                 for q, i in zip(train_X, train_y)]
            )
            X_train_enc = encoder.encode(train_X)
            X_val_enc = encoder.encode(val_X)
            X_test_enc = encoder.encode(test_X)

            # Fit classifier
            if classifier_name == "FastKAN":
                # FastKAN needs validation data
                classifier = classifier_class()
                classifier.fit(X_train_enc, train_y,
                               X_val=X_val_enc, y_val=val_y)
            else:
                # SVM doesn't need separate validation
                classifier = classifier_class()
                classifier.fit(X_train_enc, train_y)

            # Predict
            y_pred = classifier.predict(X_test_enc)
            y_proba = classifier.predict_proba(X_test_enc)

            # Compute metrics
            acc = accuracy_score(test_y, y_pred)
            prec = precision_score(test_y, y_pred, average='weighted', zero_division=0)
            rec = recall_score(test_y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(test_y, y_pred, average='weighted', zero_division=0)

            cm = confusion_matrix(test_y, y_pred)
            clf_report = classification_report(test_y, y_pred, output_dict=True)

            metrics = {
                "encoder": encoder_name,
                "classifier": classifier_name,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "confusion_matrix": cm.tolist(),
                "classification_report": clf_report,
                "test_samples": len(test_y),
                "status": "success",
            }

            print(f"✓ {encoder_name:15s} + {classifier_name:10s} | "
                  f"Acc={acc:.4f} | F1={f1:.4f}")

            return metrics

        except Exception as e:
            print(f"✗ {encoder_name:15s} + {classifier_name:10s} | ERROR: {e}")
            return {
                "encoder": encoder_name,
                "classifier": classifier_name,
                "status": "failed",
                "error": str(e),
            }

    def evaluate_all(self, test_ratio: float = 0.2) -> Dict:
        """
        Evaluate all 6 combinations (3 encoders × 2 classifiers).
        """
        from feature_extraction import (
            CTFIDFEncoder, ColBERTEncoder, Model2VecEncoder
        )
        from classifiers import SVMClassifier, FastKANClassifier

        # Split data
        train, val, test = self.split_data(test_ratio)

        train_X = [r["question_hausa"] for r in train]
        train_y = [r.get("intent", "unknown") for r in train]
        val_X = [r["question_hausa"] for r in val]
        val_y = [r.get("intent", "unknown") for r in val]
        test_X = [r["question_hausa"] for r in test]
        test_y = [r.get("intent", "unknown") for r in test]

        encoders = {
            "c-TF-IDF": CTFIDFEncoder,
            "ColBERT": ColBERTEncoder,
            "Model2Vec": Model2VecEncoder,
        }

        classifiers = {
            "SVM": SVMClassifier,
            "FastKAN": FastKANClassifier,
        }

        print("\n" + "="*70)
        print("EVALUATING ALL ENCODER + CLASSIFIER COMBINATIONS")
        print("="*70)
        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}\n")

        all_results = []

        for enc_name, enc_class in encoders.items():
            for clf_name, clf_class in classifiers.items():
                encoder = enc_class()
                metrics = self.evaluate_encoder_classifier_pair(
                    encoder_name=enc_name,
                    encoder=encoder,
                    classifier_name=clf_name,
                    classifier_class=clf_class,
                    train_X=train_X, train_y=train_y,
                    val_X=val_X, val_y=val_y,
                    test_X=test_X, test_y=test_y,
                )
                all_results.append(metrics)

        # Update results
        self.results["models"] = all_results

        # Compute summary statistics
        successful = [r for r in all_results if r["status"] == "success"]
        if successful:
            accs = [r["accuracy"] for r in successful]
            self.results["summary"] = {
                "best_model": max(successful, key=lambda x: x["f1"]),
                "best_accuracy": float(max(accs)),
                "worst_accuracy": float(min(accs)),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "total_combinations": len(all_results),
                "successful_combinations": len(successful),
            }

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)

        return self.results

    def save_report(self, output_path: str):
        """Save results as JSON report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ Report saved: {output_path}")

    def plot_confusion_matrices(self, output_dir: str = "evaluation/plots"):
        """
        Plot confusion matrices for all successful models.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fig_count = 0
        for model_result in self.results["models"]:
            if model_result["status"] != "success":
                continue

            enc = model_result["encoder"]
            clf = model_result["classifier"]
            cm = np.array(model_result["confusion_matrix"])

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"{enc} + {clf}\nAccuracy: {model_result['accuracy']:.4f}")
            ax.set_ylabel("True Intent")
            ax.set_xlabel("Predicted Intent")

            fig_path = Path(output_dir) / f"{fig_count:02d}_{enc}_{clf}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            fig_count += 1

        print(f"✓ Saved {fig_count} confusion matrix plots to {output_dir}/")

    def plot_accuracy_comparison(self, output_path: str = "evaluation/accuracy_comparison.png"):
        """
        Bar plot comparing accuracy of all models.
        """
        successful = [r for r in self.results["models"] if r["status"] == "success"]
        if not successful:
            print("No successful models to plot.")
            return

        labels = [f"{r['encoder']}\n{r['classifier']}" for r in successful]
        accuracies = [r["accuracy"] for r in successful]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(labels)), accuracies, color="steelblue")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison: Accuracy Across Encoder-Classifier Pairs")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim([0, 1.0])

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}',
                    ha='center', va='bottom')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved accuracy comparison plot: {output_path}")

    def generate_markdown_report(self, output_path: str = "evaluation/RESULTS.md"):
        """Generate a markdown report summarizing evaluation results."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        successful = [r for r in self.results["models"] if r["status"] == "success"]
        successful = sorted(successful, key=lambda x: x["accuracy"], reverse=True)

        md = []
        md.append("# HausaTaxBot — Evaluation Results")
        md.append(f"\n**Timestamp:** {self.results['timestamp']}")
        md.append(f"**Corpus Size:** {self.results['corpus_size']} Q&A pairs")
        md.append(f"\n## Summary Statistics")
        summary = self.results.get("summary", {})
        if summary:
            md.append(f"- **Best Accuracy:** {summary.get('best_accuracy', 0):.4f}")
            md.append(f"- **Mean Accuracy:** {summary.get('mean_accuracy', 0):.4f}")
            md.append(f"- **Std Dev:** {summary.get('std_accuracy', 0):.4f}")

        md.append(f"\n## Detailed Results (Ranked by Accuracy)\n")
        md.append("| Rank | Encoder | Classifier | Accuracy | Precision | Recall | F1-Score |")
        md.append("|------|---------|------------|----------|-----------|--------|----------|")

        for rank, result in enumerate(successful, 1):
            md.append(f"| {rank} | {result['encoder']:12s} | "
                      f"{result['classifier']:10s} | "
                      f"{result['accuracy']:.4f} | "
                      f"{result['precision']:.4f} | "
                      f"{result['recall']:.4f} | "
                      f"{result['f1']:.4f} |")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        print(f"✓ Markdown report saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, ".")

    # Load sample corpus
    with open("../data/raw/hausa_tax_qa_sample.json", encoding="utf-8") as f:
        corpus = json.load(f)

    # Run evaluation
    eval_fw = EvaluationFramework(corpus)
    results = eval_fw.evaluate_all(test_ratio=0.2)

    # Save reports
    eval_fw.save_report(f"../evaluation/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    eval_fw.generate_markdown_report()
    eval_fw.plot_accuracy_comparison()
    eval_fw.plot_confusion_matrices()

    print("\n✓ Evaluation complete!")
