"""
src/classifiers.py
HausaTaxBot — Intent Classifiers: SVM & FastKAN
COEN541 · Ahmadu Bello University, Zaria · 2025/2026

Two classifiers, same interface:
  classifier.fit(X_train, y_train)
  classifier.predict(X)            → List[str]  (intent labels)
  classifier.predict_proba(X)      → np.ndarray  (confidence scores)

SVM    = classical baseline   (replicates related papers)
FastKAN = our novel improvement (KAN networks — new architecture)

Lead: Eng. Amanda
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# ════════════════════════════════════════════════════════════
# 1. SVM Classifier (Baseline)
# ════════════════════════════════════════════════════════════

class SVMClassifier:
    """
    Support Vector Machine intent classifier.

    Configuration:
      - Kernel: RBF (best for dense embedding inputs)
      - C: 1.0 (regularisation — tuned via grid search)
      - Probability: True (so we get confidence scores)
      - Scaler: StandardScaler (important for SVM performance)

    Input X: embedding vectors from any encoder (c-TF-IDF,
             Model2Vec, ColBERT) shaped [N, D]
    Input y: intent label strings e.g. "vat", "pit"

    Why SVM for chatbot intent classification:
      SVM finds the maximum-margin hyperplane between intent
      classes. With good embeddings, this is very effective
      even on small datasets (our 360-pair corpus).
    """

    def __init__(self, C: float = 1.0, kernel: str = "rbf",
                 gamma: str = "scale"):
        self.label_encoder = LabelEncoder()
        self.pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("svm", SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                random_state=42,
                class_weight="balanced",   # handles class imbalance
            )),
        ])
        self.classes_: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: List[str]) -> "SVMClassifier":
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = list(self.label_encoder.classes_)
        self.pipeline.fit(X, y_enc)
        self.is_fitted = True
        print(f"[SVM] Fitted on {len(y)} samples, "
              f"{len(self.classes_)} classes: {self.classes_}")
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        y_enc = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_enc).tolist()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_single(self, x: np.ndarray) -> Tuple[str, float]:
        """Return (intent, confidence) for a single vector."""
        x = x.reshape(1, -1)
        probs   = self.predict_proba(x)[0]
        best    = int(np.argmax(probs))
        return self.classes_[best], float(probs[best])

    def save(self, path: str):
        joblib.dump({"pipeline": self.pipeline,
                     "label_encoder": self.label_encoder,
                     "classes": self.classes_}, path)
        print(f"[SVM] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SVMClassifier":
        obj  = cls()
        data = joblib.load(path)
        obj.pipeline      = data["pipeline"]
        obj.label_encoder = data["label_encoder"]
        obj.classes_      = data["classes"]
        obj.is_fitted     = True
        return obj


# ════════════════════════════════════════════════════════════
# 2. FastKAN Classifier (Our Novel Improvement)
#
#    Kolmogorov-Arnold Networks (KAN) replace the fixed
#    activation functions of standard MLP neurons with
#    learnable univariate spline functions on each edge.
#
#    FastKAN = efficient GPU/CPU implementation of KAN
#    that uses Gaussian radial basis functions instead of
#    expensive B-splines, making it practical for small
#    datasets like ours.
#
#    Why FastKAN beats SVM on our task:
#      - KANs learn the activation shape — better at
#        capturing non-linear intent boundaries
#      - Interpretable: each edge's function can be plotted
#        (good for showing in the report/presentation)
#      - Novel: no prior Hausa chatbot paper uses KANs
# ════════════════════════════════════════════════════════════

class FastKANClassifier:
    """
    FastKAN-based intent classifier.

    Architecture:
      Input [D] → KAN Layer(D → 128) → KAN Layer(128 → 64) → Output [n_intents]

    Each KAN layer uses Radial Basis Function (RBF) activations
    with G=5 grid points — learnable per edge.

    Training:
      - Optimizer: Adam (lr=1e-3)
      - Loss: CrossEntropyLoss
      - Epochs: 50 (with early stopping)
      - Batch size: 16
    """

    def __init__(self, hidden_dim: int = 128, grid_size: int = 5,
                 lr: float = 1e-3, epochs: int = 50,
                 batch_size: int = 16):
        self.hidden_dim = hidden_dim
        self.grid_size  = grid_size
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.scaler        = StandardScaler(with_mean=False)
        self.model         = None
        self.classes_: List[str] = []
        self.is_fitted    = False
        self.train_history: List[Dict] = []

    def _build_model(self, input_dim: int, n_classes: int):
        """Build the FastKAN model dynamically based on input dimensions."""
        try:
            import torch
            import torch.nn as nn
            from fastkan import FastKAN
        except ImportError:
            raise ImportError(
                "Install FastKAN: pip install fastkan torch"
            )

        # FastKAN layer dimensions: input → hidden → n_classes
        layers_hidden = [input_dim, self.hidden_dim, 64, n_classes]

        model = FastKAN(
            layers_hidden=layers_hidden,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=self.grid_size,
            use_base_update=True,      # adds residual skip connection
            base_activation=torch.nn.functional.silu,
        )
        return model

    def fit(self, X: np.ndarray, y: List[str],
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[List[str]] = None) -> "FastKANClassifier":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("Install torch: pip install torch")

        # Encode labels
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = list(self.label_encoder.classes_)
        n_classes  = len(self.classes_)

        # Scale features
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        # Build model
        self.model = self._build_model(X.shape[1], n_classes)
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # DataLoader
        X_t = torch.tensor(X_scaled).to(device)
        y_t = torch.tensor(y_enc, dtype=torch.long).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5
        )

        best_val_acc = 0.0
        best_state   = None

        print(f"[FastKAN] Training on {len(X)} samples | "
              f"device={device} | epochs={self.epochs}")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)

            # Validation check
            val_acc = 0.0
            if X_val is not None and y_val is not None:
                val_acc = self._eval_acc(X_val, y_val, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state   = {k: v.clone()
                                    for k, v in self.model.state_dict().items()}

            self.train_history.append(
                {"epoch": epoch, "loss": avg_loss, "val_acc": val_acc}
            )

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:3d}/{self.epochs} | "
                      f"loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        # Restore best checkpoint
        if best_state:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        print(f"[FastKAN] Training complete. Best val acc: {best_val_acc:.4f}")
        return self

    def _eval_acc(self, X: np.ndarray, y: List[str], device) -> float:
        import torch
        self.model.eval()
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X_scaled).to(device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds  = logits.argmax(dim=1).cpu().numpy()
        y_enc = self.label_encoder.transform(y)
        return float((preds == y_enc).mean())

    def predict(self, X: np.ndarray) -> List[str]:
        import torch
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first")
        self.model.eval()
        device   = next(self.model.parameters()).device
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_t      = torch.tensor(X_scaled).to(device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds  = logits.argmax(dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(preds).tolist()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self.model.eval()
        device   = next(self.model.parameters()).device
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_t      = torch.tensor(X_scaled).to(device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_single(self, x: np.ndarray) -> Tuple[str, float]:
        x      = x.reshape(1, -1)
        probs  = self.predict_proba(x)[0]
        best   = int(np.argmax(probs))
        return self.classes_[best], float(probs[best])

    def save(self, path: str):
        import torch
        joblib.dump({
            "model_state": self.model.state_dict(),
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "classes": self.classes_,
            "config": {
                "hidden_dim": self.hidden_dim,
                "grid_size":  self.grid_size,
                "input_dim":  self.model.layers[0].inputdim
                              if hasattr(self.model, 'layers') else None,
                "n_classes":  len(self.classes_),
            },
            "history": self.train_history,
        }, path)
        print(f"[FastKAN] Saved to {path}")


# ════════════════════════════════════════════════════════════
# Grid-search helpers (for Abdullateef's evaluation)
# ════════════════════════════════════════════════════════════

def svm_grid_search(X_train, y_train, X_val, y_val) -> Dict:
    """Find best SVM hyperparameters on validation set."""
    from sklearn.metrics import accuracy_score

    best = {"acc": 0.0, "params": {}}
    for C in [0.1, 1.0, 10.0]:
        for kernel in ["rbf", "linear"]:
            clf = SVMClassifier(C=C, kernel=kernel)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            acc   = accuracy_score(y_val, preds)
            if acc > best["acc"]:
                best = {"acc": acc, "params": {"C": C, "kernel": kernel}}

    print(f"[SVM Grid Search] Best: acc={best['acc']:.4f} | params={best['params']}")
    return best


# ════════════════════════════════════════════════════════════
# Quick smoke test
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, ".")

    from feature_extraction import CTFIDFEncoder

    with open("../data/raw/hausa_tax_qa_sample.json", encoding="utf-8") as f:
        corpus = json.load(f)

    # Encode
    enc   = CTFIDFEncoder().fit(corpus)
    X_all = enc.encode([r["question_hausa"] for r in corpus])
    y_all = [r["intent"] for r in corpus]

    print("=" * 60)
    print("CLASSIFIERS — SMOKE TEST")
    print("=" * 60)
    print(f"Feature matrix: {X_all.shape}")

    # SVM
    print("\n[SVM]")
    svm = SVMClassifier()
    svm.fit(X_all, y_all)
    preds = svm.predict(X_all[:3])
    for i, (p, true) in enumerate(zip(preds, y_all[:3])):
        print(f"  Sample {i+1}: pred={p} | true={true} | match={p==true}")

    # FastKAN
    print("\n[FastKAN]")
    try:
        kan = FastKANClassifier(epochs=5)
        kan.fit(X_all, y_all)
        preds = kan.predict(X_all[:3])
        for i, (p, true) in enumerate(zip(preds, y_all[:3])):
            print(f"  Sample {i+1}: pred={p} | true={true} | match={p==true}")
    except ImportError as e:
        print(f"  → {e}")
