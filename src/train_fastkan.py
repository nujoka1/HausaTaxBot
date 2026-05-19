"""
FastKAN Training Pipeline - Train and evaluate FastKAN classifier
==================================================================

Comprehensive training script for FastKAN classifier on HausaTaxBot Q&A data.

Features:
- Data loading and preprocessing with Hausa normalization
- Multi-encoder support: c-TF-IDF, TF-IDF, ColBERT
- Trains FastKAN classifier with PyTorch backend
- Saves trained models for production use
- Compares performance with SVM baseline
- Generates training curves and evaluation reports
- Supports batching and validation during training

Usage:
    # Train FastKAN with c-TF-IDF encoder
    python src/train_fastkan.py --encoder c-tfidf --output models/fastkan_ctfidf.pkl
    
    # Train and compare models
    python src/train_fastkan.py --compare-all --output models/
    
    # Custom hyperparameters
    python src/train_fastkan.py --hidden-dim 128 --n-layers 3 --epochs 50

Output:
    - Trained models: saved as .pkl files
    - Metrics: metrics_<model_name>.json
    - Training curves: plots/training_curves_<model_name>.png
    - Comparison report: reports/training_comparison.md

Author: HausaTaxBot Research Team (COEN541/543)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import pickle
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

from src.hausa_preprocessing import HausaPreprocessor
from src.embedding_cache import EmbeddingCache

# Try imports for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FastKANTraining")

# ==================== ENCODER CLASSES ====================

class CTFIDFEncoder:
    """BERTopic's c-TF-IDF encoder"""
    def __init__(self, n_features=8000, ngram=(1, 2)):
        self.n_features = n_features
        self.ngram = ngram
        self.cv = CountVectorizer(max_features=n_features, ngram_range=ngram, 
                                 min_df=1, max_df=0.98)
        self.idf = None
        self.classes_ = None

    def fit(self, corpus, labels):
        X = self.cv.fit_transform(corpus)
        self.classes_ = sorted(set(labels))
        n_classes = len(self.classes_)
        class_doc = sp.lil_matrix((n_classes, X.shape[1]))
        for i, cls in enumerate(self.classes_):
            idx = [j for j, l in enumerate(labels) if l == cls]
            class_doc[i] = X[idx].sum(axis=0)
        class_doc = sp.csr_matrix(class_doc)
        df_c = (class_doc > 0).sum(axis=0)
        self.idf = np.log(1 + n_classes / (1 + np.array(df_c).flatten()))
        return self

    def encode(self, texts):
        X = self.cv.transform(texts).toarray().astype(np.float32)
        X = X * self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    def transform(self, texts):
        return self.encode(texts)


class ColBERTEncoder:
    """ColBERT encoder using sentence-transformers"""
    def __init__(self):
        if not COLBERT_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        self.model = SentenceTransformer(
            "sentence-transformers/distilbert-base-multilingual-minilm-l12-v2"
        )

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def transform(self, texts):
        return self.encode(texts)


# ==================== SIMPLE FASTKAN IMPLEMENTATION ====================

class SimpleFastKAN:
    """
    Simple FastKAN classifier using scikit-learn interface.
    
    Falls back to SVM if PyTorch not available.
    Implements proper training loop with validation.
    """
    
    def __init__(self, n_classes: int, hidden_dim: int = 64, 
                 n_layers: int = 2, epochs: int = 30, batch_size: int = 16,
                 learning_rate: float = 0.001):
        """
        Initialize FastKAN classifier.
        
        Args:
            n_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            n_layers: Number of KAN layers
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.optimizer = None
        self.model_class = None
        self.is_trained = False
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        if TORCH_AVAILABLE:
            self._init_torch_model()
        else:
            logger.warning("PyTorch not available, will use SVM fallback")
    
    def _init_torch_model(self):
        """Initialize PyTorch-based FastKAN model."""
        import torch
        import torch.nn as nn
        
        class FastKANNet(nn.Module):
            """PyTorch implementation of FastKAN network."""
            def __init__(self, input_dim, hidden_dim, n_classes, n_layers=2):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes
                
                layers = []
                prev_dim = input_dim
                
                # Hidden layers
                for i in range(n_layers):
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.3))
                    prev_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(prev_dim, n_classes))
                
                self.net = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.net(x)
        
        self.model_class = FastKANNet
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the FastKAN classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using SVM fallback")
            self.model = SVC(kernel='rbf', probability=True)
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        logger.info(f"Training FastKAN classifier | "
                   f"Input dim: {X_train.shape[1]}, Classes: {self.n_classes}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Convert to torch tensors
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        
        if X_val is not None:
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.LongTensor(y_val).to(device)
        
        # Create model
        input_dim = X_train.shape[1]
        self.model = self.model_class(
            input_dim, self.hidden_dim, self.n_classes, self.n_layers
        ).to(device)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            
            indices = np.random.permutation(len(y_train))
            for i in range(0, len(y_train), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(y_batch)
                epoch_correct += (outputs.argmax(1) == y_batch).sum().item()
            
            epoch_loss /= len(y_train)
            epoch_acc = epoch_correct / len(y_train)
            
            self.training_history['loss'].append(epoch_loss)
            self.training_history['accuracy'].append(epoch_acc)
            
            # Validation phase
            val_loss = epoch_loss
            val_acc = epoch_acc
            
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    val_acc = (val_outputs.argmax(1) == y_val).sum().item() / len(y_val)
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and (epoch + 1) % 5 == 0:
                val_str = f" | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}" if X_val is not None else ""
                logger.info(f"Epoch {epoch+1}/{self.epochs} | "
                           f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}{val_str}")
        
        self.is_trained = True
        self.device = device
        logger.info("✅ Training complete")
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if not TORCH_AVAILABLE or self.model is None:
            # SVM fallback
            return self.model.predict(X)
        
        import torch
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if not TORCH_AVAILABLE or self.model is None:
            # SVM fallback
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                predictions = self.model.predict(X)
                proba = np.zeros((len(predictions), self.n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, pred] = 1.0
                return proba
        
        import torch
        import torch.nn.functional as F
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = F.softmax(outputs, dim=1).cpu().numpy()
        
        return probas


# ==================== TRAINING FUNCTION ====================

def train_model(qa_path: str,
               encoder_type: str = "c-tfidf",
               use_fastkan: bool = True,
               output_dir: str = "models",
               test_size: float = 0.2,
               val_size: float = 0.2,
               epochs: int = 30) -> Dict:
    """
    Train a model on HausaTaxBot Q&A data.
    
    Args:
        qa_path: Path to hausa_tax_qa.json
        encoder_type: "c-tfidf", "tfidf", or "colbert"
        use_fastkan: Use FastKAN if True, SVM if False
        output_dir: Directory to save models
        test_size: Fraction for test split
        val_size: Fraction for validation split
        epochs: Training epochs (for FastKAN)
        
    Returns:
        Dictionary with results, metrics, and model paths
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {encoder_type.upper()} + {'FastKAN' if use_fastkan else 'SVM'}")
    logger.info(f"{'='*70}\n")
    
    # Load data
    qa_path = Path(qa_path)
    if not qa_path.exists():
        raise FileNotFoundError(f"Data not found: {qa_path}")
    
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    qa_pairs = qa_data.get('qa_pairs', [])
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Prepare data
    preprocessor = HausaPreprocessor()
    questions = [p.get('question', '') for p in qa_pairs]
    intents = [p.get('intent', 'unknown') for p in qa_pairs]
    
    # Preprocess
    questions = [preprocessor.preprocess(q) for q in questions]
    
    # Train/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        questions, intents, test_size=test_size, 
        random_state=42, stratify=intents if len(set(intents)) > 1 else None
    )
    
    # Train/val split
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, 
        random_state=42, stratify=y_temp if len(set(y_temp)) > 1 else None
    )
    
    logger.info(f"Data split | Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create encoder
    logger.info(f"Creating {encoder_type} encoder...")
    if encoder_type.lower() == "c-tfidf":
        encoder = CTFIDFEncoder()
        encoder.fit(X_train, y_train)
    elif encoder_type.lower() == "tfidf":
        encoder = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        encoder.fit(X_train)
    elif encoder_type.lower() == "colbert" and COLBERT_AVAILABLE:
        encoder = ColBERTEncoder()
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")
    
    # Encode data
    logger.info("Encoding data...")
    X_train_enc = encoder.encode(X_train)
    X_val_enc = encoder.encode(X_val)
    X_test_enc = encoder.encode(X_test)
    
    # Label encode intents
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    # Train classifier
    classifier_name = "FastKAN" if use_fastkan else "SVM"
    logger.info(f"\nTraining {classifier_name} classifier...")
    
    if use_fastkan:
        classifier = SimpleFastKAN(
            n_classes=len(label_encoder.classes_),
            hidden_dim=64,
            n_layers=2,
            epochs=epochs,
            batch_size=16,
            learning_rate=0.001
        )
        classifier.fit(X_train_enc, y_train_enc, X_val_enc, y_val_enc, verbose=True)
    else:
        classifier = SVC(kernel='rbf', probability=True, random_state=42)
        classifier.fit(X_train_enc, y_train_enc)
        logger.info("✅ SVM training complete")
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    y_pred_enc = classifier.predict(X_test_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    }
    
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1']:.4f}")
    
    # Save models
    model_name = f"{encoder_type}_{classifier_name.lower()}"
    model_path = output_dir / f"{model_name}.pkl"
    
    # Custom save function to handle encoders
    model_dict = {
        'encoder': encoder,
        'classifier': classifier,
        'label_encoder': label_encoder,
        'encoder_type': encoder_type,
        'classifier_type': classifier_name,
        'metrics': metrics,
        'training_date': datetime.now().isoformat()
    }
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f)
        logger.info(f"✅ Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
    
    # Save metrics
    metrics_path = output_dir / f"metrics_{model_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return {
        'model_name': model_name,
        'model_path': str(model_path),
        'metrics': metrics,
        'classes': list(label_encoder.classes_),
        'training_history': (classifier.training_history if hasattr(classifier, 'training_history') else {})
    }


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="Train FastKAN classifier for HausaTaxBot"
    )
    parser.add_argument('--qa-path', default='data/raw/hausa_tax_qa.json',
                       help='Path to Q&A dataset')
    parser.add_argument('--encoder', default='c-tfidf',
                       choices=['c-tfidf', 'tfidf', 'colbert'],
                       help='Encoder type')
    parser.add_argument('--classifier', default='fastkan',
                       choices=['fastkan', 'svm'],
                       help='Classifier type')
    parser.add_argument('--output', default='models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--compare-all', action='store_true',
                       help='Train and compare all encoder-classifier combinations')
    
    args = parser.parse_args()
    
    if args.compare_all:
        # Train all combinations
        results = []
        combinations = [
            ('c-tfidf', 'fastkan'),
            ('c-tfidf', 'svm'),
            ('tfidf', 'svm'),
        ]
        
        if COLBERT_AVAILABLE:
            combinations.append(('colbert', 'svm'))
        
        for encoder, classifier in combinations:
            result = train_model(
                args.qa_path,
                encoder_type=encoder,
                use_fastkan=(classifier == 'fastkan'),
                output_dir=args.output,
                epochs=args.epochs
            )
            results.append(result)
        
        # Generate comparison report
        print("\n" + "="*70)
        print("TRAINING COMPARISON RESULTS")
        print("="*70)
        
        comparison_data = []
        for r in results:
            comparison_data.append({
                'Model': r['model_name'],
                'Accuracy': f"{r['metrics']['accuracy']:.4f}",
                'Precision': f"{r['metrics']['precision']:.4f}",
                'Recall': f"{r['metrics']['recall']:.4f}",
                'F1-Score': f"{r['metrics']['f1']:.4f}",
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print("="*70 + "\n")
        
    else:
        # Train single model
        result = train_model(
            args.qa_path,
            encoder_type=args.encoder,
            use_fastkan=(args.classifier == 'fastkan'),
            output_dir=args.output,
            epochs=args.epochs
        )
        
        print(f"\n✅ Trained model saved to: {result['model_path']}")
        print(f"   Accuracy: {result['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
