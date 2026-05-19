## 📚 FastKAN Training Pipeline - Complete Guide

### What's Included

**File:** [src/train_fastkan.py](src/train_fastkan.py) (600+ lines)

A comprehensive training pipeline for FastKAN classifier with multiple encoder support:

- **SimpleFastKAN Class**: PyTorch-based implementation with proper training loop
- **Multi-Encoder Support**: c-TF-IDF, TF-IDF, ColBERT
- **Hausa Preprocessing**: Automatic text normalization in data pipeline
- **Training Features**:
  - Stratified train/val/test splits
  - Batch processing with gradient descent
  - Learning rate scheduling
  - Early stopping on validation loss
  - Training history tracking
  - SVM fallback (if PyTorch unavailable)

---

### Quick Start

#### 1️⃣ Train FastKAN with c-TF-IDF

```bash
cd HausaTaxBot
source ~/yoloenv/bin/activate

# Train single model
python src/train_fastkan.py \
    --qa-path data/raw/hausa_tax_qa.json \
    --encoder c-tfidf \
    --classifier fastkan \
    --output models/ \
    --epochs 30
```

#### 2️⃣ Train SVM Baseline for Comparison

```bash
python src/train_fastkan.py \
    --qa-path data/raw/hausa_tax_qa.json \
    --encoder c-tfidf \
    --classifier svm \
    --output models/
```

#### 3️⃣ Train All Models (Full Comparison)

```bash
python src/train_fastkan.py \
    --qa-path data/raw/hausa_tax_qa.json \
    --compare-all \
    --output models/
```

This trains:
- c-TF-IDF + FastKAN
- c-TF-IDF + SVM
- TF-IDF + SVM
- ColBERT + SVM (if available)

---

### Output Files

| File | Purpose |
|------|---------|
| `models/c-tfidf_fastkan.pkl` | Trained FastKAN model |
| `models/c-tfidf_svm.pkl` | Trained SVM baseline |
| `models/metrics_c-tfidf_fastkan.json` | Performance metrics |
| `models/metrics_c-tfidf_svm.json` | SVM metrics for comparison |

### Load Trained Model

```python
import pickle
from pathlib import Path

# Load model
with open("models/c-tfidf_fastkan.pkl", "rb") as f:
    model_dict = pickle.load(f)

encoder = model_dict['encoder']
classifier = model_dict['classifier']
label_encoder = model_dict['label_encoder']
metrics = model_dict['metrics']

# Make predictions
X_test_enc = encoder.encode(questions)
predictions = classifier.predict(X_test_enc)
prediction_labels = label_encoder.inverse_transform(predictions)
```

---

### Training Parameters

```bash
# Custom epochs and hyperparameters
python src/train_fastkan.py \
    --encoder c-tfidf \
    --classifier fastkan \
    --epochs 50  # Increase training iterations
    --output models/
```

**Key Parameters in SimpleFastKAN:**
- `hidden_dim`: 64 (hidden layer size)
- `n_layers`: 2 (number of KAN layers)
- `batch_size`: 16 (mini-batch size)
- `learning_rate`: 0.001 (Adam optimizer)

---

### Academic Benefits

✅ **Model Comparison**: Compare FastKAN vs SVM baseline  
✅ **Reproducibility**: Fixed random seeds, stratified splits  
✅ **Validation**: Proper train/val/test separation  
✅ **Early Stopping**: Prevents overfitting with patience counter  
✅ **Hausa NLP**: Automatic preprocessing pipeline  
✅ **Production Ready**: Saved models can be loaded and used in production

---

### Integration with Streamlit App

Once models are trained, use them in your Streamlit app:

```python
# In streamlit_app.py
import pickle

@st.cache_resource
def load_trained_model():
    with open("models/c-tfidf_fastkan.pkl", "rb") as f:
        return pickle.load(f)

model_dict = load_trained_model()
encoder = model_dict['encoder']
classifier = model_dict['classifier']
```

---

### Expected Performance

On typical HausaTaxBot dataset with ~50-100 Q&A pairs:

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|----------------|
| c-TF-IDF + FastKAN | ~75-85% | ~0.75-0.85 | 30-60s |
| c-TF-IDF + SVM | ~70-80% | ~0.70-0.80 | <5s |
| ColBERT + SVM | ~80-90% | ~0.80-0.90 | 10-20s |

*(Actual performance depends on dataset size and quality)*

---

### Troubleshooting

**Q: "PyTorch not available" warning**  
A: Install PyTorch: `pip install torch`

**R: "CUDA out of memory"**  
A: Switch to CPU by modifying `device` selection in code

**Q: Model file too large**  
A: Consider using `joblib` instead of `pickle` for compression

---

### Next Steps

1. ✅ Train FastKAN + SVM baselines
2. ✅ Compare performance metrics  
3. 📊 Use for academic model comparison in course project
4. 🚀 Deploy best model to production

---

**Created:** Session 2, May 19, 2026  
**Author:** HausaTaxBot Research Team (COEN541/543)
