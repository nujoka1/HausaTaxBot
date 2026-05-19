"""
src/feature_extraction.py
HausaTaxBot — Feature Extraction & Encoding Module
COEN541 · Ahmadu Bello University, Zaria · 2025/2026

Implements all 3 required encoding techniques:
  1. c-TF-IDF   (class-based TF-IDF via BERTopic)
  2. ColBERT    (late-interaction neural retrieval via RAGatouille)
  3. Model2Vec  (distilled static embeddings)

Each encoder exposes the same interface:
    encoder.fit(corpus)
    encoder.encode(texts)        → np.ndarray [N, D]
    encoder.retrieve(query, k)   → List[Dict]

Lead: Eng. Amanda  |  Architecture review: Eng. Nuhu
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import HausaPreprocessor


# ════════════════════════════════════════════════════════════
# 1. c-TF-IDF Encoder
#    Class-based TF-IDF from BERTopic — treats each intent
#    as a "class" and builds class-level term importance.
#    This is superior to standard TF-IDF for intent-grouped
#    retrieval because rare tax terms get boosted per intent.
# ════════════════════════════════════════════════════════════

class CTFIDFEncoder:
    """
    c-TF-IDF: Class-based Term Frequency – Inverse Document Frequency.

    Difference from standard TF-IDF:
      - Standard TF-IDF: term importance per document
      - c-TF-IDF:        term importance per CLASS (intent)
        → all documents of the same intent are concatenated
          into one "super-document" before TF-IDF is computed

    Why better for our chatbot:
      All VAT questions share domain vocabulary. c-TF-IDF
      surfaces "kaso", "keɓance", "VAT" as strongly VAT-
      associated even if individual questions are short.
    """

    def __init__(self):
        self.preprocessor = HausaPreprocessor()
        self.vectorizer = None
        self.class_matrix = None       # [n_intents, vocab]
        self.intent_labels: List[str] = []
        self.corpus: List[Dict] = []
        self.doc_vectors = None        # [n_docs, vocab] for retrieval
        self.is_fitted = False

    def fit(self, corpus: List[Dict]) -> "CTFIDFEncoder":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import defaultdict

        self.corpus = corpus

        # Group documents by intent (class)
        class_docs: Dict[str, List[str]] = defaultdict(list)
        for record in corpus:
            intent = record.get("intent", "unknown")
            text = self.preprocessor.process(record["question_hausa"])
            class_docs[intent].append(text)

        self.intent_labels = sorted(class_docs.keys())

        # Concatenate each class into one super-document
        super_docs = [
            " ".join(class_docs[intent]) for intent in self.intent_labels
        ]

        # Fit TF-IDF on super-documents (this IS c-TF-IDF)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
        )
        self.class_matrix = self.vectorizer.fit_transform(super_docs)

        # Also encode every individual document for retrieval
        all_clean = [
            self.preprocessor.process(r["question_hausa"]) for r in corpus
        ]
        self.doc_vectors = self.vectorizer.transform(all_clean)
        self.is_fitted = True

        print(f"[c-TF-IDF] Fitted on {len(corpus)} docs, "
              f"{len(self.intent_labels)} classes, "
              f"vocab={len(self.vectorizer.vocabulary_)}")
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        clean = [self.preprocessor.process(t) for t in texts]
        return self.vectorizer.transform(clean).toarray()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first")
        clean = self.preprocessor.process(query)
        q_vec = self.vectorizer.transform([clean])
        sims  = cosine_similarity(q_vec, self.doc_vectors).flatten()
        return _rank(sims, self.corpus, top_k)

    def predict_intent(self, query: str) -> str:
        """Predict intent by matching query to closest class super-doc."""
        clean = self.preprocessor.process(query)
        q_vec = self.vectorizer.transform([clean])
        sims  = cosine_similarity(q_vec, self.class_matrix).flatten()
        return self.intent_labels[int(np.argmax(sims))]


# ════════════════════════════════════════════════════════════
# 2. ColBERT Encoder
#    Late-interaction neural retrieval.
#    Each token in query attends to each token in document
#    (MaxSim operation) — far more expressive than single
#    vector cosine similarity.
# ════════════════════════════════════════════════════════════

class ColBERTEncoder:
    """
    ColBERT via RAGatouille library.

    Architecture:
      query  → BERT → [q1, q2, ..., qm] token vectors
      doc    → BERT → [d1, d2, ..., dn] token vectors
      score  = Σ max_j(qi · dj)   (late interaction / MaxSim)

    Why for our chatbot:
      Short Hausa tax queries benefit from token-level matching.
      "kaso vat" can match "adadin VAT" even if phrased differently
      because BERT contextualises each token.

    Note: First call downloads the ColBERT model (~400MB).
    Subsequent calls load from cache.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name
        self.corpus: List[Dict] = []
        self.index = None
        self.rag = None
        self.is_fitted = False
        self._index_path = "/tmp/colbert_hausa_tax_index"

    def fit(self, corpus: List[Dict]) -> "ColBERTEncoder":
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError:
            print("[ColBERT] ragatouille not installed. Run: pip install ragatouille")
            return self

        self.corpus = corpus
        self.rag = RAGPretrainedModel.from_pretrained(self.model_name)

        # RAGatouille expects a list of strings — we use answer_hausa as the
        # retrieval target and question_hausa as the indexed content
        documents = [r["question_hausa"] + " " + r["answer_hausa"]
                     for r in corpus]
        doc_ids   = [r["id"] for r in corpus]

        self.rag.index(
            collection=documents,
            document_ids=doc_ids,
            index_name="hausa_tax_bot",
            split_documents=False,
            overwrite_index=True,
        )
        self.is_fitted = True
        print(f"[ColBERT] Index built for {len(corpus)} documents")
        return self

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first")
        raw_results = self.rag.search(query=query, k=top_k)
        results = []
        for r in raw_results:
            doc_id  = r["document_id"]
            record  = next((c for c in self.corpus if c["id"] == doc_id), None)
            if record:
                results.append({
                    "question_hausa": record["question_hausa"],
                    "answer_hausa":   record["answer_hausa"],
                    "intent":         record.get("intent", ""),
                    "source":         record.get("source", ""),
                    "similarity":     round(float(r["score"]), 4),
                })
        return results

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to ColBERT query vectors (for SVM/FastKAN input)."""
        if self.rag is None:
            raise RuntimeError("Call .fit() first")
        # RAGatouille doesn't expose raw embeddings easily;
        # we use the underlying model's encode for classification
        embeddings = self.rag.model.encode(texts)
        return np.array(embeddings)


# ════════════════════════════════════════════════════════════
# 3. Model2Vec Encoder
#    Ultra-fast distilled static embeddings.
#    Distilled from a sentence transformer, giving
#    sentence-level vectors without inference overhead.
# ════════════════════════════════════════════════════════════

class Model2VecEncoder:
    """
    Model2Vec: distilled sentence embeddings.

    Key properties:
      - 500x faster than the source sentence transformer
      - No GPU needed — runs on CPU in Colab free tier
      - 256-dimensional dense vectors
      - Good multilingual coverage (includes Hausa-adjacent languages)

    For our pipeline: Model2Vec vectors feed into SVM and FastKAN.
    """

    MODEL_ID = "minishlab/M2V_base_output"   # 256-dim, 7500 tokens/sec

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or self.MODEL_ID
        self.model = None
        self.corpus: List[Dict] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.is_fitted = False

    def _load(self):
        if self.model is not None:
            return
        try:
            from model2vec import StaticModel
            self.model = StaticModel.from_pretrained(self.model_id)
            print(f"[Model2Vec] Loaded: {self.model_id}")
        except ImportError:
            raise ImportError("Run: pip install model2vec")

    def fit(self, corpus: List[Dict]) -> "Model2VecEncoder":
        self._load()
        self.corpus = corpus
        questions = [r["question_hausa"] for r in corpus]
        self.doc_embeddings = self.model.encode(questions)
        self.is_fitted = True
        print(f"[Model2Vec] Encoded {len(corpus)} documents "
              f"→ shape {self.doc_embeddings.shape}")
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        self._load()
        return self.model.encode(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first")
        q_vec = self.encode([query])
        sims  = cosine_similarity(q_vec, self.doc_embeddings).flatten()
        return _rank(sims, self.corpus, top_k)



# ════════════════════════════════════════════════════════════
# Shared utility
# ════════════════════════════════════════════════════════════

def _rank(similarities: np.ndarray, corpus: List[Dict],
          top_k: int, min_score: float = 0.05) -> List[Dict]:
    """Sort corpus by similarity and return top-k results."""
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < min_score:
            break
        r = corpus[idx]
        results.append({
            "question_hausa": r["question_hausa"],
            "answer_hausa":   r["answer_hausa"],
            "intent":         r.get("intent", ""),
            "source":         r.get("source", ""),
            "similarity":     round(score, 4),
        })
    return results


def get_encoder(name: str):
    """
    Factory function — returns the correct encoder by name.
    Usage: enc = get_encoder('colbert')
    """
    mapping = {
        "ctfidf":   CTFIDFEncoder,
        "c-tfidf":  CTFIDFEncoder,
        "colbert":  ColBERTEncoder,
        "model2vec": Model2VecEncoder,
    }
    key = name.lower().replace(" ", "")
    if key not in mapping:
        raise ValueError(f"Unknown encoder: {name}. Choose from {list(mapping)}")
    return mapping[key]()


# ════════════════════════════════════════════════════════════
# Quick smoke test
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json

    with open("../data/raw/hausa_tax_qa_sample.json", encoding="utf-8") as f:
        corpus = json.load(f)

    test_query = "Nawa ne kason VAT a Najeriya?"

    print("=" * 60)
    print("FEATURE EXTRACTION — SMOKE TEST")
    print("=" * 60)

    # c-TF-IDF
    print("\n[1] c-TF-IDF")
    enc1 = CTFIDFEncoder().fit(corpus)
    r1   = enc1.retrieve(test_query, top_k=1)
    print(f"    Query  : {test_query}")
    print(f"    Answer : {r1[0]['answer_hausa'] if r1 else 'No result'}")
    print(f"    Intent : {enc1.predict_intent(test_query)}")

    # Model2Vec
    print("\n[2] Model2Vec")
    try:
        enc2 = Model2VecEncoder().fit(corpus)
        r2   = enc2.retrieve(test_query, top_k=1)
        vecs = enc2.encode([test_query])
        print(f"    Answer : {r2[0]['answer_hausa'] if r2 else 'No result'}")
        print(f"    Vector shape: {vecs.shape}")
    except ImportError:
        print("    → Install model2vec to test this encoder")

    print("\n✓ Smoke test complete")
