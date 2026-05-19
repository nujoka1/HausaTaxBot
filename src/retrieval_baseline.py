"""
src/retrieval_baseline.py
HausaTaxBot — Baseline retrieval engine (TF-IDF + cosine similarity)

Replicates the approach from:
  [1] Musa et al. — Hausa Intelligence Chatbot System (ResearchGate)
  [2] Retrieval-Based Chatbot Using Sentence Similarity (ResearchGate)

This is the BASELINE. Our improvements (c-TF-IDF, ColBERT, Model2Vec,
FastKAN) are implemented in feature_extraction.py & classifiers.py.

COEN541 · Ahmadu Bello University, Zaria
Lead: Eng. Nuhu  |  Encoding: Eng. Amanda
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import HausaPreprocessor


class BaselineRetriever:
    """
    TF-IDF + cosine similarity retrieval chatbot.
    Direct replication of the baseline papers' core approach.

    Pipeline:
        corpus Q&A  →  preprocess  →  TF-IDF matrix
        user query  →  preprocess  →  TF-IDF vector
                                    →  cosine similarity vs matrix
                                    →  top-k candidates  →  best answer
    """

    def __init__(
        self,
        top_k: int = 3,
        min_similarity: float = 0.10,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.preprocessor = HausaPreprocessor()
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=5000,
            sublinear_tf=True,      # log-normalise TF (improves retrieval)
        )
        self.corpus: List[Dict] = []
        self.question_matrix = None
        self.is_fitted = False

    # ── build ──────────────────────────────────

    def fit(self, dataset_path: str) -> "BaselineRetriever":
        """Load dataset and build the TF-IDF index."""
        self.corpus = self._load(dataset_path)
        if not self.corpus:
            raise ValueError(f"Dataset empty or not found: {dataset_path}")

        questions = [r["question_hausa"] for r in self.corpus]
        clean_questions = self.preprocessor.process_batch(questions)

        self.question_matrix = self.vectorizer.fit_transform(clean_questions)
        self.is_fitted = True

        print(f"[BaselineRetriever] Indexed {len(self.corpus)} Q&A pairs")
        print(f"[BaselineRetriever] Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self

    # ── inference ──────────────────────────────

    def retrieve(self, query: str) -> List[Dict]:
        """
        Return top-k candidates for a Hausa query.
        Each result dict contains: question, answer, similarity, source, intent.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .retrieve()")

        clean_query = self.preprocessor.process(query)
        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.question_matrix).flatten()

        # Rank and filter
        top_indices = np.argsort(similarities)[::-1][: self.top_k]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < self.min_similarity:
                break
            record = self.corpus[idx]
            results.append(
                {
                    "question_hausa": record["question_hausa"],
                    "answer_hausa": record["answer_hausa"],
                    "intent": record.get("intent", "unknown"),
                    "source": record.get("source", ""),
                    "similarity": round(score, 4),
                }
            )
        return results

    def answer(self, query: str) -> str:
        """Return the single best Hausa answer for a query."""
        results = self.retrieve(query)
        if not results:
            return (
                "Hakika. Ba mu sami amsa ga tambayar ku a cikin bayananmu ba. "
                "Da fatan za a gwada tambaya ta daban."
            )
        return results[0]["answer_hausa"]

    # ── utils ──────────────────────────────────

    @staticmethod
    def _load(path: str) -> List[Dict]:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Dataset file not found: {path}")
            return []
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept both a plain list and a dict with a 'data' key
        return data if isinstance(data, list) else data.get("data", [])

    def get_stats(self) -> Dict:
        """Return corpus statistics for the evaluation report."""
        if not self.corpus:
            return {}
        intents = [r.get("intent", "unknown") for r in self.corpus]
        from collections import Counter
        intent_counts = Counter(intents)
        return {
            "total_pairs": len(self.corpus),
            "vocab_size": len(self.vectorizer.vocabulary_) if self.is_fitted else 0,
            "intent_distribution": dict(intent_counts),
        }


# ─────────────────────────────────────────────
# Quick smoke test (run directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys

    # Use a tiny inline corpus if real data not yet available
    SAMPLE_CORPUS = [
        {
            "id": "TAX_001",
            "intent": "vat",
            "question_hausa": "Menene kason VAT da ke aiki a Najeriya a yanzu?",
            "answer_hausa": "Kason VAT a Najeriya shine kashi 7.5% akan duk kaya da ayyuka.",
            "source": "Finance Act 2025, s.4",
        },
        {
            "id": "TAX_002",
            "intent": "personal_income_tax",
            "question_hausa": "Wane adadin haraji ake karba daga albashi?",
            "answer_hausa": "Haraji akan albashi ya dogara da yawan kudin shiga. Waɗanda ke ƙasa da N300,000 ba sa biyan haraji.",
            "source": "Finance Act 2025, s.37",
        },
        {
            "id": "TAX_003",
            "intent": "company_income_tax",
            "question_hausa": "Kamfanoni masu karamin kasuwanci suna biyan nawa haraji?",
            "answer_hausa": "Kamfanoni da kudin shigarsu ya kasa da N25 miliyan a shekara ba su biyan haraji na kamfani.",
            "source": "Finance Act 2025, s.14",
        },
        {
            "id": "TAX_004",
            "intent": "withholding_tax",
            "question_hausa": "Yaya ake aiwatar da haraji na kame ga masu bada sabis?",
            "answer_hausa": "Ana kame kashi 5% ko 10% na kowane biya kafin ya isa hannun mai karba, sannan a miƙa shi ga FIRS.",
            "source": "Finance Act 2025, s.22",
        },
        {
            "id": "TAX_005",
            "intent": "digital_services_tax",
            "question_hausa": "Shin kamfanonin fasaha na waje suna biyan haraji a Najeriya?",
            "answer_hausa": "Ee. Kamfanonin waje da ke ba da ayyukan dijital a Najeriya dole ne su yi rijista da FIRS su kuma biya haraji.",
            "source": "Finance Act 2025, s.9",
        },
    ]

    # Write temp corpus
    tmp = "/tmp/sample_corpus.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_CORPUS, f, ensure_ascii=False, indent=2)

    bot = BaselineRetriever(top_k=2, min_similarity=0.05)
    bot.fit(tmp)

    print("\n" + "=" * 60)
    print("BASELINE RETRIEVAL — SMOKE TEST")
    print("=" * 60)

    test_queries = [
        "Nawa ne kason VAT a Najeriya?",
        "Haraji akan albashi nawa ne?",
        "Google da Amazon suna biyan haraji a Najeriya?",
    ]

    for q in test_queries:
        print(f"\n  Tambaya : {q}")
        results = bot.retrieve(q)
        if results:
            print(f"  Amsa    : {results[0]['answer_hausa']}")
            print(f"  Score   : {results[0]['similarity']} | Intent: {results[0]['intent']}")
        else:
            print("  → Ba a sami amsa ba")

    print("\n  Stats:", bot.get_stats())
