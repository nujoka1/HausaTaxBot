"""
HausaTaxBot - Streamlit Main Application
=========================================

Hausa-language tax QA chatbot with retrieval-based answer generation.

Architecture:
- Semantic retrieval (encoder + cosine similarity)
- Intent classification (optional secondary signal)
- Keyword fallback for low-confidence queries
- Hausa text preprocessing
- Confidence calibration with thresholds

UI:
- Chat interface in Hausa
- Real-time model diagnostics
- Response metadata display
- Export functionality

Author: HausaTaxBot Research Team (COEN541)
"""

import streamlit as st
import json
import pandas as pd
import os
from pathlib import Path
import time
from datetime import datetime
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
import io
import base64
import logging
import sys

# Import new modular components
from src.hausa_preprocessing import HausaPreprocessor, create_preprocessor
from src.retrieval_pipeline import (
    SemanticRetriever, IntentFilteredRetriever, KeywordFallbackRetriever,
    RetrievalResult, ConfidenceLevel
)
from src.embedding_cache import EmbeddingCache

# ==================== LOGGING SETUP ====================
# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "hausataxbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HausaTaxBot")
logger.info("=" * 50)
logger.info("HausaTaxBot application started")
logger.info("=" * 50)

# ==================== ENCODER CLASSES ====================
class CTFIDFEncoder:
    """BERTopic's c-TF-IDF encoder"""
    def __init__(self, n_features=8000, ngram=(1, 2)):
        self.n_features = n_features
        self.ngram = ngram
        self.cv = CountVectorizer(max_features=n_features, ngram_range=ngram, min_df=1, max_df=0.98)
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
    """ColBERT encoder with TF-IDF + SVD"""
    def __init__(self, n_char_features=6000, n_components=256, char_ngram=(3, 5)):
        self.n_components = n_components
        self.char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=char_ngram,
                                                max_features=n_char_features, min_df=1, 
                                                sublinear_tf=True, norm='l2')
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000,
                                                min_df=1, sublinear_tf=True, norm='l2')
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, corpus):
        char_X = self.char_vectorizer.fit_transform(corpus)
        word_X = self.word_vectorizer.fit_transform(corpus)
        combined = sp.hstack([char_X, word_X])
        self.svd.fit(combined)
        self._fitted = True
        return self

    def encode(self, texts):
        char_X = self.char_vectorizer.transform(texts)
        word_X = self.word_vectorizer.transform(texts)
        combined = sp.hstack([char_X, word_X])
        E = self.svd.transform(combined).astype(np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return E / norms

    def transform(self, texts):
        return self.encode(texts)


class Model2VecEncoder:
    """Model2Vec encoder using spectral decomposition"""
    def __init__(self, n_components=256, n_features=5000):
        self.n_components = n_components
        self.n_features = n_features
        self.tfidf = TfidfVectorizer(max_features=n_features, ngram_range=(1, 2),
                                      min_df=1, max_df=0.98, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42,
                                algorithm='randomized')
        self.token_embed_ = None
        self._fitted = False

    def fit(self, corpus):
        T = self.tfidf.fit_transform(corpus)
        U, S, Vt = sp.linalg.svds(T.T, k=self.n_components)
        self.token_embed_ = (U * S).astype(np.float32)
        self._fitted = True
        return self

    def encode(self, texts):
        T = self.tfidf.transform(texts).astype(np.float32)
        E = T @ self.token_embed_
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return E / norms

    def transform(self, texts):
        return self.encode(texts)


# ==================== CLASSIFIER CLASSES ====================
class SVMClassifier:
    """SVM classifier with balanced class weights"""
    def __init__(self, kernel='rbf', C=5.0):
        self.kernel = kernel
        self.C = C
        self.le_ = LabelEncoder()
        self.model_ = None
        self.classes_ = None

    def fit(self, X_train, y_train, encoder_name=''):
        y_enc = self.le_.fit_transform(y_train)
        self.classes_ = self.le_.classes_
        if encoder_name == 'c-TF-IDF':
            self.model_ = LinearSVC(C=self.C, class_weight='balanced',
                                    max_iter=5000, dual=True)
        else:
            self.model_ = SVC(kernel=self.kernel, C=self.C, gamma='scale',
                              class_weight='balanced', decision_function_shape='ovr',
                              random_state=42, probability=True)
        self.model_.fit(X_train, y_enc)
        return self

    def predict(self, X):
        pred_enc = self.model_.predict(X)
        return self.le_.inverse_transform(pred_enc)

    def predict_proba(self, X):
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)
        else:
            # LinearSVC doesn't have predict_proba, use decision_function
            decision = self.model_.decision_function(X)
            if len(decision.shape) == 1:
                decision = decision.reshape(-1, 1)
            # Softmax approximation
            decision = np.clip(decision, -100, 100)
            decision = np.clip(decision, -100, 100)
            exp_decision = np.exp(decision)
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)


class FastKANClassifier:
    """FastKAN RBF-based classifier"""
    def __init__(self, hidden_dim=128, grid_size=5, lr=0.01, n_epochs=80,
                 batch_size=64, sigma=0.5):
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sigma = sigma
        self.le_ = LabelEncoder()
        self.classes_ = None
        self._fitted = False

    def _rbf_expand(self, X, centres, sigma):
        diff = X[:, None, :] - centres[None, :, :]
        sq = (diff**2).sum(axis=-1)
        return np.exp(-sq / (2 * sigma**2))

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        eZ = np.exp(Z)
        return eZ / eZ.sum(axis=1, keepdims=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None, encoder_name=''):
        rng = np.random.default_rng(42)
        n, d = X_train.shape
        y_enc = self.le_.fit_transform(y_train)
        self.classes_ = self.le_.classes_
        n_cls = len(self.classes_)
        G = self.grid_size
        
        idx = rng.choice(n, size=min(G, n), replace=False)
        self.centres_ = X_train[idx].copy()
        self.sigma_ = self.sigma
        self.W1_ = rng.normal(0, 0.1, (len(self.centres_), self.hidden_dim))
        self.b1_ = np.zeros(self.hidden_dim)
        self.W2_ = rng.normal(0, 0.1, (self.hidden_dim, n_cls))
        self.b2_ = np.zeros(n_cls)
        
        eye = np.eye(n_cls)
        y_oh = eye[y_enc]
        self._fitted = True
        return self

    def _forward(self, X):
        Phi = self._rbf_expand(X, self.centres_, self.sigma_)
        H = np.tanh(Phi @ self.W1_ + self.b1_)
        logits = H @ self.W2_ + self.b2_
        return self._softmax(logits)

    def predict(self, X):
        probs = self._forward(X)
        pred = probs.argmax(axis=1)
        return self.le_.inverse_transform(pred)

    def predict_proba(self, X):
        return self._forward(X)

# ==================== UTILITY FUNCTIONS ====================
def export_chat_to_json(messages):
    """Export chat history to JSON"""
    export_data = {
        "an_fitar_a": datetime.now().isoformat(),
        "jimillar_sakonni": len(messages),
        "tattaunawa": messages
    }
    return json.dumps(export_data, ensure_ascii=False, indent=2)

def export_chat_to_csv(messages):
    """Export chat history to CSV"""
    rows = []
    for i, msg in enumerate(messages, 1):
        row = {
            "Saƙo #": i,
            "Matsayi": msg.get("role", "").upper(),
            "Abun ciki": msg.get("content", ""),
            "Tabbaci": msg.get("metadata", {}).get("confidence", "B/A"),
            "Nau'in": msg.get("metadata", {}).get("intent", "Ba a sani ba"),
            "Lokaci": msg.get("metadata", {}).get("timestamp", "")
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8')

def get_download_link(file_content, filename, file_format):
    """Generate download link for files"""
    if file_format == "json":
        mime = "application/json"
        content = file_content.encode()
    elif file_format == "csv":
        mime = "text/csv"
        content = file_content.encode('utf-8')
    else:
        mime = "text/plain"
        content = file_content.encode()
    
    b64 = base64.b64encode(content).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}">Sauke {file_format.upper()}</a>'

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="HausaTaxBot - Mataimakin Haraji na Najeriya",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODERN CLASSIC GREEN UI ====================

st.markdown("""
<style>

/* ================= GLOBAL ================= */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main App Background */
.stApp {

    background: linear-gradient(
        135deg,
        #010D09 0%,
        #021B13 20%,
        #06281D 45%,
        #0B3D2E 70%,
        #145A32 100%
    );

    color: white;
}
            
/* Remove Streamlit default padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ================= HEADER ================= */

.main-header {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.15);

    border-radius: 24px;

    padding: 28px;

    margin-bottom: 25px;

    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
}

.header-title {
    font-size: 42px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.header-subtitle {
    font-size: 17px;
    color: #DFFFE2;
}

.logo-icon {
    font-size: 70px;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #031E17,
        #0B3D2E
    ) !important;

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar cards */
.sidebar-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
}

/* ================= CHAT BUBBLES ================= */

.user-message {
    background: linear-gradient(
        135deg,
        #B6FFB3,
        #E9FFE9
    );

    color: #111111;

    padding: 18px;

    border-radius: 20px;

    margin-bottom: 16px;

    border-left: 7px solid #1E8449;

    box-shadow: 0px 6px 15px rgba(0,0,0,0.15);

    animation: fadeIn 0.4s ease-in-out;
}

.bot-message {
    background: rgba(255,255,255,0.92);

    color: #111111;

    padding: 18px;

    border-radius: 20px;

    margin-bottom: 16px;

    border-left: 7px solid #0B3D2E;

    box-shadow: 0px 6px 15px rgba(0,0,0,0.15);

    animation: fadeIn 0.4s ease-in-out;
}

.user-message strong {
    color: #145A32;
}

.bot-message strong {
    color: #0B3D2E;
}

/* ================= BUTTONS ================= */

.stButton > button {

    background: linear-gradient(
        90deg,
        #021B13,
        #06281D,
        #0B3D2E
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 14px !important;

    padding: 12px 22px !important;

    font-size: 15px !important;

    font-weight: 600 !important;

    transition: all 0.3s ease-in-out !important;

    box-shadow: 0px 5px 12px rgba(0,0,0,0.35);
}

/* ================= INPUT ================= */

.stTextInput input {

    background: rgba(255,255,255,0.95) !important;

    color: #111111 !important;

    border-radius: 16px !important;

    border: 2px solid #B6FFB3 !important;

    padding: 14px !important;

    font-size: 15px !important;

    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* ================= SELECT BOX ================= */

.stSelectbox div[data-baseweb="select"] {

    background: rgba(255,255,255,0.95) !important;

    color: black !important;

    border-radius: 14px !important;

    border: 2px solid #B6FFB3 !important;
}

/* ================= METRIC CARDS ================= */

div[data-testid="metric-container"] {

    background: rgba(255,255,255,0.92);

    border-radius: 20px;

    padding: 20px;

    border: 1px solid rgba(0,0,0,0.06);

    box-shadow: 0px 5px 15px rgba(0,0,0,0.12);
}

/* ================= PROGRESS BAR ================= */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #145A32,
        #B6FFB3
    );
}

/* ================= DIVIDER ================= */

hr {
    border: none;
    height: 2px;

    background: linear-gradient(
        to right,
        #B6FFB3,
        #145A32,
        #021B13
    );
}

/* ================= CONFIDENCE BADGES ================= */

.confidence-badge {

    display: inline-block;

    padding: 6px 14px;

    border-radius: 30px;

    font-size: 12px;

    font-weight: 700;
}

.confidence-high {

    background: #DFFFE2;

    color: #145A32;
}

/* ================= SCROLLBAR ================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {

    background: linear-gradient(
        #145A32,
        #27AE60
    );

    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: #DFFFE2;
}

/* ================= ANIMATION ================= */

@media (max-width: 768px) {

    .header-title {
        font-size: 28px;
    }

    .bot-message,
    .user-message {
        padding: 14px;
        font-size: 14px;
    }

    .stButton > button {
        width: 100%;
    }
}

/* Additional mobile responsive refinements */
@media (max-width: 480px) {
    .header-title { font-size: 22px; }
    .bot-message, .user-message { font-size: 13px; padding: 12px; }
    .stButton > button { width: 100%; font-size: 14px; padding: 10px 12px; }
    .stTextInput input { font-size: 14px; padding: 12px; }
}

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_resource
def load_qa_data():
    """Load Hausa Q&A data from JSON file"""
    json_path = Path(__file__).parent / "hausa_tax_qa.json"
    
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"⚠️ An kasa loda bayanan tambaya da amsa. Tabbatar cewa fayil ɗin 'hausa_tax_qa.json' yana cikin babban fayil ɗaya: {e}")
            return None
    else:
        st.warning(f"⚠️ Ba a samu fayil ɗin bayanai ba a {json_path}.")
        return None

@st.cache_resource
def load_csv_data():
    """Load full translated dataset from CSV for analytics"""
    csv_path = Path(__file__).parent / "HausaTaxBot_PIT_Data_TRANSLATED.csv"
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None
    return None

@st.cache_resource
def load_available_models():
    """Load available model pairs metadata from models/available_models.json.
    
    Returns:
        Tuple of (available_pairs: list, best_pair: str or None)
        Returns ([], None) if models not available
    """
    logger.info("Loading available model pairs...")
    models_dir = Path(__file__).parent / "models"
    metadata_file = models_dir / "available_models.json"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    logger.debug(f"Models directory: {models_dir}")
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pairs = data.get('available_pairs', [])
                best_pair = data.get('best_pair', None)
                logger.info(f"✓ Successfully loaded {len(pairs)} model pairs. Best pair: {best_pair}")
                logger.debug(f"Available pairs: {pairs}")
                return pairs, best_pair
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata file: {e}")
            return [], None
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}", exc_info=True)
            return [], None
    else:
        logger.warning(f"Model metadata file not found. Expected path: {metadata_file}")
        logger.info("Models will be initialized after notebook training completes.")
        return [], None


@st.cache_resource
def load_trained_model_pair(pair_name):
    """Load selected trained model pair"""

    model_path = Path(__file__).parent / "models" / f"{pair_name}.pkl"

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        return None, None

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        encoder = model_data["encoder"]
        classifier = model_data["classifier"]

        return encoder, classifier

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def _get_simple_encoder():
    """Returns a simple encoder that outputs 1-dimensional zero vectors."""
    class SimpleEncoder:
        def __init__(self):
            self.n_features_in_ = 0
        
        def transform(self, texts):
            return np.zeros((len(texts), 1))
        
        def encode(self, texts):
            return self.transform(texts)
    
    return SimpleEncoder()


def _get_simple_classifier():
    """Returns a simple classifier that always predicts class 0."""
    class SimpleClassifier:
        classes_ = np.array([0])
        
        def predict(self, X):
            return np.zeros((X.shape[0],), dtype=int)
        
        def predict_proba(self, X):
            # Return 50-50 probability
            return np.tile(np.array([0.5, 0.5]), (X.shape[0], 1))
    
    return SimpleClassifier()

# ==================== REFACTORED RETRIEVAL SYSTEM ====================
def retrieve_answer_v2(query: str, 
                       qa_data: dict, 
                       encoder, 
                       classifier, 
                       encoder_type: str, 
                       classifier_type: str,
                       preprocessor: HausaPreprocessor = None) -> dict:
    """
    IMPROVED Retrieval using semantic similarity with proper confidence calibration.
    
    Pipeline:
    1. Preprocess query (Hausa normalization)
    2. Semantic retrieval (encode + cosine similarity)
    3. Optional intent filtering
    4. Keyword fallback if low confidence
    
    Key Improvements:
    - HIGH confidence threshold: 0.60 (was 0.25 - too low, caused hallucinations)
    - Removed bad context concatenation
    - Proper confidence levels (HIGH/MEDIUM/LOW)
    - Full diagnostic logging
    
    Args:
        query: User question
        qa_data: Q&A database
        encoder: Fitted encoder
        classifier: Intent classifier (optional)
        encoder_type: Encoder name
        classifier_type: Classifier name
        preprocessor: Hausa text preprocessor
        
    Returns:
        Dict with match quality, confidence, diagnostics
    """
    # Constants for confidence calibration
    HIGH_CONFIDENCE_THRESHOLD = 0.70  # Only return answers >= 70% confidence
    SEMANTIC_SIMILARITY_THRESHOLD = 0.35  # Min semantic similarity to consider
    
    logger.info(f"[RETRIEVAL] Query: '{query[:50]}...' | Encoder: {encoder_type} | Classifier: {classifier_type}")
    
    if not qa_data or not qa_data.get('qa_pairs'):
        logger.error("Q&A data not available!")
        return {
            'match': None,
            'score': 0.0,
            'is_confident': False,
            'method': 'Error - No Data',
            'confidence_level': 'BA TABBATA'
        }
    
    # STEP 1: Preprocess query (remove noise, normalize Hausa text)
    processed_query = query
    if preprocessor:
        processed_query = preprocessor.preprocess(query)
        logger.debug(f"Processed query: '{query}' -> '{processed_query}'")
    
    # STEP 2: Try semantic retrieval if encoder available
    if encoder is not None:
        try:
            # Encode query
            if hasattr(encoder, 'transform'):
                query_emb = encoder.transform([processed_query])
            else:
                query_emb = encoder.encode([processed_query])
            
            query_emb = np.asarray(query_emb, dtype=np.float32)
            
            # Check for zero/broken embeddings
            if np.all(query_emb == 0):
                logger.warning("Zero embedding - encoder may be broken")
                raise ValueError("Zero embedding from encoder")
            
            # Normalize embedings
            query_emb_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
            
            # Get all questions (use cache for embeddings)
            qa_pairs = qa_data.get('qa_pairs', [])
            questions = [p.get('question', '') for p in qa_pairs]
            
            # Use embedding cache to avoid recomputation
            try:
                embedding_cache = EmbeddingCache()
                qa_emb = embedding_cache.get_cached_embeddings(
                    questions, encoder, encoder_type, use_disk_cache=True
                )
                logger.debug(f"✅ Used cached embeddings for {encoder_type}")
            except Exception as cache_error:
                logger.warning(f"Cache retrieval failed: {cache_error}, computing directly")
                if hasattr(encoder, 'transform'):
                    qa_emb = encoder.transform(questions)
                else:
                    qa_emb = encoder.encode(questions)
            
            qa_emb = np.asarray(qa_emb, dtype=np.float32)
            qa_emb_norm = qa_emb / (np.linalg.norm(qa_emb, axis=1, keepdims=True) + 1e-10)
            
            # Compute cosine similarities
            similarities = (qa_emb_norm @ query_emb_norm.T).flatten()
            best_idx = np.argmax(similarities)
            best_similarity = float(similarities[best_idx])
            
            logger.debug(f"Best semantic similarity: {best_similarity:.4f}")
            
            # If similarity > threshold, return match
            if best_similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                best_pair = qa_pairs[best_idx]
                confidence_level = 'DUR (High)' if best_similarity >= 0.70 else 'WACCE (Medium)' if best_similarity >= 0.50 else 'BA TABBATA (Low)'
                
                return {
                    'match': best_pair,
                    'score': best_similarity,
                    'confidence_level': confidence_level,
                    'is_confident': best_similarity >= HIGH_CONFIDENCE_THRESHOLD,
                    'method': 'Semantic',
                    'encoder': encoder_type,
                    'classifier': classifier_name if classifier else 'None',
                    'intent': best_pair.get('intent', 'Ba a sani ba'),
                    'matched_question': best_pair.get('question', '')
                }
            else:
                logger.info(f"Semantic similarity {best_similarity:.4f} below threshold {SEMANTIC_SIMILARITY_THRESHOLD}")
        
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}", exc_info=True)
    
    # FALLBACK: Keyword-based matching
    logger.info("Falling back to keyword-based retrieval...")
    query_lower = processed_query.lower().strip()
    best_match = None
    best_score = 0.0
    
    for pair in qa_data.get('qa_pairs', []):
        question = pair.get('question', '').lower()
        keywords_str = pair.get('keywords', '').lower()
        
        # Phrase match
        score = 1.0 if query_lower in question else 0.0
        
        # Keyword matching
        if keywords_str and score == 0:
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
            matched = sum(1 for k in keywords if k in query_lower)
            score = matched / (len(keywords) + 0.001)
        
        if score > best_score:
            best_score = score
            best_match = pair
    
    logger.debug(f"Best keyword match score: {best_score:.4f}")
    
    if best_score < 0.25:
        # No confident match found
        return {
            'match': None,
            'score': best_score,
            'confidence_level': 'BA TABBATA',
            'is_confident': False,
            'method': 'Keyword_NoMatch',
            'encoder': encoder_type,
            'classifier': classifier_type if classifier else 'None'
        }
    
    # Return keyword match if confident enough
    confidence_level = 'DUR (High)' if best_score >= 0.70 else 'WACCE (Medium)' if best_score >= 0.50 else 'BA TABBATA (Low)'
    
    return {
        'match': best_match,
        'score': best_score,
        'confidence_level': confidence_level,
        'is_confident': best_score >= 0.60,
        'method': 'Keyword',
        'encoder': encoder_type,
        'classifier': classifier_type if classifier else 'None',
        'intent': best_match.get('intent', 'Ba a sani ba'),
        'matched_question': best_match.get('question', '') if best_match else ''
    }


# Maintain backward compatibility with old function name
def retrieve_answer(query, qa_data, encoder, classifier, encoder_type, classifier_type):
    """Wrapper for backward compatibility."""
    return retrieve_answer_v2(query, qa_data, encoder, classifier, encoder_type, classifier_type)


# ==================== MAIN APP ====================
def main():
    # Header
    col1, col2 = st.columns([1, 10])
    with col1:
        st.markdown("<div class='logo-icon'>🇳🇬</div>", unsafe_allow_html=True)
    with col2:
        st.title("HausaTaxBot")
        st.markdown("**Mataimakin Haraji na Najeriya cikin Harshen Hausa**")
        st.markdown("*Samun bayanai kan haraji cikin sauƙi a Harshen Hausa | Harajin Kuɗin Shiga a Najeriya*")
    
    st.divider()
    
    # Load data
    qa_data = load_qa_data()
    csv_data = load_csv_data()
    
    
    if not qa_data:
        st.markdown("*Samun bayanai kan haraji cikin sauƙi a Harshen Hausa | Harajin Kuɗin Shiga a Najeriya*")
        return
    
# Sidebar configuration
    with st.sidebar:
        st.header("Saituna")

        # Load available model pairs
        available_pairs, best_pair = load_available_models()
        
        if available_pairs:
            # Create dropdown with model pair names
            pair_options = [f"{p['encoder'].upper()} + {p['classifier'].upper()}" 
                           for p in available_pairs]
            pair_names_internal = [p['pair_name'] for p in available_pairs]
            
            # Set default to best model if available
            default_idx = 0
            if best_pair:
                best_internal = f"{best_pair['encoder']}_{best_pair['classifier']}"
                if best_internal in pair_names_internal:
                    default_idx = pair_names_internal.index(best_internal)
            
            st.subheader("Samfuran AI")
            selected_model_display = st.selectbox(
                "Zaɓi tsarin bincike:",
                pair_options,
                index=default_idx,
                help="Zaɓi kombinuwar encoder da classifier"
            )
            
            logger.info(f"User selected model: {selected_model_display}")
            
            # Get internal pair name
            selected_idx = pair_options.index(selected_model_display)
            selected_pair_name = pair_names_internal[selected_idx]
            logger.debug(f"Internal pair name: {selected_pair_name}")
            
            # Load selected model pair
            encoder, classifier = load_trained_model_pair(selected_pair_name)
            encoder_type = selected_pair_name.split('_')[0].upper()
            classifier_type = selected_pair_name.split('_')[1].upper()
            logger.info(f"Model loaded. Encoder: {encoder_type}, Classifier: {classifier_type}")
        else:
            logger.warning("Ba a sami samfuran da aka horar ba. Ana komawa binciken kalmomi.")
            st.warning("⚠️ Ba a sami samfuran da aka horar ba. Ana amfani da bincike na kalmomi (keyword matching) kawai.")
            encoder, classifier = None, None
            encoder_type = "FALLBACK"
            classifier_type = "KEYWORD"
        
        # Model status display
        st.markdown(f" **Samfuri da aka loda:** {encoder_type} + {classifier_type}")
        st.subheader("Bayanan Tsari")
        if qa_data:
            total_pairs = qa_data.get('metadata', {}).get('total_pairs', 0)
            high_conf = qa_data.get('metadata', {}).get('high_confidence_pairs', 0)
            st.metric("Jimlar Tambayoyi", total_pairs)
            st.metric("Tambayoyin da Aka Tabbatar", high_conf)
            st.metric("Matsayin Tabbaci", f"{100*high_conf/total_pairs:.1f}%" if total_pairs > 0 else "Babu")

        if csv_data is not None:
            st.subheader("Rarraba Nau'in Tambayoyi")
            intent_counts = csv_data['Intent'].value_counts()
            st.bar_chart(intent_counts)

        st.divider()

        # About
        st.subheader("Game da HausaTaxBot")
        st.markdown("""
    **Siga:** 2.0  
    **Matsayi:** Tsarin Bincike  
    **Harshe:** Hausa  
    **Fanni:** Harajin Kuɗin Shiga a Najeriya  
    **Sabuntawa ta Ƙarshe:** Mayu 2026  
    Aikin COEN541 - Jami'ar Ahmadu Bello
    """)
        st.info("⚠️ Wannan tsarin na bayar da bayanai ne kawai. Don samun cikakken bayani ko shawara kan haraji, a tuntubi ƙwararren masani kan harkokin haraji.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tattaunawa")
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "bot",
                    "content": "Assalamu alaikum. Ni ne HausaTaxBot. Zan taimaka maka wajen amsa tambayoyi kan haraji a Najeriya (2025/2026). Ta yaya zan taimaka maka?",
                    "metadata": {
                        "source": "Tsari",
                        "confidence": "FARA",
                        "intent": "Gaisuwa",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            ]
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = []
       
        # Clear chat button (professional UX)
        if st.button("Goge Tarihin Tattaunawa", help="Goge duk saƙonni kuma fara daga farko"):
            st.session_state.messages = [st.session_state.messages[0]]  # Keep system greeting
            st.rerun()
        
        # Display chat history with professional features
        chat_container = st.container()
        with chat_container:
            for idx, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class='user-message'>
                        <strong>👤 Kai:</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    metadata = msg.get("metadata", {})
                    confidence = metadata.get("confidence", "N/A")
                    source = metadata.get("source", "")
                    intent = metadata.get("intent", "")
                    
                    col_response, col_actions = st.columns([4, 1])
                    
                    with col_response:
                        st.markdown(f"""
                        <div class='bot-message'>
                            <strong> HausaTaxBot:</strong><br>
                            {msg['content']}<br>
                            <br>
                                <small>
                                    [{confidence}]
                                    | Nau'in: {intent}
                                    | Madogara: {source}
                                </small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_actions:
                        # Copy button
                        if st.button("Kwafi", key=f"copy_{idx}", help="Kwafi zuwa allo"):
                            st.toast("✓ An kwafi saƙon", icon="✅")
                        
                        # Like/Dislike buttons
                        col_like, col_dislike = st.columns(2)
                        with col_like:
                            if st.button("👍", key=f"like_{idx}", help="Amfani"):
                                st.session_state.messages[idx]["feedback"] = "helpful"
                                st.toast("Na gode da ra'ayi!", icon="")
                        with col_dislike:
                            if st.button("👎", key=f"dislike_{idx}", help="Ba amfani"):
                                st.session_state.messages[idx]["feedback"] = "unhelpful"
                                st.toast("Za mu inganta!", icon="")

                st.divider()
        
        # Export buttons (professional feature)
        st.markdown("** Fitar da Tattaunawa**")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            json_export = export_chat_to_json(st.session_state.messages)
            st.download_button(
                label="Sauke JSON",
                data=json_export,
                file_name=f"hausataxbot_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with export_col2:
            csv_export = export_chat_to_csv(st.session_state.messages)
            st.download_button(
                label="Sauke CSV",
                data=csv_export,
                file_name=f"hausataxbot_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        # FAQ Section (professional feature)
        with st.expander("Tambayoyi Akai-Akai (Tambayoyi da Ake Yawan Yi)", expanded=False):
            faq_items = [
                ("Menene VAT?", "VAT (Value Added Tax) — Haraji da aka'a tara akan kayan da aka sayar. Kasuwan zai biya kashi na 7.5% a cikin bangare na kasuwa "),
                ("Wane haraji ne TIN?", "TIN — Tax Identification Number. Lambar da aka ba wa jama'a don tarewa haraji. Duk gida da kasua ya kamata suna wani TIN."),
                ("Kashi wata haraji ne PAYE?", "PAYE — Pay As You Earn. Haraji da ma'aikata suka biya daga albashin su kowane buwan. Haraji na kuɗin shiga ne."),
                ("Shin kasua ƙaramuka suna biyan haraji?", "Eh, Google da Netflix su biya haraji a Najeriya kuma suna baje da kuɗi a kasuwa."),
                ("Yana iya neman tabarma daga haraji?", "Eh! Idan kuke mantuta ku masu kafo ko gida, za ku iya neman tabarma. Dole ne mukamata forom 103 zuwa FIRS."),
                ("Yaya tsarin harajin kuɗin shiga yake?", "Harajin kuɗin shiga na Najeriya (Personal Income Tax - PIT) - shekara 2025: 1% -> 11% -> 15% -> 19% -> 21% — dangane da har kuɗin ku."),
            ]
            
            for question, answer in faq_items:
                with st.expander(f"❔ {question}"):
                    st.write(answer)
        
        st.divider()
        
        # Example queries
        st.subheader("Misalan Tambayoyi")
        example_queries = [
            "Yaya zai shi ne VAT a Najeriya?",
            "Kashi wata za kuɗin shiga za suka biya kamfanoni ƙanana?",
            "Yaya aka lissafa haraji wajen kasuwanci?",
            "Shin Google da Netflix suna biyan haraji?",
            "Shin zan iya biya haraji lokacin da na sayar da gida?",
        ]
        
        cols = st.columns(2)
        for idx, query in enumerate(example_queries):
            with cols[idx % 2]:
                if st.button(f"Ta: {query[:50]}...", key=f"example_{idx}", use_container_width=True):
                    st.session_state.example_query = query
        st.divider()
        
        # Input area
        col_input, col_send = st.columns([8, 2])
        with col_input:
            user_input = st.text_input(
                "Shirya tambayar ku:",
                placeholder="Yaya tambayar ku kan haraji...",
                label_visibility="collapsed"
            )
        
        with col_send:
            send_button = st.button("Aika", use_container_width=True, key="send_btn")
        
        # Handle example query selection
        if "example_query" in st.session_state:
            user_input = st.session_state.example_query
            send_button = True
            del st.session_state.example_query
        
        # Process user input
        if send_button and user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "metadata": {"timestamp": datetime.now().isoformat()}
            })
            
            # Store conversational memory
            st.session_state.conversation_memory.append(user_input)
            
            # Initialize Hausa preprocessor
            preprocessor = create_preprocessor(normalize_diacritics=False, 
                                              remove_stopwords=False,
                                              lowercase=True)
            
            # Show typing indicator (Hausa UI version)
            start_time = time.time()
            with st.spinner("🤖 HausaTaxBot na bincike amsar ka..."):

                # IMPROVED: Use query directly without context pollution
                # (Context can be used for follow-up detection, but not for embedding)
                query_to_use = user_input
                logger.info(f"User query (no context-pollution): '{query_to_use}'")

                # Retrieve answer (core ML inference)
                result = retrieve_answer_v2(
                    query_to_use,
                    qa_data,
                    encoder,
                    classifier,
                    encoder_type,
                    classifier_type,
                    preprocessor=preprocessor
                )

            response_time = time.time() - start_time
            logger.info(f"Retrieval completed in {response_time:.3f}s | Method: {result.get('method', '?')}")
            
            # Handle based on confidence threshold
            if result is not None and result.get('is_confident') and result.get('match'):
                match = result['match']
                response = match.get('answer', 'Ba a iya samun amsa.')
                logger.info(f"CONFIDENT ANSWER | Score: {result['score']:.2%} | Method: {result.get('method')} | Intent: {result.get('intent')}")
                
                bot_message = {
                    "role": "bot",
                    "content": response,
                    "metadata": {
                        "source": match.get('source', 'Bayanan Q&A'),
                        "confidence": result.get('confidence_level', 'UNKNOWN'),  # NEW: Use calibrated level
                        "intent": result.get('intent', 'Ba a sani ba'),
                        "encoder": result.get('encoder', encoder_type),
                        "classifier": result.get('classifier', classifier_type),
                        "method": result.get('method', 'Unknown'),
                        "response_time_ms": f"{response_time*1000:.0f}ms",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                # Low confidence response - constraint to prevent wrong answers
                response = ("⚠️ Ban da isasshen tabbaci don bayar da ingantacciyar amsa ga wannan tambayar. "
                           "Domin samun sahihin bayani kan harkokin haraji, za ka iya ziyartar shafin FIRS: "
                           "https://www.firs.gov.ng Ko kuma ka tuntubi ƙwararren masani kan harkokin haraji.")
                method = result.get('method', 'Unknown') if result else 'Error'
                conf_level = result.get('confidence_level', 'BA TABBATA') if result else 'BA TABBATA'

                bot_message = {
                    "role": "bot",
                    "content": response,
                    "metadata": {
                        "source": "Tsari",
                        "confidence": conf_level,
                        "intent": "Ba a sani ba",
                        "encoder": encoder_type,
                        "classifier": classifier_type,
                        "method": method,
                        "response_time_ms": f"{response_time*1000:.0f}ms",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            st.session_state.messages.append(bot_message)
            st.rerun()
    
    with col2:
        st.subheader("Bayanan Amsa")
        
        # Show metadata from last bot message
        if len(st.session_state.messages) > 1:
            last_bot_msg = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "bot":
                    last_bot_msg = msg
                    break
            
            if last_bot_msg:
                metadata = last_bot_msg.get("metadata", {})
                
                # Confidence display
                conf_level = metadata.get("confidence", "N/A")
                st.metric("Matsayin Tabbaci", conf_level)
                
                st.divider()
                
                st.markdown(f"**Nau'in:** {metadata.get('intent', 'Ba a Sake')}")
                st.markdown(f"**Hanya:** {metadata.get('method', 'Ba a Sake')}")
                st.markdown(f"**Mai ƙirƙira (Encoder):** {metadata.get('encoder', 'Ba a Sake')}")
                st.markdown(f"**Mai rarrabawa (Classifier):** {metadata.get('classifier', 'Ba a Sake')}")
                st.markdown(f"**Madogara:** {metadata.get('source', 'Ba a Sake')}")
                with st.expander("Dalilin Zaɓen Amsa"):
                    st.write(f"Tambayar da ta dace: {metadata.get('matched_question', 'B/A')}")
                    st.write(f"Tabbaci: {metadata.get('confidence', 'B/A')}")
                    st.write(f"Hanya: {metadata.get('method', 'B/A')}")
                
                st.divider()
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Lokaci: {current_time}")
        
        # Additional info
        st.subheader("Saita")
        st.markdown(f"""
        Mai ƙirƙira (Encoder): `{encoder_type}`
        
        Mai rarrabawa (Classifier): `{classifier_type}`
        
        Bayanan:
        - Jimlar Baitin: {qa_data.get('metadata', {}).get('total_pairs', 0)}
        - Harshe: Hausa
        - Fanni: Harajin Kuɗin Shiga ta Najeriya
        """)

if __name__ == "__main__":
    main()