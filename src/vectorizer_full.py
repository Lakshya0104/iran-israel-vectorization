"""
FINAL VERSION — ML Assignment (24CYS214)
Dataset: Iran-Israel Tweets (Own Dataset)
Task: 3-class Sentiment Classification

Methods:
- BoW
- TF-IDF
- Word2Vec
- GloVe-like (PPMI + SVD)
- BERT (optional - run separately if needed)
"""

import os, time, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# NLP
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Word2Vec
from gensim.models import Word2Vec

# GloVe-like
from collections import defaultdict
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD

lemmatizer = WordNetLemmatizer()

# =====================================================
# 1. LOAD DATA
# =====================================================
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FILE = os.path.join(BASE_DIR, "data", "iran_israel_tweets.csv")

df = pd.read_csv(DATASET_FILE)
texts = df["text"].tolist()
labels = df["label"].values

print(f"Dataset Loaded: {len(texts)} tweets")

# =====================================================
# 2. PREPROCESSING (WITH LEMMATIZATION ✅)
# =====================================================
STOPWORDS = set([
    "the","is","in","and","to","of","for","on","with","at","by",
    "a","an","this","that","it","rt","amp"
])

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = []
    for word in text.split():
        if word not in STOPWORDS and len(word) > 2:
            word = lemmatizer.lemmatize(word)
            tokens.append(word)

    return tokens

start = time.time()
tokenized = [preprocess(t) for t in texts]
clean_texts = [" ".join(t) for t in tokenized]
print(f"Preprocessing Time: {round(time.time()-start,2)}s")

# Split
X_tr, X_te, y_tr, y_te, tok_tr, tok_te = train_test_split(
    clean_texts, labels, tokenized,
    test_size=0.2, random_state=42, stratify=labels
)

# =====================================================
# HELPER FUNCTION
# =====================================================
def evaluate(name, X_train, X_test):
    print(f"\n=== {name} ===")

    results = {}

    for model_name, model in [
        ("LogReg", LogisticRegression(max_iter=1000)),
        ("SVM", LinearSVC())
    ]:
        start = time.time()
        model.fit(X_train, y_tr)
        train_time = time.time() - start

        pred = model.predict(X_test)

        acc = accuracy_score(y_te, pred)
        f1  = f1_score(y_te, pred, average="macro")

        print(f"{model_name}: Accuracy={acc:.4f}, F1={f1:.4f}, TrainTime={train_time:.2f}s")

        results[model_name] = (acc, f1, train_time)

    return results

# =====================================================
# 3. BAG OF WORDS
# =====================================================
start = time.time()
bow = CountVectorizer(max_features=15000, ngram_range=(1,2))
X_tr_bow = bow.fit_transform(X_tr)
X_te_bow = bow.transform(X_te)
vec_time = time.time() - start

print(f"\nBoW Vectorization Time: {vec_time:.2f}s")
evaluate("BoW", X_tr_bow, X_te_bow)

# =====================================================
# 4. TF-IDF
# =====================================================
start = time.time()
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
X_tr_tfidf = tfidf.fit_transform(X_tr)
X_te_tfidf = tfidf.transform(X_te)
vec_time = time.time() - start

print(f"\nTF-IDF Vectorization Time: {vec_time:.2f}s")
evaluate("TF-IDF", X_tr_tfidf, X_te_tfidf)

# =====================================================
# 5. WORD2VEC
# =====================================================
start = time.time()
w2v = Word2Vec(sentences=tok_tr, vector_size=100, window=5, min_count=2)
train_time = time.time() - start

def avg_vec(tokens):
    vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_tr_w2v = np.array([avg_vec(t) for t in tok_tr])
X_te_w2v = np.array([avg_vec(t) for t in tok_te])

print(f"\nWord2Vec Time: {train_time:.2f}s")
evaluate("Word2Vec", X_tr_w2v, X_te_w2v)

# =====================================================
# 6. GLOVE-LIKE (PPMI + SVD) ✅ RENAMED
# =====================================================
print("\nGloVe-like (PPMI + SVD)")

start = time.time()

freq = defaultdict(int)
for t in tok_tr:
    for w in t:
        freq[w] += 1

vocab = {w:i for i,(w,_) in enumerate(sorted(freq.items(), key=lambda x:-x[1])[:3000])}
V = len(vocab)

cooc = lil_matrix((V, V))

for t in tok_tr:
    ids = [vocab[w] for w in t if w in vocab]
    for i in range(len(ids)):
        for j in range(max(0,i-5), min(len(ids), i+5)):
            if i != j:
                cooc[ids[i], ids[j]] += 1

cooc = cooc.tocsr()
cooc.data = np.log1p(cooc.data)

svd = TruncatedSVD(n_components=100)
emb = svd.fit_transform(cooc)

lookup = {w:emb[i] for w,i in vocab.items()}

def glove_vec(t):
    vecs = [lookup[w] for w in t if w in lookup]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_tr_gl = np.array([glove_vec(t) for t in tok_tr])
X_te_gl = np.array([glove_vec(t) for t in tok_te])

print(f"GloVe-like Time: {time.time()-start:.2f}s")
evaluate("GloVe-like", X_tr_gl, X_te_gl)

# =====================================================
# 7. BERT (OPTIONAL)
# =====================================================
print("\nNOTE: Run BERT separately in Google Colab (recommended)")

print("\n=== DONE ===")