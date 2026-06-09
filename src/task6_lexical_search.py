"""Task 6: Lexical Search — BM25 and TF-IDF implementations.

BM25 is the default; TF-IDF cosine similarity is the bonus alternative (+5).

TF-IDF explanation:
  TF(t,d) = count of term t in doc d / total terms in d
  IDF(t)  = log((1 + N) / (1 + df(t))) + 1   (smooth variant)
  Score   = cosine_similarity(TF-IDF(query), TF-IDF(doc))
"""
from __future__ import annotations

import math
import re
from typing import Literal


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class _BM25:
    """Okapi BM25 over a list of string documents."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = [_tokenize(d) for d in docs]
        self.N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(1, self.N)
        # document frequency per term
        self._df: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        dl = len(doc)
        tf_map: dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1
        total = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            idf = self._idf(term)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            total += idf * num / den
        return total


def bm25_search(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Return top-k chunks ranked by BM25 score."""
    if not chunks:
        return []
    texts = [c.get("content", "") for c in chunks]
    bm25 = _BM25(texts)
    q_tokens = _tokenize(query)
    scores = [(i, bm25.score(q_tokens, i)) for i in range(len(chunks))]
    scores.sort(key=lambda x: x[1], reverse=True)
    result = []
    for idx, sc in scores[:top_k]:
        if sc <= 0:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = round(sc, 4)
        chunk["source"] = "bm25"
        result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# TF-IDF (bonus alternative — cosine similarity)
# ---------------------------------------------------------------------------

def _tfidf_vectors(docs: list[list[str]]) -> tuple[list[str], list[dict[str, float]]]:
    """Return (vocab, list of TF-IDF dicts) for the corpus."""
    N = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    vocab = list(df)

    def idf(t: str) -> float:
        return math.log((1 + N) / (1 + df.get(t, 0))) + 1

    vectors = []
    for doc in docs:
        total = max(len(doc), 1)
        tf_map: dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1
        vec = {t: (tf_map.get(t, 0) / total) * idf(t) for t in vocab}
        vectors.append(vec)
    return vocab, vectors


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    dot = sum(v1.get(t, 0.0) * v2.get(t, 0.0) for t in v2)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    denom = norm1 * norm2
    return dot / denom if denom else 0.0


def tfidf_search(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Return top-k chunks ranked by TF-IDF cosine similarity.

    Bonus: uses smooth IDF and full cosine similarity so synonymous terms
    in both query and document raise the score proportionally.
    """
    if not chunks:
        return []
    texts = [c.get("content", "") for c in chunks]
    q_tokens = _tokenize(query)
    # build corpus = query + all docs to share vocabulary/IDF
    all_docs = [q_tokens] + [_tokenize(t) for t in texts]
    _, vecs = _tfidf_vectors(all_docs)
    q_vec = vecs[0]
    scores = [(i, _cosine(q_vec, vecs[i + 1])) for i in range(len(chunks))]
    scores.sort(key=lambda x: x[1], reverse=True)
    result = []
    for idx, sc in scores[:top_k]:
        if sc <= 0:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = round(sc, 4)
        chunk["source"] = "tfidf"
        result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def lexical_search(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    method: Literal["bm25", "tfidf"] = "bm25",
) -> list[dict]:
    """Run lexical search over chunks. method='bm25' (default) or 'tfidf'."""
    if method == "tfidf":
        return tfidf_search(query, chunks, top_k)
    return bm25_search(query, chunks, top_k)
