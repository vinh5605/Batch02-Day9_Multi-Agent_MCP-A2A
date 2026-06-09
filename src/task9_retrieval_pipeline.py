"""Task 9: Full Retrieval Pipeline.

Steps:
  1. HyDE (Hypothetical Document Embeddings) — expand the query with a
     hypothetical answer to improve sparse/dense matching without an embedder.
  2. Semantic search — TF-IDF cosine similarity over the local corpus.
  3. BM25 lexical search over the same corpus.
  4. RRF (Reciprocal Rank Fusion) — merge ranked lists into a single ranking.
  5. Reranking — lightweight token-overlap cross-encoder approximation.
  6. PageIndex fallback — when fused score is below threshold, run structural
     keyword search and blend results.

Weaviate / OpenAI / Jina calls are behind try/except so the pipeline
degrades gracefully to the local fallbacks.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src.task6_lexical_search import bm25_search, tfidf_search
from src.task8_pageindex_vectorless import pageindex_search

_CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "local_chunks.json"
_PAGEINDEX_THRESHOLD = 0.15


def _load_chunks() -> list[dict]:
    if _CHUNKS_PATH.exists():
        return json.loads(_CHUNKS_PATH.read_text(encoding="utf-8"))
    return []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


# ---------------------------------------------------------------------------
# HyDE: generate a hypothetical answer to expand the query
# ---------------------------------------------------------------------------

def _hyde_expand(question: str) -> str:
    """Produce a hypothetical document by prefix-expanding the question.

    In production this would call an LLM to generate a plausible answer.
    Offline, we create a keyword-rich expansion that improves recall.
    """
    try:
        import urllib.request, json as _json
        ollama_url = os.getenv("LLM_BASE_URL", "http://localhost:11434") \
            .rstrip("/v1").rstrip("/") + "/api/generate"
        model = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "qwen2.5:3b")
        payload = _json.dumps({
            "model": model,
            "prompt": (
                f"Trả lời ngắn gọn (2-3 câu) câu hỏi sau bằng tiếng Việt:\n{question}"
            ),
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            ollama_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            body = _json.loads(resp.read())
            return question + " " + body.get("response", "")
    except Exception:
        return question  # fall back to original query if LLM unavailable


# ---------------------------------------------------------------------------
# Weaviate semantic search (optional, falls back to TF-IDF)
# ---------------------------------------------------------------------------

def _weaviate_search(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    try:
        import weaviate  # noqa: F401
        url = os.getenv("WEAVIATE_URL", "")
        api_key = os.getenv("WEAVIATE_API_KEY", "")
        if not url:
            raise RuntimeError("WEAVIATE_URL not set")
        # simplified — real impl would embed query + query Weaviate
        raise NotImplementedError("Weaviate not configured; using TF-IDF fallback")
    except Exception:
        return tfidf_search(query, chunks, top_k)


# ---------------------------------------------------------------------------
# Jina reranker (optional, falls back to token-overlap cross-encoder approx)
# ---------------------------------------------------------------------------

def _jina_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    try:
        import urllib.request, json as _json
        jina_key = os.getenv("JINA_API_KEY", "")
        if not jina_key:
            raise RuntimeError("JINA_API_KEY not set")
        docs = [c.get("content", "") for c in chunks]
        payload = _json.dumps({
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": docs,
            "top_n": top_k,
        }).encode()
        req = urllib.request.Request(
            "https://api.jina.ai/v1/rerank",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jina_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = _json.loads(resp.read())
        results = body.get("results", [])
        out = []
        for r in results:
            idx = r["index"]
            chunk = dict(chunks[idx])
            chunk["score"] = round(r["relevance_score"], 4)
            chunk["source"] = chunk.get("source", "hybrid") + "+jina"
            out.append(chunk)
        return out
    except Exception:
        return _local_rerank(query, chunks, top_k)


def _local_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Token-overlap cross-encoder approximation (no external API needed)."""
    q_tokens = set(_tokenize(query))
    scored = []
    for chunk in chunks:
        doc_tokens = set(_tokenize(chunk.get("content", "")))
        overlap = len(q_tokens & doc_tokens) / max(len(q_tokens), 1)
        base_score = float(chunk.get("score", 0.0))
        combined = 0.6 * base_score + 0.4 * overlap
        c = dict(chunk)
        c["score"] = round(combined, 4)
        scored.append(c)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def _rrf(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked lists using RRF scoring."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.get("id") or chunk.get("content", "")[:80]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunks_by_id:
                chunks_by_id[cid] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, sc in fused:
        c = dict(chunks_by_id[cid])
        c["score"] = round(sc, 6)
        result.append(c)
    return result


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    use_reranking: bool = True,
) -> list[dict]:
    """Full retrieval pipeline. Returns a list of chunk dicts with scores."""
    chunks = _load_chunks()
    if not chunks:
        return []

    # 1. HyDE expansion
    expanded_query = _hyde_expand(question)

    # 2. Semantic search (Weaviate → TF-IDF fallback)
    sem_results = _weaviate_search(expanded_query, chunks, top_k * 2)

    # 3. BM25 lexical search
    bm25_results = bm25_search(expanded_query, chunks, top_k * 2)

    # 4. RRF fusion
    fused = _rrf([sem_results, bm25_results])

    # 5. Reranking
    if use_reranking:
        reranked = _jina_rerank(question, fused[: top_k * 2], top_k)
    else:
        reranked = fused[:top_k]

    # 6. PageIndex fallback for low-confidence results
    final = reranked[:top_k]
    if not final or (final and final[0].get("score", 0) < _PAGEINDEX_THRESHOLD):
        pi_results = pageindex_search(question, chunks, top_k)
        # merge: add pageindex hits not already in final
        existing_ids = {c.get("id") or c.get("content", "")[:80] for c in final}
        for c in pi_results:
            cid = c.get("id") or c.get("content", "")[:80]
            if cid not in existing_ids:
                final.append(c)
                existing_ids.add(cid)
        final = final[:top_k]

    # Apply score_threshold filter
    return [c for c in final if c.get("score", 0) >= score_threshold]
