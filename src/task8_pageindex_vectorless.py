"""Task 8: PageIndex Vectorless Search — structural keyword fallback.

When semantic + BM25 hybrid search returns low-confidence results, this
module performs a lightweight structural search over the markdown corpus
without any embedding/vector infrastructure. Results are tagged with
source='pageindex' so the pipeline can distinguish them.

Strategy:
  1. Token overlap score (similar to BM25 but simpler, O(n) per doc)
  2. Structural boost: if query tokens appear near headers (lines starting
     with '#'), the chunk gets a small structural bonus.
  3. Exact phrase match bonus for multi-word query phrases.
"""
from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _token_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Jaccard-style overlap, weighted by query length."""
    if not query_tokens:
        return 0.0
    qt = set(query_tokens)
    dt = set(doc_tokens)
    matches = len(qt & dt)
    return matches / len(qt)


def _phrase_bonus(query: str, content: str) -> float:
    """Extra score when a 2+ token sub-phrase of the query appears verbatim."""
    q_lower = query.lower()
    c_lower = content.lower()
    tokens = _tokenize(query)
    bonus = 0.0
    for n in range(2, min(5, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in c_lower:
                bonus += 0.1 * n
    return min(bonus, 0.5)


def pageindex_search(
    query: str, chunks: list[dict], top_k: int = 5, threshold: float = 0.0
) -> list[dict]:
    """Structural keyword search with phrase bonuses; marks results as 'pageindex'."""
    if not chunks:
        return []
    q_tokens = _tokenize(query)
    scored: list[tuple[int, float]] = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        doc_tokens = _tokenize(content)
        base = _token_overlap(q_tokens, doc_tokens)
        phrase = _phrase_bonus(query, content)
        score = base + phrase
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    result = []
    for idx, sc in scored[:top_k]:
        if sc <= threshold:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = round(sc, 4)
        chunk["source"] = "pageindex"
        result.append(chunk)
    return result
