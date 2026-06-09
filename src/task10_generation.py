"""Task 10: Generation with LLM and citation.

Priority order:
  1. Ollama (local Qwen) via OpenAI-compatible /v1/chat/completions
  2. OpenRouter / any OpenAI-compatible provider (LLM_BASE_URL env var)
  3. Extractive fallback — splice sentences from retrieved chunks

Returns:
  {"answer": str, "sources": list[dict], "llm": str}
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from src.task9_retrieval_pipeline import retrieve

OLLAMA_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "qwen2.5:3b")


def _ollama_available() -> bool:
    """Check whether the configured Ollama endpoint is reachable."""
    try:
        import urllib.request
        base = OLLAMA_BASE_URL.rstrip("/v1").rstrip("/")
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def _call_llm(prompt: str) -> Optional[str]:
    """Call the LLM via OpenAI-compatible chat completion API."""
    try:
        import urllib.request
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "ollama")
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 512,
        }).encode()
        url = OLLAMA_BASE_URL.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _extractive_answer(question: str, chunks: list[dict]) -> str:
    """Rule-based extractive fallback: return the most relevant sentences."""
    if not chunks:
        return "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

    q_tokens = set(re.findall(r"\w+", question.lower()))
    best_chunk = chunks[0]
    content = best_chunk.get("content", "")

    # Pick sentences that overlap most with the query
    sentences = [s.strip() for s in re.split(r"[.!?;]", content) if len(s.strip()) > 20]
    scored = []
    for s in sentences:
        s_tokens = set(re.findall(r"\w+", s.lower()))
        overlap = len(q_tokens & s_tokens)
        scored.append((overlap, s))
    scored.sort(reverse=True)

    top_sentences = [s for _, s in scored[:3] if s]
    answer = ". ".join(top_sentences).strip()
    if not answer:
        answer = content[:400]
    return answer + f"\n\n*(Nguồn: {best_chunk.get('metadata', {}).get('source', 'unknown')})*"


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        src = (c.get("metadata") or {}).get("source", f"Nguồn {i}")
        context_parts.append(f"[{i}] {src}:\n{c.get('content', '')}")

    context = "\n\n".join(context_parts)
    return (
        "Bạn là trợ lý pháp lý chuyên về luật phòng chống ma túy Việt Nam.\n"
        "Dựa vào các đoạn văn bản dưới đây, hãy trả lời câu hỏi một cách chính xác "
        "và có trích dẫn nguồn (dạng [số]).\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời (bằng tiếng Việt, ngắn gọn, có trích dẫn):"
    )


def generate_with_citation(
    question: str,
    context_chunks: Optional[list[dict]] = None,
    top_k: int = 5,
) -> dict:
    """Generate an answer with source citations.

    Args:
        question: The user question.
        context_chunks: Pre-retrieved chunks (skips retrieval if provided).
        top_k: Number of chunks to retrieve (used only when context_chunks is None).

    Returns:
        {"answer": str, "sources": list[dict], "llm": str}
    """
    if context_chunks is None:
        chunks = retrieve(question, top_k=top_k)
    else:
        chunks = context_chunks[:top_k]

    if not chunks:
        return {
            "answer": "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.",
            "sources": [],
            "llm": "none",
        }

    prompt = _build_prompt(question, chunks)
    llm_answer = _call_llm(prompt)

    if llm_answer:
        llm_name = OLLAMA_MODEL
        answer = llm_answer
    else:
        llm_name = "extractive"
        answer = _extractive_answer(question, chunks)

    return {"answer": answer, "sources": chunks, "llm": llm_name}
