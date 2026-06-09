"""Supervisor-Workers Pattern — DrugLaw Intel RAG Chatbot

Cải tiến Day 08 Group Project (RAG Pipeline đơn luồng) bằng LangGraph
Supervisor-Workers với 3 workers chuyên biệt:

  Supervisor        : phân loại câu hỏi (keyword-based), routing song song
  StatuteWorker     : tìm điều luật/quy định (BM25 + legal keyword boosting)
  CaseWorker        : tìm vụ án, tin tức, sự kiện thực tế (BM25)
  ContextWorker     : tìm ngữ nghĩa tổng quát (TF-IDF cosine similarity)
  GenerationWorker  : gộp context → sinh câu trả lời có trích dẫn (LLM / extractive)

Graph topology:
  START
    │
    ▼
  supervisor ──── classify question ──── sets query_type
    │
    ├─ [statute / all] ──► statute_worker ─────┐
    ├─ [case    / all] ──► case_worker    ─────┤
    └─ [general / all] ──► context_worker ─────┤
                                               ▼
                                      generation_worker
                                               │
                                              END

Drop-in replacement cho generate_with_citation() trong web_app.py:
    from Lab_Assignment.Supervisor_Workers import answer_question
    result = answer_question(question)
    # {"answer": str, "sources": list[dict], "llm": str, "query_type": str}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# ── sys.path: import src.* từ project root ───────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.task6_lexical_search import bm25_search, tfidf_search
from src.task10_generation import generate_with_citation


# ═══════════════════════════════════════════════════════════════════════════════
# Corpus helper
# ═══════════════════════════════════════════════════════════════════════════════

_CHUNKS_PATH = _ROOT / "data" / "local_chunks.json"


def _load_chunks() -> list[dict]:
    """Load corpus từ JSON file (seed data của group project Day 08)."""
    if _CHUNKS_PATH.exists():
        return json.loads(_CHUNKS_PATH.read_text(encoding="utf-8"))
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Shared State
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_chunks(a: list, b: list) -> list:
    """Reducer: gộp 2 danh sách chunks, loại trùng lặp theo 80 ký tự đầu."""
    seen = {c.get("content", "")[:80] for c in a}
    result = list(a)
    for item in b:
        key = item.get("content", "")[:80]
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


class DrugLawState(TypedDict):
    """State dùng chung cho toàn bộ graph."""

    question: str

    # Supervisor output
    query_type: str                                       # statute | case | general | all

    # Worker outputs — dùng Annotated để parallel branches có thể ghi đồng thời
    statute_chunks: Annotated[list, _merge_chunks]        # StatuteWorker
    case_chunks:    Annotated[list, _merge_chunks]        # CaseWorker
    context_chunks: Annotated[list, _merge_chunks]        # ContextWorker

    # Generation output
    final_answer: str
    sources:      list[dict]
    llm_used:     str


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor
# ═══════════════════════════════════════════════════════════════════════════════

# Từ khóa phân loại câu hỏi (không cần LLM — nhanh, luôn hoạt động offline)
_STATUTE_KW: frozenset[str] = frozenset({
    "điều", "khoản", "điểm", "nghị định", "thông tư", "luật",
    "hình phạt", "tội", "quy định", "xử phạt", "chế tài",
    "phạt tù", "bị truy cứu", "tội danh",
})
_CASE_KW: frozenset[str] = frozenset({
    "vụ án", "vụ bắt", "bắt giữ", "triệt phá", "khởi tố",
    "tin tức", "tử hình", "cảnh sát", "công an", "xét xử",
    "bị cáo", "bản án", "phiên tòa",
})
_DEF_KW: frozenset[str] = frozenset({
    "là gì", "định nghĩa", "khái niệm", "nghĩa là",
    "phân loại", "các loại", "ví dụ", "gồm những gì",
})


def supervisor(state: DrugLawState) -> dict:
    """Supervisor node: phân loại câu hỏi và xác định query_type."""
    q = state["question"].lower()

    has_statute = any(kw in q for kw in _STATUTE_KW)
    has_case    = any(kw in q for kw in _CASE_KW)
    has_def     = any(kw in q for kw in _DEF_KW)

    if has_statute and has_case:
        query_type = "all"
    elif has_statute:
        query_type = "statute"
    elif has_case:
        query_type = "case"
    elif has_def:
        query_type = "general"
    else:
        query_type = "all"   # câu hỏi không rõ → dùng cả 3 workers

    print(f"\n  [Supervisor] '{state['question'][:60]}...' → query_type={query_type!r}")
    return {"query_type": query_type}


def route_to_workers(state: DrugLawState) -> list[Send]:
    """Routing function: dispatch các workers phù hợp chạy song song.

    Trả về list[Send] — LangGraph sẽ chạy tất cả đồng thời.
    """
    qt = state.get("query_type", "all")
    sends: list[Send] = []

    if qt in ("statute", "all"):
        sends.append(Send("statute_worker", state))
    if qt in ("case", "all"):
        sends.append(Send("case_worker", state))
    if qt in ("general", "all"):
        sends.append(Send("context_worker", state))

    # Fallback an toàn
    if not sends:
        sends = [
            Send("statute_worker", state),
            Send("case_worker", state),
            Send("context_worker", state),
        ]

    print(f"  [Supervisor] Dispatching to: {[s.node for s in sends]}")
    return sends


# ═══════════════════════════════════════════════════════════════════════════════
# Worker 1 — StatuteWorker
# ═══════════════════════════════════════════════════════════════════════════════

_LEGAL_BOOST_TERMS = [
    "luật", "điều", "khoản", "nghị định", "ma túy",
    "hình phạt", "chất ma túy", "tiền chất",
]
_STATUTE_MARKERS = ("điều", "khoản", "nghị định", "thông tư", "luật số")


def statute_worker(state: DrugLawState) -> dict:
    """StatuteWorker: tìm điều luật/quy định bằng BM25 với legal keyword boosting.

    Chiến lược:
    1. Thêm legal keywords vào query để boosting
    2. BM25 search trên toàn bộ corpus
    3. Ưu tiên chunks chứa dấu hiệu điều luật ("điều X", type=="law")
    """
    print("  [StatuteWorker] Searching legal statutes...")
    chunks = _load_chunks()
    if not chunks:
        return {"statute_chunks": []}

    # Keyword boosting: chỉ thêm terms đã có trong câu hỏi để không lạc ngữ nghĩa
    boost = " ".join(kw for kw in _LEGAL_BOOST_TERMS if kw in state["question"].lower())
    boosted_query = (state["question"] + " " + boost).strip()

    results = bm25_search(boosted_query, chunks, top_k=6)

    # Sắp xếp lại: ưu tiên chunks chứa statute markers
    statute_hits = [
        c for c in results
        if any(m in c.get("content", "").lower() for m in _STATUTE_MARKERS)
        or (c.get("metadata") or {}).get("type") == "law"
    ]
    seen = {c.get("content", "")[:80] for c in statute_hits}
    for c in results:
        if c.get("content", "")[:80] not in seen:
            statute_hits.append(c)
            seen.add(c.get("content", "")[:80])

    hits = statute_hits[:3]
    for c in hits:
        c["worker"] = "statute_worker"

    print(f"  [StatuteWorker] → {len(hits)} statute chunks")
    return {"statute_chunks": hits}


# ═══════════════════════════════════════════════════════════════════════════════
# Worker 2 — CaseWorker
# ═══════════════════════════════════════════════════════════════════════════════

_CASE_MARKERS = (
    "vụ", "bắt", "triệt phá", "khởi tố", "xét xử",
    "tử hình", "bị cáo", "phiên tòa", "công an",
)


def case_worker(state: DrugLawState) -> dict:
    """CaseWorker: tìm vụ án, tin tức, sự kiện thực tế bằng BM25.

    Chiến lược:
    1. Thêm context terms về vụ án vào query
    2. BM25 search
    3. Ưu tiên chunks có dấu hiệu tin tức/vụ án hoặc metadata type==news/case
    """
    print("  [CaseWorker] Searching case reports and news...")
    chunks = _load_chunks()
    if not chunks:
        return {"case_chunks": []}

    case_query = state["question"] + " vụ án tội phạm ma túy bắt giữ"
    results = bm25_search(case_query, chunks, top_k=6)

    case_hits = [
        c for c in results
        if (c.get("metadata") or {}).get("type") in ("news", "case")
        or any(m in c.get("content", "").lower() for m in _CASE_MARKERS)
    ]
    seen = {c.get("content", "")[:80] for c in case_hits}
    for c in results:
        if c.get("content", "")[:80] not in seen:
            case_hits.append(c)
            seen.add(c.get("content", "")[:80])

    hits = case_hits[:3]
    for c in hits:
        c["worker"] = "case_worker"

    print(f"  [CaseWorker] → {len(hits)} case chunks")
    return {"case_chunks": hits}


# ═══════════════════════════════════════════════════════════════════════════════
# Worker 3 — ContextWorker
# ═══════════════════════════════════════════════════════════════════════════════

def context_worker(state: DrugLawState) -> dict:
    """ContextWorker: tìm ngữ cảnh tổng quát bằng TF-IDF cosine similarity.

    Chiến lược:
    - TF-IDF tốt hơn BM25 với câu hỏi mang ngữ nghĩa (định nghĩa, giải thích)
    - Không cần boosting — dùng câu hỏi gốc để giữ ngữ nghĩa trung thực
    """
    print("  [ContextWorker] Searching with TF-IDF semantic similarity...")
    chunks = _load_chunks()
    if not chunks:
        return {"context_chunks": []}

    results = tfidf_search(state["question"], chunks, top_k=4)

    for c in results:
        c["worker"] = "context_worker"

    print(f"  [ContextWorker] → {len(results)} context chunks")
    return {"context_chunks": results}


# ═══════════════════════════════════════════════════════════════════════════════
# GenerationWorker
# ═══════════════════════════════════════════════════════════════════════════════

def generation_worker(state: DrugLawState) -> dict:
    """GenerationWorker: gộp output của tất cả workers → sinh câu trả lời.

    Thứ tự ưu tiên context:
      1. statute_chunks (điều luật — quan trọng nhất)
      2. case_chunks    (vụ án — minh họa thực tế)
      3. context_chunks (ngữ cảnh tổng quát)

    Sau khi gộp + dedup + sort theo score → gọi generate_with_citation()
    (Ollama/Qwen → extractive fallback nếu LLM không có sẵn)
    """
    print("  [GenerationWorker] Merging context and generating answer...")

    # Gộp tất cả chunks, dedup, sort theo score
    all_chunks: list[dict] = []
    seen: set[str] = set()

    for chunk_list in (
        state.get("statute_chunks") or [],
        state.get("case_chunks")    or [],
        state.get("context_chunks") or [],
    ):
        for c in chunk_list:
            key = c.get("content", "")[:80]
            if key not in seen:
                all_chunks.append(c)
                seen.add(key)

    all_chunks.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    if all_chunks:
        result = generate_with_citation(
            state["question"],
            context_chunks=all_chunks[:5],
        )
    else:
        # Không có chunk nào → fallback full retrieval pipeline
        print("  [GenerationWorker] No chunks available, falling back to full pipeline...")
        result = generate_with_citation(state["question"], top_k=3)

    # Chuẩn hóa sources
    sources: list[dict] = []
    for s in (result.get("sources") or [])[:5]:
        md = s.get("metadata") or {}
        sources.append({
            "source": md.get("source") or md.get("path") or "unknown",
            "score":  round(float(s.get("score", 0)), 4),
            "type":   md.get("type", "unknown"),
            "worker": s.get("worker", ""),
        })

    print(f"  [GenerationWorker] Done — LLM: {result.get('llm')}, "
          f"chunks used: {len(all_chunks)}")

    return {
        "final_answer": result.get("answer", ""),
        "sources":      sources,
        "llm_used":     result.get("llm", "unknown"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════════

def create_graph():
    """Build và compile Supervisor-Workers StateGraph.

    Topology:
      START → supervisor → [conditional parallel] → workers → generation_worker → END
    """
    graph = StateGraph(DrugLawState)

    # Nodes
    graph.add_node("supervisor",        supervisor)
    graph.add_node("statute_worker",    statute_worker)
    graph.add_node("case_worker",       case_worker)
    graph.add_node("context_worker",    context_worker)
    graph.add_node("generation_worker", generation_worker)

    # Entry
    graph.add_edge(START, "supervisor")

    # Supervisor → workers (parallel dispatch qua Send API)
    graph.add_conditional_edges(
        "supervisor",
        route_to_workers,
        ["statute_worker", "case_worker", "context_worker"],
    )

    # Workers → generation (hội tụ)
    graph.add_edge("statute_worker",    "generation_worker")
    graph.add_edge("case_worker",       "generation_worker")
    graph.add_edge("context_worker",    "generation_worker")

    # Generation → END
    graph.add_edge("generation_worker", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — drop-in cho web_app.py
# ═══════════════════════════════════════════════════════════════════════════════

_compiled_graph = None


def answer_question(question: str) -> dict:
    """Thay thế generate_with_citation() trong web_app.py.

    Sử dụng trong web_app.py:
        from Lab_Assignment.Supervisor_Workers import answer_question
        result = answer_question(question)

    Returns:
        {
            "answer":     str,
            "sources":    list[dict],   # [{source, score, type, worker}]
            "llm":        str,          # "qwen2.5:3b" | "extractive"
            "query_type": str,          # "statute" | "case" | "general" | "all"
        }
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_graph()

    state_result = _compiled_graph.invoke({
        "question":       question,
        "query_type":     "",
        "statute_chunks": [],
        "case_chunks":    [],
        "context_chunks": [],
        "final_answer":   "",
        "sources":        [],
        "llm_used":       "",
    })

    return {
        "answer":     state_result["final_answer"],
        "sources":    state_result["sources"],
        "llm":        state_result["llm_used"],
        "query_type": state_result["query_type"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Demo — chạy trực tiếp
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    TEST_QUESTIONS = [
        # Câu hỏi về điều luật → StatuteWorker
        "Điều 249 Bộ luật Hình sự quy định hình phạt gì cho tội tàng trữ chất ma túy?",
        # Câu hỏi về vụ án → CaseWorker
        "Gần đây có vụ bắt giữ ma túy lớn nào không?",
        # Câu hỏi định nghĩa → ContextWorker
        "Ma túy tổng hợp là gì? Gồm những loại nào?",
        # Câu hỏi tổng quát → cả 3 workers
        "Hình phạt và các vụ án về tội vận chuyển heroin ở Việt Nam?",
    ]

    print("=" * 70)
    print("DrugLaw Intel — Supervisor-Workers Pattern Demo")
    print("3 Workers: StatuteWorker | CaseWorker | ContextWorker")
    print("=" * 70)

    for q in TEST_QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"Q: {q}")
        print("─" * 70)

        r = answer_question(q)

        print(f"\nQuery type : {r['query_type']}")
        print(f"LLM used   : {r['llm']}")
        print(f"\nAnswer:\n{r['answer']}")

        if r["sources"]:
            print(f"\nSources ({len(r['sources'])}):")
            for i, s in enumerate(r["sources"][:3], 1):
                print(f"  {i}. {s['source']}  "
                      f"score={s['score']:.3f}  "
                      f"worker={s.get('worker', '?')}")

    print("\n" + "=" * 70)
    print("So sánh với Day 08 (single pipeline):")
    print("  Trước: generate_with_citation(question)  — 1 luồng tuần tự")
    print("  Sau:   answer_question(question)          — Supervisor + 3 Workers song song")
    print("  Lợi ích: routing thông minh, mỗi worker chuyên biệt, dễ mở rộng thêm workers")
    print("=" * 70)
