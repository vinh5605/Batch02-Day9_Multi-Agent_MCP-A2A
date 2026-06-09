# Lab 9 — Multi-Agent System with MCP & A2A Protocol
## Báo Cáo Tổng Hợp

**Học viên:** Vũ Ngọc Vinh
**Ngày hoàn thành:** 09/06/2026  
**Môi trường:** Python 3.14, Windows 11, uv package manager  
**LLM Backend:** OpenRouter (`openai/gpt-oss-120b:free`) qua OpenAI-compatible API

---

## 1. Tổng quan dự án

Lab này xây dựng một **hệ thống tư vấn pháp lý đa tác nhân** theo kiến trúc A2A (Agent-to-Agent) Protocol của Google. Hệ thống gồm hai phần:

- **Exercises (Bài Tập 2 & 4):** Thực hành kỹ năng cốt lõi — Tool Use và Multi-Agent với LangGraph.
- **Group Project:** Hệ thống phân tán 5 dịch vụ với RAG pipeline, web app, và evaluation pipeline.

---

## 2. Bài Tập 2 — Tool Use và Knowledge Base

**File:** `exercises/exercise_2_tools.py`

### Những gì đã làm:

**Bài Tập 2.1 — Thêm Knowledge Base Entry:**
Bổ sung mục `labor_law` vào `LEGAL_KNOWLEDGE`:
```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": "Theo Bộ luật Lao động Việt Nam 2019..."
}
```

**Bài Tập 2.2 — Thêm Tool mới:**
Cài đặt `check_statute_of_limitations(case_type: str)` — tra cứu thời hiệu khởi kiện:
```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

**Luồng xử lý:**
```
User question → LLM (bind_tools) → Tool calls → ToolMessages → LLM synthesizes → Final answer
```

---

## 3. Bài Tập 4 — Multi-Agent với Privacy Agent

**File:** `exercises/exercise_4_multiagent.py`

### Những gì đã làm:

**Thêm `privacy_agent` node** chuyên về GDPR/CCPA/luật bảo vệ dữ liệu cá nhân.

**Thêm field `privacy_analysis`** vào State (với `Annotated[str, _last_wins]` reducer để parallel branches không xung đột).

**Cập nhật `check_routing`** với keyword routing cho privacy:
```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
    tasks.append(Send("privacy_agent", state))
```

**Graph topology (6 nodes):**
```
START → law_agent → check_routing → [tax_agent | compliance_agent | privacy_agent]
                                              ↓              ↓              ↓
                                          aggregate_results → END
```

**Cơ chế `Send` API:** Parallel dispatch với LangGraph `Send` — tất cả specialist agents chạy song song, sau đó `aggregate_results` tổng hợp.

---

## 4. Group Project — Legal RAG System

### 4.1 Kiến trúc tổng thể

```
group_project/
├── web_app.py              # FastAPI web interface (upload, chat, ingest)
├── app.py                  # Entry point
├── evaluation/
│   ├── eval_pipeline.py    # A/B evaluation: with/without reranking
│   └── golden_dataset.json # 5 câu hỏi pháp lý + expected answers
└── ...

src/
├── task6_lexical_search.py     # BM25 + TF-IDF
├── task8_pageindex_vectorless.py # Structural keyword fallback
├── task9_retrieval_pipeline.py  # Full hybrid pipeline
├── task10_generation.py         # LLM generation with citations
└── ingestion.py                 # Document ingestion

data/
└── local_chunks.json  # 15 seed chunks (BLHS 2015, Luật PCMT 2021, news 2024-2026)
```

### 4.2 Task 6 — Lexical Search

**File:** `src/task6_lexical_search.py`

Cài đặt hai thuật toán:

**BM25 (Okapi BM25)** — scoring chính:
```
score(d,q) = Σ IDF(t) × [tf(t,d)×(k1+1)] / [tf(t,d) + k1×(1-b+b×|d|/avgdl)]
```
- `k1 = 1.5`, `b = 0.75` (tham số chuẩn)
- Penalize documents ngắn/dài hơn trung bình

**TF-IDF với Cosine Similarity** (bonus +5):
- Vector hóa corpus → cosine similarity với query vector
- Giải thích sự khác biệt TF-IDF vs BM25: BM25 có term saturation (tf không tăng vô hạn), TF-IDF thì không

### 4.3 Task 8 — PageIndex (Vectorless Fallback)

**File:** `src/task8_pageindex_vectorless.py`

Structural keyword search dùng làm fallback khi BM25/TF-IDF score thấp:
- Token overlap ratio
- Phrase bonus (+0.3 nếu tìm thấy exact phrase trong chunk)
- Ngưỡng mặc định `threshold=0.05`
- Tag kết quả với `source='pageindex'`

### 4.4 Task 9 — Full Retrieval Pipeline

**File:** `src/task9_retrieval_pipeline.py`

Pipeline 6 bước:

```
Query → HyDE → TF-IDF Search → BM25 Search → RRF Fusion → Reranking → [PageIndex fallback]
```

**HyDE (Hypothetical Document Embeddings):**
- Gọi LLM (timeout 2s) sinh hypothetical answer để expand query
- Fallback: keyword-based expansion nếu LLM unavailable

**RRF (Reciprocal Rank Fusion):**
```
RRF(d) = Σ 1 / (k + rank(d))  với k = 60
```
Merge ranked lists từ TF-IDF và BM25 không cần normalize scores.

**Reranking:**
- Jina Reranker API (nếu có key)
- Fallback: token overlap cross-encoder approximation

**PageIndex fallback:** Kích hoạt khi max RRF score < `_PAGEINDEX_THRESHOLD` (0.15)

### 4.5 Task 10 — Generation with Citation

**File:** `src/task10_generation.py`

- Thử Ollama local trước, fallback extractive answer
- Trả về `{"answer": str, "sources": list[dict], "llm": str}`
- Sources có `chunk_id`, `content`, `score`, `source`

### 4.6 Web App

**File:** `group_project/web_app.py`

FastAPI với các endpoint:
- `GET /` — Chat UI
- `POST /chat` — RAG-powered chat
- `POST /upload` — Upload PDF/DOCX/TXT
- `POST /ingest-url` — Ingest from URL
- `GET /health` — Health check

**Sửa lỗi Python 3.14:** Module `cgi` bị xóa khỏi stdlib. Thay thế `cgi.FieldStorage` bằng manual multipart boundary parser:
```python
boundary_match = _re.search(r"boundary=([^\s;]+)", ct)
boundary = boundary_match.group(1).encode()
parts = raw_body.split(b"--" + boundary)
```

### 4.7 Evaluation Pipeline

**File:** `group_project/evaluation/eval_pipeline.py`

A/B evaluation trên 4 metrics (DeepEval-style, không cần API key):

| Metric | Mô tả |
|--------|-------|
| **Faithfulness** | Câu trả lời có được hỗ trợ bởi context không |
| **Answer Relevance** | Câu trả lời có liên quan đến câu hỏi không |
| **Context Recall** | Context có chứa thông tin cần thiết không |
| **Context Precision** | Tỷ lệ context thực sự hữu ích |

So sánh 2 config: `hybrid+reranking` vs `hybrid without reranking`

---

## 5. Stage 5 — Distributed A2A System

### 5.1 Kiến trúc

```
┌──────────────────────────────────────────────────────────┐
│                   Registry Service                        │
│              http://localhost:10000                        │
│   POST /register | GET /discover/{task} | GET /agents     │
└─────────────────────┬────────────────────────────────────┘
                      │ service discovery
         ┌────────────┼────────────────────┐
         ↓            ↓                    ↓
  Customer Agent   Law Agent          Tax | Compliance
  :10100           :10101             :10102 | :10103
  (ReAct)          (StateGraph)       (ReAct)  (ReAct)
```

**Luồng xử lý đầy đủ:**
```
User → Customer Agent → [discovers Law Agent] → Law Agent
                                                    ├── analyze_law (LLM)
                                                    ├── check_routing (LLM)
                                                    ├── call_tax ──→ Tax Agent (parallel)
                                                    ├── call_compliance → Compliance Agent (parallel)
                                                    └── aggregate (LLM) → final answer
```

### 5.2 Service Registry

**File:** `registry/__main__.py`

FastAPI service — in-memory store, 3 endpoints:
```python
POST /register    # agent self-registers với tasks list
GET /discover/{task}  # tìm agent xử lý task
GET /agents       # list tất cả agents đã đăng ký
```

### 5.3 Common Layer

**`common/llm.py`:** `get_llm()` trả về `ChatOpenAI` với `temperature=0.3`. Hỗ trợ `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.

**`common/registry_client.py`:** `discover(task)` — async HTTP call đến registry để resolve endpoint.

**`common/a2a_client.py`:** `delegate(endpoint, question, ...)` — gửi câu hỏi đến A2A agent khác và đọc response qua legacy JSON-RPC transport. Timeout: 900 giây.

### 5.4 Law Agent — LangGraph StateGraph

**File:** `law_agent/graph.py`

```
analyze_law → check_routing → route_to_subagents() → [call_tax | call_compliance] → aggregate → END
```

`route_to_subagents()` sử dụng **LangGraph `Send` API** để dispatch song song:
```python
def route_to_subagents(state: LawState) -> list[Send]:
    sends = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    return sends or [Send("aggregate", state)]
```

**Sửa lỗi deprecation:** `from langgraph.constants import Send` → `from langgraph.types import Send`

### 5.5 Customer Agent — ReAct

**File:** `customer_agent/graph.py`

`create_react_agent` với tool `delegate_to_legal_agent` — discover law agent qua registry rồi gọi A2A.

### 5.6 Tax & Compliance Agents

Mỗi agent là một `create_react_agent` chuyên biệt với system prompt riêng. Cả hai là leaf nodes — không delegate tiếp.

### 5.7 A2A Protocol

Mỗi agent serve:
- `GET /.well-known/agent.json` — Agent Card (tên, version, capabilities)
- `POST /` — JSON-RPC endpoint nhận/trả message

Metadata truyền giữa agents:
```python
metadata = {
    "trace_id": str,       # UUID sinh tại Customer Agent, propagate xuyên suốt
    "context_id": str,     # A2A context ID
    "delegation_depth": int  # Chống vòng lặp vô hạn (max=3)
}
```

### 5.8 Windows Scripts

**`start_all.ps1`:** Khởi động 5 services theo thứ tự (Registry → Tax+Compliance → Law → Customer), ghi log vào `logs/`.

**`stop_all.ps1`:** Dừng bằng PID file hoặc fallback kill by port.

---

## 6. Kết quả kiểm thử

### 6.1 Unit Check (19/19 passed)

Script `_check_stage5.py` kiểm tra:
- Syntax tất cả 17 Python files
- Import a2a-sdk core classes
- `get_llm()` trả về đúng temperature
- Registry FastAPI app có đầy đủ endpoints
- `route_to_subagents()` dispatch đúng nodes
- `build_graph()` định nghĩa đúng

### 6.2 End-to-End Test

**Câu hỏi:** *"If a company breaks a contract and avoids taxes, what are the legal and regulatory consequences?"*

**Kết quả:** Hệ thống trả về phân tích pháp lý đầy đủ gồm:

- Bảng phân biệt Contract Breach vs Tax Avoidance/Evasion
- Compensatory damages, consequential damages, specific performance
- Civil penalties (20-40% underpayment), TFRP (100% cá nhân), criminal prosecution (5 năm tù)
- SEC/antitrust exposure với public companies
- Checklist hành động khẩn cấp (internal investigation, voluntary IRS disclosure, etc.)

**Timeline thực tế:**
```
16:00:25  Customer Agent nhận request
16:00:21  Customer LLM routing → delegate to Law Agent
16:00:25  Law Agent nhận request
16:00:27  analyze_law LLM hoàn thành
16:02:58  check_routing LLM hoàn thành (needs_tax=True, needs_compliance=True)
16:03:01  Discover Tax + Compliance agents
16:03:01  Dispatch SONG SONG → Tax Agent + Compliance Agent
16:05:11  Tax Agent trả về 12,424 chars ✅
16:05:18  Compliance Agent trả về 12,499 chars ✅
16:05:22  aggregate LLM tổng hợp → final answer ✅
~16:05:25 Customer Agent nhận response → trả về User ✅
```

**Tổng thời gian:** ~5 phút (do free LLM model chậm)

### 6.3 Các vấn đề đã giải quyết

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| `cgi` module missing | Python 3.14 xóa stdlib `cgi` | Manual multipart boundary parser |
| `Send` deprecation warning | `langgraph.constants.Send` deprecated | Dùng `langgraph.types.Send` |
| ReadTimeout khi test | Free LLM chậm, chain mất ~290s, timeout 300s | Tăng `a2a_client.py` lên 900s, `test_client.py` lên 1200s |
| UnicodeEncodeError trên Windows | Windows console cp1252 không encode `‑` | Redirect stdout sang utf-8 wrapper |
| OpenRouter 402/429 | Key không có credit; model rate-limited | Dùng `openai/gpt-oss-120b:free` |

---

## 7. Cấu hình môi trường

**`.env`:**
```ini
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-oss-120b:free
REGISTRY_URL=http://localhost:10000
```

**Khởi động hệ thống (Windows):**
```powershell
.\start_all.ps1         # Khởi động 5 services
uv run python test_client.py  # Chạy end-to-end test
.\stop_all.ps1          # Dừng tất cả
```

---

## 8. Tổng kết

| Hạng mục | Trạng thái |
|----------|-----------|
| Exercise 2 — Tool Use (knowledge base + tools) | ✅ Hoàn thành |
| Exercise 4 — Multi-Agent + Privacy Agent | ✅ Hoàn thành |
| Group Project — Task 6 BM25 + TF-IDF | ✅ Hoàn thành |
| Group Project — Task 8 PageIndex fallback | ✅ Hoàn thành |
| Group Project — Task 9 Hybrid RAG Pipeline | ✅ Hoàn thành |
| Group Project — Task 10 Generation + Citation | ✅ Hoàn thành |
| Group Project — Web App | ✅ Hoàn thành |
| Group Project — Evaluation Pipeline | ✅ Hoàn thành |
| Stage 5 — Service Registry | ✅ Hoàn thành |
| Stage 5 — Customer Agent (ReAct + A2A) | ✅ Hoàn thành |
| Stage 5 — Law Agent (StateGraph + parallel Send) | ✅ Hoàn thành |
| Stage 5 — Tax Agent (ReAct) | ✅ Hoàn thành |
| Stage 5 — Compliance Agent (ReAct) | ✅ Hoàn thành |
| Stage 5 — End-to-end test passed | ✅ Hoàn thành |

**Hệ thống đã chạy thành công:** 5 microservices giao tiếp qua A2A Protocol, Law Agent điều phối Tax và Compliance song song, toàn bộ chain trả về phân tích pháp lý toàn diện cho câu hỏi phức tạp.
