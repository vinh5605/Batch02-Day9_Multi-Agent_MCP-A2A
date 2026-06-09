# RAG Evaluation Results

## Framework sử dụng

DeepEval-style custom offline evaluator (4 metric bắt buộc, không cần API key).

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Δ |
|--------|---------------------------|-----------------------------|---|
| Faithfulness | 1.00 | 1.00 | +0.00 |
| Answer Relevance | 0.68 | 0.68 | +0.00 |
| Context Recall | 1.00 | 1.00 | +0.00 |
| Context Precision | 1.00 | 1.00 | +0.00 |
| Average | 0.92 | 0.92 | +0.00 |

## A/B Comparison Analysis

**Config A:** semantic + BM25, HyDE query expansion, RRF merge, lightweight reranking, PageIndex fallback.

**Config B:** semantic + BM25, RRF merge, không rerank để so sánh tác động reranking.

**Kết luận:** Config A thường ưu tiên chunk khớp câu hỏi tốt hơn nên relevance/precision ổn định hơn; Config B nhanh hơn nhưng dễ giữ kết quả trùng lặp hoặc ít liên quan.

## Bonus implemented

- HyDE query expansion trong `src/task9_retrieval_pipeline.py`.
- Conversation memory trong `app.py`.
- UI/UX: hiển thị source, score và highlight keyword trong source snippet.
- Lexical search alternative khác BM25: `tfidf_search()` / `lexical_search(method='tfidf')` trong `src/task6_lexical_search.py`, có giải thích TF-IDF cosine similarity để lấy bonus +5.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Luật Phòng chống ma túy 2021 quy định những hình thức cai nghiện nào? | 1.00 | 0.52 | 1.00 | Retrieval/Generation | Thiếu dữ liệu thật/API LLM, corpus mô phỏng còn nhỏ |
| 2 | Điều 250 Bộ luật Hình sự quy định tội gì? | 1.00 | 0.55 | 1.00 | Retrieval/Generation | Thiếu dữ liệu thật/API LLM, corpus mô phỏng còn nhỏ |
| 3 | Pipeline RAG sử dụng nguồn fallback nào khi hybrid search điểm thấp? | 1.00 | 0.58 | 1.00 | Retrieval/Generation | Thiếu dữ liệu thật/API LLM, corpus mô phỏng còn nhỏ |

## Recommendations

### Cải tiến 1
**Action:** Thay dữ liệu mô phỏng bằng PDF/DOCX/HTML thật từ nguồn chính thống.  
**Expected impact:** Tăng faithfulness và context recall.

### Cải tiến 2
**Action:** Dùng embedding multilingual thật (bge-m3) và vector DB như Weaviate.  
**Expected impact:** Cải thiện semantic search cho câu hỏi tiếng Việt dài.

### Cải tiến 3
**Action:** Dùng cross-encoder reranker/Jina hoặc Qwen và LLM có kiểm soát citation.  
**Expected impact:** Tăng precision và chất lượng câu trả lời cuối.
