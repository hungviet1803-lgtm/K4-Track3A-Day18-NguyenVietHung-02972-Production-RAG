# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Việt Hùng
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 với `all-MiniLM-L6-v2` tạo **208 chunks** (avg 99 ký tự) so với basic **51 chunks** (avg 410). Đo trên 182 cặp câu liên tiếp: trung vị cosine chỉ 0.62, **0% cặp ≥ 0.85** → thực tế mỗi câu thành 1 chunk. Nguyên nhân: MiniLM là model tiếng Anh, độ tương đồng giữa câu tiếng Việt bị nén thấp. Threshold phải hiệu chỉnh theo model (vd. theo percentile) hoặc dùng model đa ngôn ngữ. |
| Hierarchical (parent-child) | M1 + pipeline | `chunk_hierarchical()`, `run_query()` | 127 children (≤256) / 26 parents. Pipeline search trên child, sau rerank trả **parent** cho LLM (key `(source, parent_id)` vì `parent_0` lặp lại giữa các tài liệu). Context recall tăng 0.85 → 0.925. Vì mỗi tài liệu < 2048 ký tự, parent = cả tài liệu. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks, mỗi chunk = 1 mục `##` kèm `section_path`; bảng lương/bảng mua sắm được giữ nguyên trong 1 chunk thay vì bị cắt như ở child 256 ký tự. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | Đo fact-recall@5 trên test set: BM25 0.57, Dense **0.70**, Hybrid 0.64 — với bộ dữ liệu nhỏ, đồng nhất, RRF trọng số đều bị BM25 kéo xuống. Lowercase + lọc token ký hiệu giúp BM25 từ 0.53 → 0.57. Sau khi `replace("_", " ")`, lợi ích còn lại của underthesea chủ yếu là tách dấu câu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Rerank top-20 hybrid → top-3 nâng fact-recall@3 từ **0.51 → 0.67**. Latency trên CPU (không GPU): ~1.6 s cho 21 đoạn ngắn, 6–9 s cho 20 chunk thật — **vượt xa mục tiêu 150 ms**. Flashrank MultiBERT nhanh hơn (~200 ms) nhưng chấm 0.986 cho đoạn "mật khẩu" với câu hỏi nghỉ phép → không đủ tốt cho tiếng Việt. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production vs baseline: faithfulness 0.843 → 0.916, answer relevancy 0.709 → 0.888, recall 0.850 → 0.925, precision 0.868 → 0.850. Judge flash-lite trả NaN ở 4/20 câu và chấm faithfulness thấp cho câu trả lời có phép tính đúng → điểm RAGAS cần đọc cùng câu trả lời thật, không đọc riêng con số. |
| Contextual embeddings | M5 | `_enrich_single_call()` | 1 call/chunk trả summary + 3 câu hỏi HyQA + câu context + metadata. LLM được xem toàn văn tài liệu nên câu context ghi đúng tên chính sách và phiên bản (vd. "thuộc Chính sách nghỉ phép năm (Phiên bản 2024), mục Số ngày phép năm"). Câu hỏi HyQA được nối vào text index — nếu không, chúng được sinh ra mà không có tác dụng. 127 chunks mất 380 s. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. `openai.RateLimitError: Error code: 429 - ... 'code': 'credit_balance_exhausted'` — key OpenAI hết credit.
  2. `Error code: 404 - This model models/gemini-2.5-flash is no longer available to new users` và `400 - Multiple candidates is not enabled for this model` khi chuyển sang Gemini.
  3. `Enrichment API failed: Expecting property name enclosed in double quotes: line 11 column 5` — 5/127 chunk.
  4. Download model `BAAI/bge-m3` treo ở 0 byte / `peer closed connection without sending complete message body`.
  5. `ModuleNotFoundError: No module named 'pypdf'` khi load PDF.

- **Nguyên nhân gốc rễ & Cách debug:**
  1–2. Chuyển provider sang Gemini qua endpoint tương thích OpenAI (`base_url=.../v1beta/openai/`), gom cấu hình vào `config.py` (`LLM_MODEL`, `make_llm_client()`). Liệt kê model khả dụng bằng `client.models.list()`, test từng model: `gemini-3.8-flash` là model "thinking" nên với `max_tokens` nhỏ toàn bộ token bị dùng cho suy nghĩ → content rỗng; chọn `gemini-3.5-flash-lite`. RAGAS `answer_relevancy` gửi request `n=3` → đặt `strictness=1`. `OpenAIEmbeddings` gửi token id của tiktoken → đặt `check_embedding_ctx_length=False` để gửi text thô.
  3. Gemini json-mode đôi khi để dấu phẩy thừa trước `}`; viết `_loads_lenient()` xóa trailing comma rồi parse lại (retry với temperature 0 sẽ ra cùng output lỗi).
  4. Backend Xet của Hugging Face treo trên Windows → tắt bằng `HF_HUB_DISABLE_XET=1`, đặt `HF_HOME` sang ổ D; mạng chập chờn nhưng `huggingface_hub` tự resume.
  5. Cài lại `requirements.txt`.
  - Bug tự phát hiện khi kiểm tra trên dữ liệu thật (không có test bắt được): pipeline gốc chỉ index child và gửi child cho LLM → hierarchical chunking không phát huy tác dụng; `parent_id` trùng giữa các tài liệu.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Threshold của semantic chunking phụ thuộc phân bố similarity của từng model — học cách đo phân bố trước khi chọn ngưỡng thay vì dùng con số mặc định.
  - Unit test pass không có nghĩa pipeline đúng: tự viết kiểm tra trên dữ liệu thật (không mất nội dung, child nằm trong parent, fact-recall@k) mới thấy hybrid kém dense và semantic chunking suy biến.
  - Metric RAGAS có điểm mù (phạt phép tính đúng, NaN khi judge yếu) — cần đọc từng câu trả lời khi phân tích lỗi.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

> ⚠️ **[Cần tự điền]** — phần này phụ thuộc vào project của bạn. Các gợi ý dưới đây rút ra từ kết quả lab; sửa lại cho khớp với dữ liệu và mục tiêu thực tế.

### Project: [Tên project của bạn]

#### 1. Hiện trạng
- **Pipeline hiện tại:** [Mô tả ngắn kiến trúc RAG đang áp dụng]
- **Vấn đề / Bottlenecks đang gặp:** [Retrieval precision thấp, hallucination, latency cao, ...]

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** [Gợi ý: tài liệu có cấu trúc (quy chế, hướng dẫn) → structure-aware + hierarchical; nếu dùng semantic thì đo phân bố similarity trước để chọn threshold]
2. **Search retrieval:** [Gợi ý: đo dense vs BM25 vs hybrid trên bộ câu hỏi của chính project trước khi chọn; cân nhắc RRF có trọng số]
3. **Reranking:** [Gợi ý: bge-reranker-v2-m3 cho tiếng Việt; cần GPU hoặc giảm số ứng viên nếu yêu cầu latency thấp]
4. **Evaluation:** [Gợi ý: RAGAS 4 metrics + answer correctness cho câu hỏi tính toán; judge đủ mạnh, chạy nhiều lần]
5. **Enrichment:** [Gợi ý: contextual prepend có truyền toàn văn tài liệu; metadata version/ngày hiệu lực nếu tài liệu có nhiều phiên bản]

#### 3. Timeline triển khai
- **Tuần 1:** ...
- **Tuần 2:** ...
