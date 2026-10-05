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

### Project: Trợ lý hỏi đáp quy chế nội bộ (HR/IT Policy Assistant)

Chatbot giúp nhân viên tra cứu chính sách nhân sự, tài chính, IT (nghỉ phép, lương, tạm ứng, mật khẩu, mua sắm...) bằng tiếng Việt, trả lời kèm trích dẫn tài liệu và phiên bản đang hiệu lực. Phát triển tiếp từ pipeline của lab này.

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG — chunk theo đoạn (≤500 ký tự) → dense search bge-m3 top-3 → LLM trả lời. RAGAS đo được: faithfulness 0.84, answer relevancy 0.71, context precision 0.87, context recall 0.85.
- **Vấn đề / Bottlenecks đang gặp:**
  - **Xung đột phiên bản:** tài liệu có nhiều bản (nghỉ phép 2023/2024, mật khẩu v1/v2) — retrieval lấy lẫn bản cũ, hoặc thiếu bản cũ khi người dùng hỏi "có gì thay đổi".
  - **Câu hỏi nhiều ý (multi-hop):** "Senior 9 năm thâm niên được bao nhiêu ngày phép và lương bao nhiêu" — một truy vấn chỉ lấy được tài liệu của một ý.
  - **Câu hỏi tính toán:** phạt tạm ứng quá hạn, hoàn trả chi phí đào tạo — LLM áp đúng quy định nhưng bỏ bước (không quy đổi số ngày quá hạn).
  - **Latency:** rerank bằng bge-reranker-v2-m3 trên CPU mất 6–9 s cho 20 ứng viên, quá chậm cho chat.
  - **PDF scan:** 2/3 file PDF (`BCTC.pdf`, Nghị định 13/2023) không có text layer nên hiện bị bỏ qua.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Structure-aware theo heading (`#`, `##`) làm đơn vị child để giữ trọn từng điều khoản và bảng biểu, kết hợp hierarchical (child → parent = cả tài liệu/chương) khi gửi cho LLM. Không dùng semantic chunking với MiniLM: đã đo 0% cặp câu tiếng Việt đạt threshold 0.85; nếu cần semantic thì dùng bge-m3 và chọn threshold theo percentile phân bố similarity.
2. **Search retrieval:** Hybrid BM25 + dense nhưng **RRF có trọng số** nghiêng về dense (trên dữ liệu lab dense fact-recall@5 = 0.70 > hybrid 0.64 > BM25 0.57); giữ BM25 cho mã văn bản, số hiệu, từ viết tắt (MFA, VPN, P3-P4). Thêm **query decomposition** cho câu hỏi nhiều ý: LLM tách sub-query → search + rerank từng sub-query → gộp context.
3. **Reranking:** Giữ bge-reranker-v2-m3 (Flashrank MultiBERT nhanh hơn nhưng phân biệt kém với tiếng Việt). Giảm latency bằng: chạy trên GPU hoặc bản ONNX/FP16, giảm ứng viên từ 20 xuống ~10, cache kết quả cho câu hỏi lặp lại. Mục tiêu < 500 ms cho bước rerank.
4. **Evaluation:** Xây test set riêng ~50 câu chia nhóm (tra cứu, số liệu, multi-hop, xung đột phiên bản, tính toán). Chạy RAGAS 4 metrics + **answer correctness** (để không phạt nhầm phép tính đúng như faithfulness đang làm). Dùng judge mạnh hơn model sinh câu trả lời, chạy 2–3 lần và báo cáo trung bình ± độ lệch; đưa vào CI để mỗi thay đổi pipeline đều so được với baseline.
5. **Enrichment:** Contextual prepend có truyền toàn văn tài liệu (câu context ghi tên chính sách + phiên bản), HyQA nối vào text index. Bổ sung metadata có cấu trúc `policy_family`, `version`, `effective_date` để: (a) mặc định lọc bản đang hiệu lực, (b) khi người dùng hỏi về thay đổi thì kéo kèm các phiên bản cùng họ. OCR (vd. PaddleOCR/Tesseract tiếng Việt) cho PDF scan trước khi chunk.

#### 3. Timeline triển khai
- **Tuần 1:** Xây test set 50 câu + harness đánh giá (RAGAS + answer correctness, judge cố định, chạy lặp). Chuyển chunking sang structure-aware + hierarchical, thêm metadata phiên bản; OCR 2 file PDF scan. Đo lại baseline trên test set mới.
- **Tuần 2:** RRF có trọng số + query decomposition cho câu hỏi nhiều ý; prompt trả lời từng bước cho câu hỏi tính toán. Tối ưu latency rerank (GPU/ONNX, giảm ứng viên, cache). So sánh từng thay đổi với baseline bằng harness tuần 1, giữ lại thay đổi nào cải thiện thật.
