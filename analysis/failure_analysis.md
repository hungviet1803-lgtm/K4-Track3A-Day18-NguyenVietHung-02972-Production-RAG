# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Việt Hùng
**Khóa:** K4 - Track 3A

---

## Cấu hình chạy

| Thành phần | Naive Baseline | Production |
|---|---|---|
| Chunking | `chunk_basic` (paragraph, ≤500 ký tự) | `chunk_hierarchical` — child 256 / parent 2048 |
| Enrichment | — | `_enrich_single_call` (context + HyQA + metadata, 1 call/chunk) |
| Search | Dense only (bge-m3), top-3 | BM25 + Dense → RRF top-20 |
| Rerank | — | bge-reranker-v2-m3 → top-3 child → trả **parent** cho LLM |
| Generation | `gemini-3.5-flash-lite`, prompt gốc | cùng model, temperature 0, prompt ưu tiên phiên bản mới nhất |
| RAGAS judge | `gemini-3.5-flash-lite` + `gemini-embedding-001` | như baseline |

Ghi chú: key OpenAI hết credit nên cả hai pipeline và judge chạy bằng Gemini qua endpoint tương thích OpenAI.
`answer_relevancy` dùng `strictness=1` vì Gemini không hỗ trợ `n>1`. 5/127 chunk bị Gemini trả JSON lỗi
(dấu phẩy thừa) → dùng fallback (đã sửa bằng parser chịu lỗi sau lần chạy này).

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8431 | 0.9164 | +0.0732 |
| Answer Relevancy | 0.7086 | 0.8882 | +0.1796 |
| Context Precision | 0.8684 | 0.8500 | −0.0184 |
| Context Recall | 0.8500 | 0.9250 | +0.0750 |

**Nhận xét chung:**
- Recall tăng nhờ trả về **parent**: mỗi tài liệu trong `data/` < 2048 ký tự nên parent = cả tài liệu → LLM thấy đủ điều khoản liên quan.
- Answer relevancy tăng mạnh nhất (+0.18): prompt production yêu cầu trả lời thẳng, temperature 0.
- Context precision giảm nhẹ: khi câu hỏi chỉ cần 1 tài liệu, top-3 vẫn kéo thêm tài liệu khác (vd. bản 2023 khi hỏi chính sách 2024) và RAGAS chấm các context đó là không hữu ích.
- 4/20 câu có faithfulness = NaN và 2/20 câu có context_precision = NaN (judge flash-lite trả output không parse được); trung bình đã bỏ qua NaN. Điểm này cho thấy judge là một nguồn nhiễu — chênh lệch < 0.02 như context precision không nên coi là có ý nghĩa với 20 câu.

## Bottom-5 Failures

(Xếp theo điểm trung bình 4 metric, lấy từ `reports/ragas_report.json` → `failures`.)

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (chính sách 2024). Lương Senior (P3-P4): 20–35 triệu VNĐ/tháng.
- **Got:** "18 ngày phép (15 ngày cơ bản + 3 ngày thâm niên)… Context không có thông tin về mức lương."
- **Worst metric:** context_precision = 0.0 (recall 0.5, faithfulness 0.83)
- **Error Tree:** Output sai một nửa → Context đúng? **Không** — chỉ có `nghi_phep_nam_v2024` và `nghi_phep_nam_v2023`, thiếu `bang_luong_2024` → Query OK? Câu hỏi gộp 2 ý thuộc 2 tài liệu khác nhau.
- **Root cause:** Câu hỏi multi-hop. Một truy vấn duy nhất bị phần "nghỉ phép / thâm niên" chi phối (nhiều từ khóa hơn), nên cả BM25, dense và reranker đều xếp 2 phiên bản chính sách nghỉ phép lên đầu; tài liệu lương không vào top-3. LLM xử lý đúng: trả lời phần có căn cứ và nói rõ thiếu thông tin lương thay vì bịa.
- **Suggested fix:** M2 — query decomposition: tách câu hỏi thành các sub-query ("số ngày phép 9 năm thâm niên", "khung lương Senior"), search riêng rồi gộp; hoặc tăng top-k sau rerank khi câu hỏi có nhiều ý.

### #2
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Dữ liệu **Bí mật** (cấp 3), cấm chia sẻ với đồng nghiệp; phải mã hóa khi truyền, truy cập theo need-to-know.
- **Got:** "Bí mật"
- **Worst metric:** context_precision = 0.0 (faithfulness NaN, recall 1.0)
- **Error Tree:** Output đúng (nhưng cụt) → Context đúng? **Có** — `phan_loai_du_lieu` và `ky_luong` đều được lấy, recall = 1.0 → Query OK.
- **Root cause:** Không phải lỗi retrieval: recall = 1.0 mà precision = 0.0 là mâu thuẫn — cả 2 context đều chứa đáp án. Đây là lỗi của judge (flash-lite đánh giá "context có giúp tạo ra câu trả lời không" trên một câu trả lời chỉ 2 chữ, faithfulness cũng NaN). Vấn đề thật ở phía generation: câu trả lời quá ngắn, bỏ mất các yêu cầu xử lý (mã hóa, need-to-know) có trong ground truth.
- **Suggested fix:** Prompt (generation) — yêu cầu "trả lời kèm căn cứ / hệ quả liên quan" thay vì chỉ "trả lời thẳng"; M4 — dùng judge mạnh hơn (gpt-4o-mini / gemini flash bản đầy đủ) và chạy nhiều lần để giảm nhiễu.

### #3
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày → quá hạn 5 ngày; phí 2%/tháng × 15.000.000 = 300.000 VNĐ/tháng, pro-rata ≈ 50.000 VNĐ cho 5 ngày.
- **Got:** "15.000.000 × 2% = 300.000 VNĐ/tháng" — không tính số ngày quá hạn.
- **Worst metric:** faithfulness = 0.4 (recall 0.5)
- **Error Tree:** Output sai (thiếu bước) → Context đúng? **Có** — `tam_ung.md` là context duy nhất, chứa điều khoản 15 ngày và 2%/tháng → Query OK.
- **Root cause:** Lỗi suy luận của LLM: áp đúng mức phí nhưng bỏ qua "quá hạn 5 ngày" và không quy đổi theo ngày. Faithfulness thấp vì judge coi phép tính tự suy ra là "không có trong context". Recall 0.5 vì con số 50.000 trong ground truth là kết quả tính, không có trong tài liệu.
- **Suggested fix:** Generation — prompt yêu cầu liệt kê từng bước (thời hạn → số ngày quá hạn → công thức → kết quả); dùng model có reasoning cho câu hỏi tính toán. Đây không phải lỗi của M1/M2/M3.

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết ≥ 1 năm; nghỉ sau 8 tháng → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** Đúng: "8 tháng < 1 năm → hoàn trả 100% chi phí = 25.000.000 VNĐ", có trích phiên bản 1.1.
- **Worst metric:** faithfulness = 0.43 (precision 1.0, recall 1.0)
- **Error Tree:** Output **đúng** → Context đúng (`hoan_chi_dao_tao` xếp đầu) → Query OK.
- **Root cause:** False positive của metric: câu trả lời đúng và có căn cứ, nhưng các mệnh đề suy luận ("8 tháng < 1 năm", "25.000.000 = 100% của 25 triệu") không xuất hiện nguyên văn trong context nên judge đánh "unsupported". Faithfulness của RAGAS phạt mọi phép tính dù đúng.
- **Suggested fix:** M4 — với câu hỏi tính toán, đánh giá bằng answer correctness so với ground truth thay vì chỉ faithfulness; không cần sửa pipeline.

### #5
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành (email, VPN, hệ thống nội bộ). Bản cũ v1.0 không yêu cầu MFA.
- **Got:** "Có, tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ."
- **Worst metric:** context_recall = 0.5
- **Error Tree:** Output đúng → Context đúng? Một phần — có `mat_khau_v2` (xếp đầu) nhưng thiếu `mat_khau_v1`; 2 context còn lại (`phan_loai_du_lieu`, `mua_sam`) không liên quan → Query OK.
- **Root cause:** Bản v1.0 **không nhắc tới MFA**, nên cả BM25 lẫn dense đều không có lý do xếp nó cao cho truy vấn "MFA" — ý "bản cũ không yêu cầu" trong ground truth là thông tin dạng "vắng mặt", retrieval theo độ tương đồng không lấy được. Câu trả lời cho người dùng vẫn đúng.
- **Suggested fix:** M1/M5 — gắn metadata `policy_family` + `version` cho chunk, khi lấy một phiên bản thì kéo kèm các phiên bản cùng họ (metadata filter/expansion) để LLM so sánh được thay đổi.

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. Output đúng? → Đúng một nửa: 18 ngày phép đúng, phần lương trả "không có thông tin".
2. Context đúng? → Không: top-3 sau rerank thuộc 2 tài liệu nghỉ phép (v2024, v2023); `bang_luong_2024.md` bị loại.
3. Query rewrite OK? → Không: một câu hỏi chứa 2 nhu cầu thông tin độc lập; embedding của cả câu nghiêng về chủ đề chiếm nhiều từ hơn.
4. Fix ở bước: **M2 (query side)** — query decomposition trước hybrid search, rerank từng sub-query rồi hợp nhất top-k.

Điểm tích cực: nhờ prompt "chỉ dựa trên context", LLM không bịa khung lương — faithfulness vẫn 0.83. Lỗi lộ ra ở retrieval chứ không phải hallucination, đúng như Diagnostic Tree dự đoán (precision/recall thấp → sửa retrieval).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu hỏi nhiều ý (#1) — ảnh hưởng trực tiếp context recall.
- Prompt generation dạng từng bước cho câu hỏi tính toán (#3), và thêm metric answer correctness để không phạt nhầm phép tính đúng (#4).
- Metadata `version` / `effective_date` để lọc hoặc mở rộng theo phiên bản (#5) — test set có nhiều câu xung đột 2023/2024.
- Chạy RAGAS 2–3 lần với judge mạnh hơn để tách nhiễu của judge (#2, các giá trị NaN) khỏi thay đổi thật của pipeline.
