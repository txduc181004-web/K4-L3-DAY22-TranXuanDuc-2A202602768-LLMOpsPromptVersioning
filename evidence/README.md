# Evidence & phân tích — Day 22 (Trần Xuân Đức · 2A202602768)

Provider: OpenAI `gpt-4o-mini` + `text-embedding-3-small` · LangSmith project: `day22-tranxuanduc`

## Danh sách tệp

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | Danh sách traces `rag-query` trên LangSmith (≥ 50) |
| `01_rag_pipeline_log.txt` | Log chạy Bước 1 (50 Q/A) — bổ sung |
| `02_prompt_hub.png` | 2 prompt `tran-xuan-duc-rag-prompt-v1` / `-v2` trên Prompt Hub |
| `02_prompt_hub_detail.png` | Chi tiết prompt V2 trên Hub: system prompt có `{context}`, commit history — bổ sung |
| `02_ab_routing_log.txt` | Log A/B routing: nhãn `[prompt-v1]`/`[prompt-v2]` cho từng `request_id`, kiểm tra tất định |
| `03_ragas_scores.png` | Bảng so sánh V1 vs V2 trên terminal |
| `03_ragas_report.json` | Bản sao `data/ragas_report.json` |
| `03_ragas_run_log.txt` | Log đầy đủ lần chạy RAGAS — bổ sung |
| `04_pii_demo_log.txt` | Demo PIIDetector (7 case) |
| `04_json_demo_log.txt` | Demo JSONFormatter (6 case) |

> Script Bước 4 in cả 2 demo trong một lần chạy, nên 2 file log của Bước 4 có cùng nội dung (theo hướng dẫn CHECKPOINTS).

## Kết quả RAGAS (50 cặp QA × 2 prompt, mọi chỉ số chấm đủ 50/50 mẫu)

| Metric | V1 (ngắn gọn) | V2 (chuyên gia, có cấu trúc) | Tốt hơn |
|---|---:|---:|---|
| faithfulness | **0.9713** | 0.9162 | V1 |
| answer_relevancy | 0.9029 | **0.9107** | V2 (chênh nhỏ) |
| context_recall | 1.0000 | 1.0000 | bằng nhau |
| context_precision | 0.9417 | 0.9417 | bằng nhau |

Cả 2 phiên bản đều có faithfulness ≥ 0.9.

## Phân tích: vì sao V1 và V2 khác nhau

**1. Context recall/precision bằng nhau, đúng như mong đợi.** Hai chỉ số này chỉ đánh giá phần retrieval (câu hỏi, các chunk truy xuất, đáp án chuẩn) và không phụ thuộc vào câu trả lời. V1 và V2 dùng chung FAISS index, `chunk_size=500` và `k=3`, nên context giống hệt nhau. Prompt chỉ ảnh hưởng đến phần generation, tức là faithfulness và answer_relevancy.

**2. V1 trung thực hơn (0.971 so với 0.916).** Faithfulness = số claim trong câu trả lời suy ra được từ context chia cho tổng số claim.
- V1 yêu cầu trả lời trong 2–4 câu và "giữ nguyên thuật ngữ, con số như trong Context". Câu trả lời ngắn (trung bình **~48 từ**), ít claim, và phần lớn là chép lại câu trong context. Vì vậy gần như claim nào cũng được kiểm chứng.
- V2 yêu cầu thêm 2–4 câu giải thích "định nghĩa, cơ chế hoặc ví dụ". Câu trả lời dài hơn (**~70 từ**, gấp khoảng 1.5 lần), nhiều claim hơn và hay diễn giải lại. Mỗi claim thêm vào là thêm một cơ hội có câu RAGAS không suy ra được từ context, kể cả khi nội dung đúng về mặt kiến thức.
- Trong các trace A/B trên LangSmith, V2 trả lời bằng **tiếng Việt ở 10/89 lần**, V1 thì **0/55**. Nguyên nhân là system prompt viết bằng tiếng Việt, thêm vai "chuyên gia giải thích cho kỹ sư", làm model đôi khi bỏ qua câu "trả lời cùng ngôn ngữ với câu hỏi". Câu trả lời đã dịch phải so khớp claim với context tiếng Anh, nên đây có thể là một nguyên nhân phụ khiến faithfulness của V2 thấp hơn.

**3. V2 nhỉnh hơn về answer_relevancy (0.911 so với 0.903).** Prompt V2 bắt buộc câu đầu tiên là "một câu kết luận trả lời thẳng câu hỏi". Answer relevancy được tính bằng cách sinh lại câu hỏi từ câu trả lời rồi so độ tương đồng embedding với câu hỏi gốc, nên câu mở đầu bám sát câu hỏi giúp tăng điểm. Tuy nhiên mức chênh chỉ 0.008, gần với dao động của LLM-judge, nên không thể coi là khác biệt rõ ràng.

**Kết luận.** Với RAG trên tài liệu nội bộ, nơi độ trung thực quan trọng nhất, nên chọn **V1**. Nếu phát triển V3, nên giữ câu kết luận đầu của V2 (tăng relevancy) nhưng giới hạn độ dài như V1. Đồng thời viết chỉ dẫn ngôn ngữ thật rõ, hoặc viết system prompt bằng tiếng Anh cho khớp với tài liệu, để tránh trả lời sai ngôn ngữ.

## Ghi chú kỹ thuật

- **RAGAS timeout:** lần chạy thử đầu tiên dùng cấu hình mặc định (16 worker, timeout 180s) và chạy song song với Bước 1–2. Kết quả là 9 job bị `TimeoutError`, các mẫu đó thành NaN và bị loại khỏi điểm trung bình. Bản cuối dùng `RunConfig(max_workers=8, timeout=300)`, chạy riêng, và report ghi `valid_scores_per_metric` để chứng minh đủ 50/50.
- **Guardrails:** validator trả về `FailResult(fix_value=...)` để `OnFailAction.FIX` thay được output. `PassResult` sẽ giữ nguyên input.
- **Regex PHONE:** đã sửa `\b` đầu pattern thành `(?<!\w)`. `\b` không khớp được trước dấu `(`, khiến `(555) 867-5309` bị redact thiếu dấu ngoặc.
- **Push lần 2+:** Hub trả về `409 Nothing to commit` khi prompt không đổi. Code coi đây là trạng thái bình thường và in `ℹ️`, không coi là lỗi.
