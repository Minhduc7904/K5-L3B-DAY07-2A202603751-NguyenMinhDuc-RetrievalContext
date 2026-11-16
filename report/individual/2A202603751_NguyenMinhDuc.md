# Báo cáo Day 7: Agentic RAG (retrieval và context engineering)

Nguyễn Minh Đức · MSSV 2A202603751 · Lớp K5-L3B · Repo: `K5-L3B-DAY07-2A202603751-NguyenMinhDuc-RetrievalContext` (remote `origin` đã được cấu hình; xem mục 7 về việc push).

Quy ước: mọi đường dẫn tính từ thư mục gốc của repo; `para:phil:p0287:01` được viết tắt thành `p0287:01` khi đã rõ sách. Mọi số liệu trong bảng được tính lại bằng script chỉ-đọc từ các file `summary.json`, `metrics_*.json`, `reviews.json`, `traces.json` và `results/retrieval/*/summary.json` (không chép từ văn bản mô tả). Cách chạy lại: [HOW_TO_RUN.md](../../HOW_TO_RUN.md); phiên bản gói: [requirements.txt](../../requirements.txt).

## 0. Tóm tắt và trạng thái

- Hệ thống: agent gọi model thật (`gpt-4o-mini`, snapshot `gpt-4o-mini-2024-07-18`) với một tool `search`; bốn phương pháp retrieval (dense trên vector store Day 6, BM25, hybrid RRF, hybrid + cross-encoder rerank) trên cùng 662 chunk Day 6.
- Benchmark: 26 câu có đáp án + 4 câu thiếu evidence = 30 câu; dev 21 + 2, test 5 + 2 (test mới viết cho Day 7).
- Kết quả dev (filters none, 23 câu, một lần chạy mỗi cấu hình): baseline hybrid đạt Context Evidence Rate (CER) 21/21 và tổng điểm câu trả lời 41/42; rerank đạt 21/21 và 42/42. Dev đã bão hòa: các cấu hình dựa trên hybrid không phân biệt được bằng chỉ số agent.
- Cấu hình được chọn là **rerank (A)**, và việc thay baseline bằng rerank được quyết định bởi **điều khoản R** của quy tắc chọn, có ngưỡng được đề xuất **sau khi** đã biết số liệu retrieval của reranker (không phải preregistration mù; xem mục 2.6).
- Test (5 câu có đáp án + 2 thiếu evidence, mỗi cấu hình chạy đúng một lần): cả hai cấu hình trả lời 5/5 với điểm 2/2, từ chối đúng 2/2, 0 từ chối nhầm. Citation được xác nhận hỗ trợ: baseline 4/6, rerank 5/6. Khác biệt này là một citation và không có ý nghĩa thống kê.
- **Nhãn chấm (context, câu trả lời, citation, abstention) do sinh viên Nguyễn Minh Đức trực tiếp đọc trace và chấm thủ công theo BENCHMARK.md.** Trạng thái quy tắc chọn vẫn là PROVISIONAL vì ba run lặp `-rep2` chưa có nhãn (xem mục 2.6 và 5); trường `student_confirmed` trong provenance sẽ được chủ repo cập nhật riêng.
- Giá model trong `src/agent.py` đã được đối chiếu ngày 2026-11-16 với [trang model chính thức của OpenAI](https://developers.openai.com/api/docs/models/gpt-4o-mini): 0.15 USD/1M input token và 0.60 USD/1M output token, khớp bảng dùng để tính kết quả.

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk.** Ba giáo trình OpenStax: *Introduction to Philosophy* (`phil`, 421 trang PDF), *Business Ethics* (`bizethics`, 355), *Business Law I Essentials 2e* (`bizlaw`, 164). Giấy phép **CC BY-NC-SA 4.0**, © Rice University ("Access for free at openstax.org"). Sách tải miễn phí từ openstax.org (ba trang sách ghi trong [HOW_TO_RUN.md](../../HOW_TO_RUN.md)); manifest nguồn, giấy phép, ngày truy cập (2026-11-14) nằm trong repo Day 6 (`data/manifest.csv`, https://github.com/Minhduc7904/K5-L3B-DAY06-2A202603751-DataIngestion). PDF **không** nằm trong repo này (giấy phép và dung lượng). Corpus Day 6 dùng các chương được chọn: 137 trang PDF vật lý (phil 55, bizethics 55, bizlaw 27; đếm từ `data/chunks.json`), 134 trang tính theo quy tắc đếm của Day 6. Số trang trong benchmark là số trang PDF vật lý đánh từ 1, không phải số in ở chân trang. [`data/chunks.json`](../../data/chunks.json) có 662 chunk chứa nguyên văn đoạn sách; giấy phép chỉ cho dùng phi thương mại, phải ghi nguồn và chia sẻ cùng giấy phép, nên repo chỉ nên chia sẻ với người chấm và giữ dòng ghi nguồn này.
- **Vector store, embedding, phiên bản, số chiều.** Vector store của Day 6: container `day6_student_starter-postgres-1`, image `pgvector/pgvector:0.8.7-pg16` (PostgreSQL 16, pgvector 0.8.7), `127.0.0.1:55433`, bảng `chunks`, `strategy = 'B'` (chunk theo tiêu đề/đoạn, tối đa 1000 ký tự; 662 dòng), tìm kiếm cosine **chính xác** (không dùng index xấp xỉ). Embedding `BAAI/bge-base-en-v1.5`, **768 chiều**, vector chuẩn hóa, query dùng tiền tố bge `Represent this sentence for searching relevant passages: ` như Day 6 (`src/retrieval.py`). `DenseRetriever` kiểm tra số chiều của model và số dòng `strategy B` trong pgvector bằng số chunk của `data/chunks.json`, và báo lỗi thay vì dùng phương án dự phòng. Lưu ý phiên bản: README Day 6 ghi đã dựng index với `sentence-transformers 6.3.0`; môi trường Day 7 dùng `6.2.0` (cùng trọng số model; dense tái hiện đúng kết quả Day 6, xem 2.1).
- **Agent model, cách chạy, dev/test.** `gpt-4o-mini` qua OpenAI API, function calling, một tool `search`, temperature 0 ([`src/agent.py`](../../src/agent.py), [`src/run_agent_eval.py`](../../src/run_agent_eval.py)). Retrieval và agent chỉ đọc [`data/questions_public.json`](../../data/questions_public.json) (`id`, `split`, `question`, `filter`); gold nằm trong `data/questions.json` và `data/absent_questions.json`, chỉ code đánh giá đọc sau khi agent chạy xong.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm.** Câu có đáp án: `gold` là danh sách `{source_id, page, evidence}` (evidence là đoạn y nguyên trong chunk; nhiều phần tử `gold` là các lựa chọn thay thế, ví dụ d05 có hai). *Hit* = một chunk đúng `source_id`, đúng trang và chứa `evidence` nguyên văn ([`src/eval_retrieval.py`](../../src/eval_retrieval.py), hàm `is_hit`). CER = gold nằm nguyên văn trong chunk của context cuối cùng gửi cho model, chia số câu có đáp án (chỉ số về context, không phải chất lượng câu trả lời). Câu thiếu evidence: `absence_reason` + `absence_check` (regex) + `grader_note`. Người chấm đọc từng đoạn được cite để gán `context_grade` (0/1/2), `answer_grade` (0/1/2), `citation_support` (đúng/sai cho từng citation) và `abstention_correct`, theo [BENCHMARK.md](../../BENCHMARK.md) mục 2 và 3 (chấm nghiêm: một citation chỉ đúng nếu chính đoạn đó nêu từng mệnh đề được gắn).

| Tập | Có đáp án | Thiếu evidence | Tổng | ID | Loại câu có đáp án |
|---|---:|---:|---:|---|---|
| dev | 21 | 2 | 23 | d01-d21; x01, x02 | direct 8, conditional 7, paraphrase 6 |
| test | 5 | 2 | 7 | t06-t10; x03, x04 | paraphrase 3, direct 2 |
| tổng | 26 | 4 | 30 | | |

Đạt mức tối thiểu của BENCHMARK.md (dev 15 + 2, test 5 + 2) và của SUBMISSION.md (ít nhất 20 câu có evidence, 4 câu thiếu evidence). Chỉ 5 câu dev có filter do người dùng cung cấp (d05, d06, d13, d18, d21; `{document_id}` hoặc `{document_id, chapter}`); không câu thiếu evidence hay câu test nào có filter, nên setting chính là `--filters none` (mô phỏng test) và `--filters question` là setting phụ.

**Quyết định chia tập (quan trọng cho tính độc lập của test).**
- 5 câu test của Day 6 (t01-t05) đã được đánh giá và bàn luận trong báo cáo Day 6 nên không còn độc lập. Chúng được chuyển thành **dev `d17`-`d21`** (giữ `day6_id`, `day6_split` trong `data/questions.json`). Dev = d01-d21 + x01, x02.
- **Test Day 7** là 5 câu mới `t06`-`t10` ([`data/day7_authored.json`](../../data/day7_authored.json); ghi nhận viết ngày 2026-11-16, file lưu lúc 10:10, trước mọi lần chạy agent: file kết quả agent sớm nhất lưu lúc 10:48) cộng 2 câu thiếu evidence `x03`, `x04`. Test chỉ được chạy sau khi đóng băng (`results/FROZEN_CONFIG.md` lưu 22:06:52; `summary.json` của hai run test lưu 22:08:57 và 22:10:00; giờ lấy từ thời gian sửa file).
- Một lần chạy TF-IDF baseline đã được thực hiện trên 5 câu test mới chỉ để kiểm tra định dạng (3/5 Hit@1, 5/5 Hit@3; ghi trong `data/README.md`); output đã bị xóa và không có thay đổi nào được thực hiện dựa trên kết quả đó. Như vậy test đã bị "chạm" một lần bằng baseline từ vựng nằm ngoài tập phương pháp được báo cáo; nêu ra để minh bạch. Thư mục `starter/results/tfidf-dev-smoke/` còn trên máy là một lần chạy TF-IDF trên **dev** (21 câu: Hit@1 15, Hit@3 18), không phải lần chạy test nói trên; nó không được dùng trong bảng nào và bị `.gitignore` bỏ qua.
- **Kiểm tra câu thiếu evidence.** [`src/verify_absent.py`](../../src/verify_absent.py) chuẩn hóa văn bản (bỏ soft hyphen, gộp khoảng trắng để ngắt dòng không che cụm từ) rồi quét mọi regex `absence_check` trên **662 chunk và toàn bộ 940 trang của ba PDF** (1602 văn bản quét), không chỉ top-k của một retriever. Kết quả (chạy lại khi viết báo cáo): `problems=0`; không regex nào khớp, riêng `decoy_check` của x03 (Jimmy John's, "two years", PDF tr.195, chunk `para:bizethics:p0195:01`) khớp đúng chunk đó và không khớp chunk khác. x01 (thời hiệu khởi kiện): sách có negligence nhưng không có "statute of limitations". x02 (phản biện đạo đức về nhân bản người): "clone" chỉ xuất hiện trong tên công ty ImClone. x03 (thời hạn tối đa của non-compete): sách chỉ nói "usually limited by time and distance". x04 (tuổi nghỉ hưu bắt buộc): ADEA chỉ bảo vệ người từ 40 tuổi trở lên.
- **Ghi nhận về rà soát độc lập benchmark.** Theo quy trình làm bài, lần rà soát độc lập của benchmark phát hiện `absence_reason` đầu tiên của x03 là sai và nó đã được sửa; bản cũ không còn trong repo nên điều này không kiểm chứng được từ file (chỉ bản đã sửa còn lại, và `verify_absent.py` xác nhận bản đó).

## 2. Retrieval và thử nghiệm

### 2.1 Bốn phương pháp (cùng corpus, cùng index, cùng bộ câu hỏi và nhãn dev)

| Phương pháp | Cách hoạt động | Ví dụ từ benchmark (hạng của chunk gold, filters none) |
|---|---|---|
| dense | embedding query bằng bge-base-en-v1.5 rồi cosine chính xác trong pgvector Day 6 (`DenseRetriever`) | d01 "In Kant's second main test of a moral rule, how must we treat other people, and ourselves?": dense hạng 2 (gold `p0291:03` nói "in your own person as well as in the person of any other"), BM25 hạng 5 vì ít từ trùng. Ngược lại d17 (câu hỏi nói "partly responsible", gold nói "comparative negligence"): dense hạng 10 |
| BM25 | Okapi BM25 (k1 1.5, b 0.75), Python thuần, bỏ stop-word, trên cùng văn bản chunk (`BM25Retriever`) | d17: BM25 hạng 1; d01: hạng 5 |
| hybrid | Reciprocal Rank Fusion của dense và BM25: điểm = tổng 1/(60 + rank), 50 ứng viên mỗi bên, hòa thì theo id chunk (`HybridRetriever`) | d17: hạng 2; d01: hạng 2; d18 ("How much advance notice must employers give before closing a plant ..."): dense 8, BM25 2, hybrid 4 |
| rerank | lấy pool 30 ứng viên từ hybrid, chấm từng cặp (query, đoạn) bằng cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` chạy local, sắp xếp lại và trả top-k (`RerankRetriever`) | d18: hybrid 4, rerank 1; d01: hybrid 2, rerank 1; d17: rerank 2 |

Nguồn hạng: `results/retrieval/dev-filters-none/traces.json` và `results/retrieval/dev-filters-none-rerank-minilm/traces.json`. Mỗi kết quả có `id`, `page`, `source_id`, `score`, `rank` (hybrid và rerank có thêm `components` để kiểm vết trước/sau); filter của người dùng được áp dụng **trước** khi xếp hạng ở cả bốn phương pháp. Khi model embedding hoặc kết nối pgvector lỗi, code ném `RuntimeError`, không có kết quả thay thế giả.

**Dense tái hiện Day 6.** Trên 16 câu dev gốc (d01-d16), dense đạt Hit@1 13/16 và Hit@3 15/16 (cả hai setting filter; tính từ `results/retrieval/dev-filters-*/traces.json`), đúng bằng kết quả cấu hình B của Day 6 (13/16 và 15/16 trong README repo Day 6). Trên 21 câu dev của Day 7 (có thêm d17-d21) dense đạt 15/21 và 18/21.

### 2.2 Bảng retrieval-only (dev, 21 câu có đáp án, query gốc, độ sâu 10)

Hit@k tính trên query gốc của người dùng, không qua agent. Độ trễ (ms trung vị mỗi query, CPU) lấy từ cùng một lần chạy `-rerank-minilm` (hoặc `-rerank-bge` cho dòng bge) để bốn phương pháp so được với nhau; số Hit của dense/BM25/hybrid trùng với `dev-filters-none` và `dev-filters-question` (độ trễ giữa các lần chạy khác nhau do máy). Chi phí dịch vụ ngoài: 0 USD (mọi phương pháp chạy local). "Trượt@3" là các câu mà chunk gold không nằm trong top 3.

| Filter | Phương pháp | Hit@1 | Evidence Hit@3 | Hit@5 | MRR@10 | ms/query | Trượt@3 |
|---|---|---|---|---|---|---|---|
| none | dense | 15/21 | 18/21 | 19/21 | 0.798 | 80.1 | d12, d17, d18 |
| none | bm25 | 18/21 | 20/21 | 21/21 | 0.914 | 0.7 | d01 |
| none | hybrid (RRF) | 16/21 | 20/21 | 21/21 | 0.861 | 80.6 | d18 |
| none | rerank MiniLM (pool 30) | 19/21 | 21/21 | 21/21 | 0.944 | 1059.3 | - |
| none | rerank bge-reranker-base (pool 30) | 19/21 | 20/21 | 20/21 | 0.929 | 7044.9 | d12 |
| question | dense | 15/21 | 18/21 | 20/21 | 0.804 | 67.0 | d12, d17, d18 |
| question | bm25 | 20/21 | 20/21 | 21/21 | 0.962 | 0.4 | d01 |
| question | hybrid (RRF) | 17/21 | 21/21 | 21/21 | 0.897 | 77.6 | - |
| question | rerank MiniLM (pool 30) | 19/21 | 21/21 | 21/21 | 0.944 | 1055.6 | - |
| question | rerank bge-reranker-base (pool 30) | 19/21 | 20/21 | 20/21 | 0.929 | 7196.9 | d12 |

Đọc kết quả: reranker MiniLM tăng Hit@1 từ 16/21 (hybrid) lên 19/21 và Hit@3 từ 20/21 lên 21/21 khi không có filter; reranker lớn hơn `BAAI/bge-reranker-base` không tốt hơn (Hit@3 20/21, mất d12) và chậm khoảng 7 giây mỗi query, nên không được dùng trong agent (quy tắc chọn reranker của `SELECTION.md` mục 2 và amendment 2: tổng Hit@1 + Hit@3 là 19 + 21 = 40 so với 19 + 20 = 39). Dense yếu nhất ở Hit@1 và MRR; BM25 mạnh trên bộ câu tiếng Anh này và sửa được các câu dense trượt (d17, d18, d12) nhưng trượt d01.

### 2.3 Bảng cấu hình theo mẫu báo cáo

Cột "Hit@1" và "Evidence Hit@3" của dòng dev là retrieval-only (query gốc, filters none, 21 câu); cột "Context Evidence Rate" là CER của lần chạy agent tương ứng (filters none); cột cuối là thời gian trung vị mỗi câu và chi phí ước tính của cả lần chạy 23 câu hoặc 7 câu (đơn giá đã đối chiếu với OpenAI ngày 2026-11-16). Các dòng B, C, B+C và A+B+C dùng cùng retriever với hybrid hoặc rerank nên Hit@k bằng của phương pháp nền. **(\*) Test chưa chạy retrieval-only** (test chỉ được chạy một lần qua agent), nên hai cột Hit của dòng test là hạng của chunk gold ở **lượt search đầu tiên của agent** (`first_search_evidence_rank` trong `traces.json`, query do model viết, 5 câu có đáp án); không so trực tiếp với số dev.

| Cấu hình | Phương pháp, top-k, filter, rerank, budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Phương pháp 1: dense (Day 6) | dense, top-k 5, filter none, không rerank, budget 10000 ký tự, prompt base | tham chiếu: vector store Day 6, bge-base-en-v1.5 | dev | 15/21 | 18/21 | 19/21 | 4.14 s / $0.00936 |
| Phương pháp 2: BM25 | BM25 (k1 1.5, b 0.75), cùng tham số | từ khóa trùng khớp bù chỗ dense yếu | dev | 18/21 | 20/21 | 21/21 | 3.54 s / $0.00957 |
| Phương pháp 3: hybrid RRF = baseline B0 | RRF(dense, BM25), k 60, 50 ứng viên/bên, cùng tham số | kết hợp hai bảng xếp hạng | dev | 16/21 | 20/21 | 21/21 | 3.34 s / $0.00844 |
| Cải thiện 1 (A): rerank MiniLM, cũng là phương pháp thứ 4 | hybrid pool 30, cross-encoder ms-marco-MiniLM-L-6-v2, top 5 | sửa lỗi xếp hạng E1 (d17, d18) | dev | 19/21 | 21/21 | 21/21 | 4.46 s / $0.0085 |
| Cải thiện 2 (B): prompt grounded, đúng đặc tả | hybrid, prompt `grounded`, không id-repair | giảm câu thừa/không căn cứ (E2-E4); kết quả âm: 4 từ chối nhầm | dev | 16/21 | 20/21 | 21/21 | 3.08 s / $0.0093 |
| Phụ C: id-repair trên prompt base | hybrid, prompt base, `--repair-ids` | chỉ khôi phục tiền tố `para:` | dev | 16/21 | 20/21 | 21/21 | 3.69 s / $0.00841 |
| Phụ B+C | hybrid, prompt grounded, `--repair-ids` | phép thử công bằng cho B | dev | 16/21 | 20/21 | 21/21 | 3.49 s / $0.00938 |
| Phụ A+B+C | rerank MiniLM, prompt grounded, `--repair-ids` | gộp tất cả | dev | 19/21 | 21/21 | 21/21 | 4.37 s / $0.00909 |
| Cấu hình đã chọn (A: rerank MiniLM) | như A, prompt base, không id-repair, agent v1c-frozen | đã chốt trên dev (clause R) | test | 5/5 (*) | 5/5 (*) | 5/5 | 5.32 s / $0.00279 |
| Baseline B0 (hybrid) | hybrid, prompt base, agent v1c-frozen | đối chứng | test | 5/5 (*) | 5/5 (*) | 5/5 | 4.27 s / $0.00284 |

Tham số chung của mọi dòng: top-k 5 mỗi lượt search, budget 10000 ký tự cho cả lượt chạy, tối đa 3 lượt search rồi một lượt trả lời bắt buộc, temperature 0, `gpt-4o-mini`. Rerank vừa là phương pháp thứ tư vừa là cấu hình cải thiện 1 (một thay đổi duy nhất: bước xếp hạng lại). Cải thiện 2 là prompt `grounded` (một thay đổi duy nhất: prompt). C (`--repair-ids`) là biến thể **phụ, được thêm sau khi đã thấy kết quả B**, không thuộc hai cấu hình cải thiện bắt buộc (xem amendment 3 trong `results/SELECTION.md`).

### 2.4 Lỗi dev, nguyên nhân, thử nghiệm, trước/sau

Phân tích đầy đủ, viết **trước** khi chạy cấu hình mới: [results/ERROR_ANALYSIS.md](../../results/ERROR_ANALYSIS.md). Nhãn nêu dưới đây lấy từ `reviews.json` và `reviews_reasons.json` của các run tương ứng.

1. **E1, d17 và d18 (dense, filters none): lỗi xếp hạng.** d17: chunk gold `para:bizlaw:p0074:02` ("comparative negligence") ở hạng 10 của dense; câu hỏi nói "partly responsible" nên dense xếp các đoạn "contributory negligence" lên trước. Trong `dev-dense-base-nofilter` agent tìm ba lần, trôi sang "contributory negligence", trả lời sai học thuyết và cite `p0074:00`, `p0075:00`: context_grade 1, answer_grade 0, một citation sai (`p0074:00` nói contributory negligence **chặn** quyền đòi bồi thường, trái với câu trả lời). d18: gold `p0102:00` ở hạng 8 của dense; agent chỉ thấy `bizethics:p0157:01` (60 ngày nhưng không có ngưỡng quy mô), trả lời kèm dòng `INSUFFICIENT_EVIDENCE` ở cuối: answer_grade 1. *Thử nghiệm:* hybrid (đã là baseline) và cấu hình A rerank. *Trước/sau (retrieval-only):* d17 hạng 10 (dense), 2 (hybrid), 2 (rerank); d18 hạng 8, 4, 1. *Trước/sau (agent, filters none):* d17 answer_grade 0 (dense), 2 (hybrid), 2 (rerank); d18 1, 2, 2. Giới hạn: phần lớn cải thiện đến từ việc chuyển dense sang hybrid, vốn đã là baseline; rerank chỉ cải thiện thêm Hit@1 và Hit@3, không thấy được ở chỉ số agent vì CER của hybrid đã là 21/21.
2. **E2, x02 (bm25): từ chối thất bại ở câu thiếu evidence.** Câu hỏi về phản biện đạo đức đối với nhân bản người; sách không bàn về nhân bản. Trong `dev-bm25-base`, bm25 trả lời bằng nội dung lấy từ các đoạn lân cận chủ đề (phá thai/an tử `p0325:01`, thử nghiệm lâm sàng `p0328:01`, chỉnh sửa gen `p0332:00`): `abstention_correct = false`, cả 3 citation sai. Nguyên nhân: các đoạn gần về từ vựng ("ethical", "human", "objections") bị model coi là bằng chứng. Lỗi **không ổn định**: bm25 trả lời x02 ở 3 trong 4 lần chạy base (base, base-rep2, base-nofilter) và từ chối ở lần thứ tư (base-nofilter-rep2). *Thử nghiệm:* prompt `grounded` (B). *Giới hạn quan trọng:* B chỉ được chạy với hybrid, vốn đã từ chối x02 trong mọi lần chạy, nên tác dụng của B lên lỗi của bm25 **chưa bao giờ được kiểm tra**. Việc tránh lỗi này trong cấu hình chọn thực chất là nhờ chọn hybrid/rerank thay vì BM25 thuần (gate G3 của quy tắc loại bm25).
3. **E3/E4, câu thừa và citation đặt sai chunk (d08, d20, d01).** d08 (hybrid, filters none): lõi câu trả lời đúng (non-compete không thi hành được ở California) nhưng thêm "reflects the approach that most states are now taking" trong khi đoạn được cite chỉ nêu đó là một câu hỏi: answer_grade 1, citation sai. d20: câu utilitarian thừa cite `p0331:04`, đoạn không nói "prohibitively costly". *Thử nghiệm:* B+C và A+B+C. *Trước/sau:* d08 answer_grade 1 (hybrid) sang 2 (B+C, A+B+C), nhưng câu "2017 California law" được cite vào `p0195:04` (nonsolicitation) thay vì `p0194:03` nơi nó thực sự nằm: citation vẫn sai. Tức B không loại được câu thừa mà chỉ biến "không có căn cứ" thành "cite nhầm chunk". Cấu hình rerank cũng đạt answer_grade 2 cho d08 với một citation sai tương tự, nên khác biệt giữa các cấu hình ở câu này một phần là dao động lấy mẫu của model chứ không chỉ do cấu hình.
4. **E5, tiền tố `para:` bị bỏ (d05, d12, d13, d16).** `dev-hybrid-grounded-nofilter` (B đúng đặc tả) **từ chối nhầm 4 câu có đáp án** (d05, d12, d13, d16), cả bốn với lý do `invalid_citation` vì model viết đúng id nhưng bỏ `para:` (ví dụ `bizlaw:p0104:03`): tổng điểm 31/42 so với 41/42. Đây là **kết quả âm** của B, giữ nguyên trong báo cáo; B không qua gate G2 của quy tắc chọn. Biến thể C (`--repair-ids`, chỉ khôi phục tiền tố `para:` khi `para:X` là id đã hiển thị) biến các lỗi này lại thành câu trả lời, và mọi lần repair được ghi trong trace; B+C đạt 42/42 và 0 từ chối nhầm. C **che** chứ không loại bỏ chi phí định dạng của prompt `grounded`: B+C cần 3 lần repair (filters none), prompt base trên hybrid cần 0.

### 2.5 Hai cấu hình cải thiện và các bài học ở mức harness

| | Thay đổi | Kết quả dev (filters none) | Kết luận |
|---|---|---|---|
| A rerank | pool 30 rồi cross-encoder MiniLM | Hit@1 16/21 sang 19/21, Hit@3 20/21 sang 21/21; CER 21/21 (không đổi); điểm câu trả lời 42/42 so với 41/42; từ chối đúng 2/2 | cải thiện retrieval, không đo được cải thiện ở agent; trung vị 4.46 s so với 3.34 s (+34%) |
| B grounded | prompt nghiêm hơn về bằng chứng và vị trí citation | CER 21/21; 4 từ chối nhầm (d05, d12, d13, d16), điểm 31/42; trung vị 3.08 s | **kết quả âm**. B+C: 42/42 nhưng citation hỗ trợ 21/27 (0.778) so với 20/23 (0.870) của baseline (nhiều citation hơn, nhiều citation sai hơn) |

Hai sửa lỗi harness trước khi đóng băng (các run `-v0`, `-v1a` được giữ làm bằng chứng, `compare_runs.py` bỏ qua chúng):

- **Over-search (v0 sang v1b).** Ở v0, prompt có dòng "Searches left: N" và lời nhắc "rephrase and search again"; model chỉ chọn 1 hoặc 3 lượt (hybrid và dense: 19/23 câu dùng 3 lượt; bm25: 16/23). Sau khi sửa (bỏ đếm ngược, quan sát tool trung tính, prompt quy định chỉ tìm lần nữa khi đoạn chưa nêu câu trả lời), số lượt trung bình giảm từ 2.39-2.65 xuống 1.17-1.30. Bảng (v0 / v1a / v1b, filters = question):

| Phương pháp | Số lượt TB | Trung vị (s) | Chi phí (USD) | Từ chối nhầm câu có đáp án |
|---|---|---|---|---|
| hybrid | 2.65 / 1.17 / 1.17 | 6.03 / 4.1 / 3.65 | 0.01817 / 0.00814 / 0.00835 | - / d05 / - |
| dense | 2.65 / 1.3 / 1.3 | 5.68 / 3.36 / 4.02 | 0.01814 / 0.00906 / 0.00934 | d03 / d05,d13,d17 / - |
| bm25 | 2.39 / 1.22 / 1.22 | 5.41 / 3.77 / 3.38 | 0.01688 / 0.00872 / 0.00894 | - / d20 / - |

- **Sổ citation.** v0: ví dụ id trong prompt (`[para:book:p0001:00]`) khiến model tự bịa tiền tố (dense d03 cite `para:book:p0328:01`: `invalid_citation`, lỗi do prompt chứ không do retrieval), id không hợp lệ bị bỏ khỏi danh sách citation, mẫu số CER khác. v1a: model bỏ tiền tố `para:` và bộ trích citation bỏ qua (`no_citation`); v1b: mẫu id rộng hơn và định nghĩa id tường minh trong quy tắc 3 của prompt; từ chối nhầm giảm (xem cột cuối bảng trên). Các run v1a và v1b khác nhau cả vì dao động của model nên phần giảm không thể quy hoàn toàn cho bản sửa.

### 2.6 Quy tắc chọn, kết quả và mức độ tin cậy

Quy tắc ([results/SELECTION.md](../../results/SELECTION.md)) được viết sau `ERROR_ANALYSIS.md` và trước các cấu hình mới: setting chính là dev filters none; cổng G1 (CER không thấp hơn incumbent quá 1), G2 (không lỗi nhà cung cấp, từ chối nhầm không quá incumbent + 1), G3 (số câu thiếu evidence được từ chối đúng không ít hơn incumbent); rồi so L1 (tổng điểm câu trả lời), L2 (CER), L3 (tỷ lệ citation hỗ trợ) với ngưỡng T = max(sàn, nhiễu giữa hai lần chạy giống hệt); hòa thì xét độ trễ rồi chi phí. **Có bốn amendment, đều ghi giờ trong file:** (1) làm rõ nhiễu theo từng phương pháp; (2) áp dụng quy tắc chọn reranker (MiniLM); (3) thêm biến thể C **sau khi đã thấy kết quả B** và các run nhiễu; (4) viết **sau khi đã có bảng so sánh tạm thời không nhãn**: ngưỡng độ trễ 10% đổi thành 25%, L3 thành tỷ lệ, "mức đầu tiên phân biệt được quyết định theo cả hai chiều", và thêm **điều khoản R** (challenger thay incumbent nếu Hit@1 cao hơn ít nhất 3 câu, Hit@3 không thấp hơn và độ trễ trung vị không quá 1.4 lần). `src/compare_runs.py` được viết lại sau khi có kết quả và được kiểm bởi `tests/test_compare_runs.py`.

Kết quả (nhãn do sinh viên chấm thủ công; số tính lại từ `reviews.json`, `summary.json` và retrieval-only):

| Cấu hình | Run | Tổng điểm câu trả lời | Citation hỗ trợ / được chấp nhận | CER | Từ chối đúng (x01, x02) | Hit@1 | Hit@3 | MRR@10 | Trung vị (s) | Chi phí (USD) |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid base (B0, baseline) | `dev-hybrid-base-nofilter` | 41/42 | 20/23 (0.870) | 21/21 | 2/2 | 16 | 20 | 0.861 | 3.34 | 0.00844 |
| bm25 base | `dev-bm25-base-nofilter` | 39/42 | 17/23 (0.739) | 21/21 | 1/2 | 18 | 20 | 0.914 | 3.54 | 0.00957 |
| dense base | `dev-dense-base-nofilter` | 37/42 | 19/24 (0.792) | 19/21 | 2/2 | 15 | 18 | 0.798 | 4.14 | 0.00936 |
| A: rerank MiniLM | `dev-rerank-minilm-nofilter` | 42/42 | 20/23 (0.870) | 21/21 | 2/2 | 19 | 21 | 0.944 | 4.46 | 0.0085 |
| B: grounded (như đặc tả) | `dev-hybrid-grounded-nofilter` | 31/42 | 17/21 (0.810) | 21/21 | 2/2 | 16 | 20 | 0.861 | 3.08 | 0.0093 |
| C: id-repair (phụ) | `dev-hybrid-repair-nofilter` | 41/42 | 21/23 (0.913) | 21/21 | 2/2 | 16 | 20 | 0.861 | 3.69 | 0.00841 |
| B+C | `dev-hybrid-grounded-repair-nofilter` | 42/42 | 21/27 (0.778) | 21/21 | 2/2 | 16 | 20 | 0.861 | 3.49 | 0.00938 |
| A+B+C | `dev-rerank-minilm-grounded-repair-nofilter` | 42/42 | 23/27 (0.852) | 21/21 | 2/2 | 19 | 21 | 0.944 | 4.37 | 0.00909 |

Kết quả áp dụng quy tắc ([DEV_COMPARISON.md](../../results/DEV_COMPARISON.md), [dev_comparison.json](../../results/dev_comparison.json)): bm25 không qua gate G3 (từ chối đúng 1/2 so với 2/2); dense không qua G1 (CER 19 so với 21); B không qua G2 (4 từ chối nhầm); **A rerank thay baseline, quyết định bởi điều khoản R** (Hit@1 19 so với 16, Hit@3 21 so với 20, độ trễ ×1.34 không quá ×1.4; L1 42 so với 41 chênh 1 < T = 2, L2 21 so với 21, L3 0.870 so với 0.870); C, B+C và A+B+C qua gate nhưng không phân thắng thua ở L1-L3 hay R và kết thúc ở "giữ incumbent mặc định (inconclusive)", nghĩa là **không** chứng tỏ incumbent tốt hơn.

Điều cần nói rõ:
- Ngưỡng của R (Hit@1 +3, độ trễ ×1.4) được đề xuất **sau** khi biết số liệu retrieval của reranker (16 sang 19, 20 sang 21) và độ trễ agent (+33%); đây **không** phải preregistration mù và có thể được khớp cho rerank. Nếu bác bỏ R, đầu ra của quy tắc là "không challenger nào thắng baseline" và baseline hybrid vẫn là incumbent. Việc chọn rerank vì vậy là bằng chứng về lợi ích retrieval không kèm tổn thất đo được ở agent, không phải bằng chứng về câu trả lời tốt hơn.
- Quy tắc **khó thắng trên dev theo cấu trúc**: CER của baseline đã là 21/21 (L2 không phân biệt), điểm câu trả lời gần trần (41-42/42, ngưỡng T = 2) và hòa rơi về độ trễ, nơi reranker hoặc prompt dài luôn bị tính là chi phí. Vì thế "không ai thắng baseline" là kết quả được kỳ vọng.
- Trạng thái **PROVISIONAL**: ba lần chạy lặp `-rep2` của baseline (hybrid, bm25, dense) **chưa được chấm** nên nhiễu của L1 và L3 được tính là 0 và T nằm ở sàn (nhiễu chỉ làm T tăng nên không đổi được các kết luận "không mức nào phân biệt", nhưng nhãn PROVISIONAL được giữ).
- Baseline và cấu hình chọn đều đã chạy **một lần** trên test (xem mục 3).

Nhiễu giữa hai lần chạy giống hệt (filters none; **chưa chấm**, chỉ chỉ số tự động):

| Phương pháp | CER (run 1 / rep2) | Từ chối nhầm (run 1 / rep2) | Trả lời câu thiếu evidence (run 1 / rep2) | Trung vị (s) |
|---|---|---|---|---|
| hybrid | 21/21 / 21/21 | - / - | - / - | 3.34 / 3.54 |
| bm25 | 21/21 / 21/21 | d13 / d13 | x02 / - | 3.54 / 3.33 |
| dense | 19/21 / 21/21 | - / d17 | - / - | 4.14 / 3.64 |

## 3. Agent và đánh giá

### 3.1 Thiết kế agent và harness đã đóng băng

- **Tool:** một tool `search(query)`; model chỉ viết nội dung query. Phương pháp retrieval và filter của người dùng do harness áp dụng (`--filters question` đưa filter trong `questions_public.json`; `--filters none` bỏ qua); model không thấy gold, đáp án hay ghi chú.
- **Chọn context:** top-k 5 mỗi lượt; khử trùng theo id chunk giữa các lượt; context cuối là hợp của mọi đoạn đã hiển thị theo thứ tự hạng; budget 10000 ký tự trên cả lượt chạy (khoảng 13 chunk; chunk trung bình 679, tối đa 1085 ký tự); chunk không bao giờ bị cắt, chunk không vừa budget bị bỏ và ghi `dropped_budget`. Context trung bình của hai run test: 4367.9 và 4261.1 ký tự.
- **Điều kiện dừng:** model trả lời không gọi tool, hoặc đủ `max_searches` = 3 rồi một lượt trả lời với `tool_choice="none"`.
- **Citation:** một câu trả lời chỉ được nhận nếu có ít nhất một citation và mọi `[chunk-id]` đều nằm trong context cuối và trong văn bản câu trả lời; nếu không, trạng thái là `abstain` với lý do ghi lại (`invalid_citation`, `no_citation`). Id hợp lệ chưa chứng minh nguồn hỗ trợ câu trả lời; việc đó do người chấm đọc đối chiếu.
- **Thiếu evidence:** model trả một dòng bắt đầu bằng `INSUFFICIENT_EVIDENCE:`. Từ `v1c-frozen`, marker xuất hiện **ở vị trí khác đầu câu trả lời** là `abstain` với lý do `mixed_marker`. Các thư mục dev cũ **không được tạo lại** và giữ trạng thái cũ (ảnh hưởng một ca: d18 của `dev-dense-base-nofilter` vẫn là `answered`).
- **Lỗi và retry:** lỗi tạm thời (kết nối, timeout, 429, 5xx) được thử lại tối đa 4 lần (`api_attempts` 4, nghỉ 2 giây); lỗi còn lại của nhà cung cấp hoặc retriever ghi `status: error` (loại khỏi chỉ số chất lượng, báo riêng) và run tiếp tục; không có kết quả thay thế giả. Lỗi định dạng phía model (tool lạ, tham số sai, query rỗng, câu trả lời bị cắt) là `abstain` và **tính vào bất lợi của agent**. Trong 35 thư mục `results/agent/*` không có lần chạy nào có `status.error > 0`.
- **Tham số cố định khi chạy test:** temperature 0, `max_completion_tokens` 600, `parallel_tool_calls` false, không gửi `seed` (không kiểm chứng được tác dụng nếu không gọi API trả phí; vì vậy mỗi lần chạy chỉ là một mẫu), `--max-cost` 0.50 USD mỗi lần chạy test.
- **Đóng băng ([results/FROZEN_CONFIG.md](../../results/FROZEN_CONFIG.md)):** `AGENT_VERSION` = `v1c-frozen`. SHA-256 (12 ký tự đầu; đã chạy lại `python src/freeze_info.py` khi viết báo cáo và đối chiếu với `FROZEN_CONFIG.md` mục 2): `src/agent.py` `d27b6c595ca8`; `src/run_agent_eval.py` `df12a7fe9bc0`; `src/retrieval.py` `ef429958650d`; `src/compare_runs.py` `b5bfdc322f46`; `data/chunks.json` `8e3d3d2d2b53`; `data/questions_public.json` `82318c71f620`; `data/questions.json` `2719cdcd2a58`; `data/absent_questions.json` `502a7f44fc83`; prompt base `3c00e31013fc`; prompt grounded `641bd4b6f274`. `summary.json` của hai run test lưu `agent_code_sha256`, `system_prompt_sha256`, `chunks_sha256`, `questions_public_sha256` và các giá trị này khớp bảng đóng băng. Các hash đóng băng nhận dạng chính xác mã/dữ liệu đã dùng cho hai run test; lịch sử Git của repo nhận dạng toàn bộ gói bài nộp.

### 3.2 Chỉ số tự động và chỉ số chấm bằng AI

Cột theo BENCHMARK.md mục 3. "Số lượt gọi" là phân bố số lượt search (số lượt:số câu) và trung bình (TB). "Grounded answer" là số câu ở mức 0/1/2 và trung bình trên 2, tính trên câu có đáp án. "Citation hợp lệ" là số citation được người chấm xác nhận hỗ trợ / tổng citation được chấp nhận (id có trong context). "Câu có citation" là số câu trả lời có citation / số câu trả lời. "Abstention đúng" là số câu thiếu evidence được từ chối đúng / tổng câu thiếu evidence (dev 2, test 2); "Từ chối nhầm" là số câu có đáp án bị từ chối / tổng câu có đáp án. "Context grade 2" là số câu có đáp án mà người chấm cho context đủ evidence và nguồn (mức 2).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và TB/2 | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context grade 2 |
|---|---|---|---|---|---|---|---|---|
| dev, filters, baseline (`dev-bm25-base`) | 23/23 | 1:20 / 2:1 / 3:2 (TB 1.22) | 0/1/20, 1.95 | 20/24 | 22/22 | 1/2 | 0/21 | 21/21 |
| dev, filters, baseline (`dev-dense-base`) | 23/23 | 1:19 / 2:1 / 3:3 (TB 1.3) | 0/1/20, 1.95 | 22/23 | 21/21 | 2/2 | 0/21 | 21/21 |
| dev, filters, baseline (`dev-hybrid-base`) | 23/23 | 1:21 / 3:2 (TB 1.17) | 0/2/19, 1.90 | 20/23 | 21/21 | 2/2 | 0/21 | 21/21 |
| dev nofilter (`dev-bm25-base-nofilter`) | 23/23 | 1:19 / 2:1 / 3:3 (TB 1.3) | 1/1/19, 1.86 | 17/23 | 21/21 | 1/2 | 1/21 | 21/21 |
| dev nofilter (`dev-dense-base-nofilter`) | 23/23 | 1:18 / 2:3 / 3:2 (TB 1.3) | 1/3/17, 1.76 | 19/24 | 21/21 | 2/2 | 0/21 | 18/21 |
| dev nofilter (`dev-hybrid-base-nofilter`) | 23/23 | 1:21 / 3:2 (TB 1.17) | 0/1/20, 1.95 | 20/23 | 21/21 | 2/2 | 0/21 | 21/21 |
| dev nofilter, A (`dev-rerank-minilm-nofilter`) | 23/23 | 1:21 / 3:2 (TB 1.17) | 0/0/21, 2.00 | 20/23 | 21/21 | 2/2 | 0/21 | 21/21 |
| dev nofilter, B (`dev-hybrid-grounded-nofilter`) | 23/23 | 1:21 / 3:2 (TB 1.17) | 5/1/15, 1.48 | 17/21 | 17/17 | 2/2 | 4/21 | 21/21 |
| dev nofilter, C (`dev-hybrid-repair-nofilter`) | 23/23 | 1:21 / 3:2 (TB 1.17) | 0/1/20, 1.95 | 21/23 | 21/21 | 2/2 | 0/21 | 21/21 |
| dev nofilter, B+C (`dev-hybrid-grounded-repair-nofilter`) | 23/23 | 1:21 / 3:2 (TB 1.17) | 0/0/21, 2.00 | 21/27 | 21/21 | 2/2 | 0/21 | 21/21 |
| dev nofilter, A+B+C (`dev-rerank-minilm-grounded-repair-nofilter`) | 23/23 | 1:21 / 2:1 / 3:1 (TB 1.13) | 0/0/21, 2.00 | 23/27 | 21/21 | 2/2 | 0/21 | 21/21 |
| test, baseline (`test-hybrid-base`) | 7/7 | 1:6 / 3:1 (TB 1.29) | 0/0/5, 2.00 | 4/6 | 5/5 | 2/2 | 0/5 | 5/5 |
| test, chọn (A) (`test-rerank-minilm`) | 7/7 | 1:6 / 3:1 (TB 1.29) | 0/0/5, 2.00 | 5/6 | 5/5 | 2/2 | 0/5 | 5/5 |

- Mọi nhãn trong bảng do **sinh viên Nguyễn Minh Đức** chấm thủ công theo rubric nghiêm. Các dòng "dev nofilter" gồm 8 run dev có nhãn; ba run `dev-*-base-nofilter-rep2` **chưa được chấm** nên trạng thái lựa chọn vẫn là PROVISIONAL.
- `dev-bm25-base-nofilter` có một từ chối nhầm (d13, `invalid_citation`) và một câu thiếu evidence bị trả lời (x02). `dev-dense-base-nofilter` có CER 19/21 (d17, d18 không có gold trong context).
- Phân bố lượt gọi của baseline hybrid dev: 21 câu có đáp án dùng 1 lượt, hai câu thiếu evidence (x01, x02) dùng 3 lượt; đó là hành vi mong muốn khi không tìm thấy bằng chứng.

### 3.3 Kết quả test (n = 5 + 2): từng câu

Test chỉ chạy **một lần** mỗi cấu hình, bằng hai lệnh trong [results/FROZEN_CONFIG.md](../../results/FROZEN_CONFIG.md) mục 6 (cũng ghi trong HOW_TO_RUN.md). Trace và nhãn: `results/agent/test-hybrid-base/` và `results/agent/test-rerank-minilm/` (`traces.json`, `reviews.json`, `reviews_reasons.json`, `reviews_provenance.json`, `metrics_test.json`). Cột "Nhãn" gồm context grade (cg), answer grade (ag) và citation (T = hỗ trợ, F = không).

**Baseline: hybrid, prompt base (`test-hybrid-base`)**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t06 | có đáp án (paraphrase) | 1 search "John Stuart Mill revise Bentham view of pleasure"; gold `p0287:01` hạng 1 | Mill thừa nhận khoái lạc khác nhau về lượng **và chất**, phân biệt khoái lạc cao/thấp; `[p0287:01]` | cg 2, ag 2, [T]: đoạn nêu đủ mọi mệnh đề |
| t07 | có đáp án (direct) | 1 search "act utilitarianism basis for deciding right action"; gold `p0288:02` hạng 1 | nguyên tắc hạnh phúc lớn nhất áp dụng từng trường hợp; thêm "giá trị nội tại duy nhất là hạnh phúc, chỉ hậu quả mới đáng kể"; `[p0288:02]`, `[p0289:00]` | cg 2, ag 2, [T, **F**]: mệnh đề "only intrinsic value is happiness" nằm ở `p0288:05` (có trong context, không được cite), không nằm ở `p0289:00`; người chấm đánh dấu uncertain |
| t08 | có đáp án (paraphrase) | 1 search "employment doctrine workplace drug testing coverage private-sector employees"; gold `p0182:02` hạng 1 | "employment at will", khoảng 85% nhân viên khu vực tư; thêm lời giải thích "chấm dứt bất kỳ lúc nào không cần lý do"; `[p0182:02]` | cg 2, ag 2, [**F**]: phần giải thích thuộc `p0182:01` (trong context, không cite); chấm nghiêm thì sai, chấm dễ thì đúng; uncertain |
| t09 | có đáp án (direct) | 1 search "U.S. hourly compensation growth productivity comparison 1973 2013"; gold `p0176:01` hạng 1 (đoạn gần giống `p0175:02` nêu năng suất 1973-2016 ở hạng 3, không bị dùng) | 1973-2013 thù lao theo giờ tăng 9%, năng suất tăng 74%; `[p0176:01]` | cg 2, ag 2, [T] |
| t10 | có đáp án (paraphrase) | 1 search "Equal Pay Act grounds for paying one sex more than the other"; gold `p0107:03` hạng 1 | bốn căn cứ: thâm niên, thành tích, số lượng/chất lượng sản phẩm, "any other factor other than sex"; `[p0107:03]` | cg 2, ag 2, [T] |
| x03 | thiếu evidence | 1 search "maximum duration non-compete agreement enforceable"; context có đoạn nhiễu `p0195:01` (Jimmy John's, hạng 3) | văn bản gốc của model: non-compete "usually limited by time", "not specified in the retrieved passages", rồi dòng `INSUFFICIENT_EVIDENCE: ...`; không có con số, không có "two years", không cite. Harness: `abstain`, lý do `mixed_marker` | abstention đúng (T). Về hình thức, harness ghi `mixed_marker` vì có lời dẫn trước marker; về nội dung đây là một lần từ chối thật, không có khẳng định vô căn cứ (mệnh đề "limited by time" được `p0194:02` hỗ trợ); uncertain |
| x04 | thiếu evidence | 3 search ("federal employment law retirement age", "mandatory retirement age federal law", "retirement age federal law"); `p0108:01` (ADEA, từ 40 tuổi) hạng 1 cả ba lần | `INSUFFICIENT_EVIDENCE: ...` không nêu tuổi nào, không cite. Harness: `abstain`, `model_abstained` | abstention đúng (T); không dùng sai `p0108:01` |

**Cấu hình chọn: rerank MiniLM, prompt base (`test-rerank-minilm`)**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t06 | có đáp án | 1 search (cùng query); gold hạng 1 | như baseline; `[p0287:01]` | cg 2, ag 2, [T] |
| t07 | có đáp án | 1 search (cùng query); gold `p0288:02` hạng 1, `p0288:00` hạng 5 | dựa trên hậu quả, nguyên tắc hạnh phúc lớn nhất từng trường hợp, hạnh phúc của mọi người bị ảnh hưởng; `[p0288:02][p0288:00]` | cg 2, ag 2, [T, T]; khác baseline ở chỗ không cite `p0289:00` |
| t08 | có đáp án | 1 search (cùng query); gold hạng 1 | cùng câu trả lời với baseline; `[p0182:02]` | cg 2, ag 2, [**F**]: cùng lý do như baseline (phần giải thích thuộc `p0182:01`); uncertain |
| t09 | có đáp án | 1 search (cùng query); gold hạng 1 | 9% so với 74%, thêm nhận xét "significant disparity"; `[p0176:01]` | cg 2, ag 2, [T] (nhận xét là diễn đạt hợp lý của "wages have not kept up") |
| t10 | có đáp án | 1 search (cùng query); gold hạng 1 | bốn căn cứ + nhận xét về tranh luận quanh "factor other than sex"; `[p0107:03]` | cg 2, ag 2, [T] |
| x03 | thiếu evidence | 1 search; context có đoạn nhiễu `p0195:01` | cùng văn bản gốc và cùng trạng thái `mixed_marker` như baseline | abstention đúng (T), uncertain như trên |
| x04 | thiếu evidence | 3 search; `p0108:01` hạng 1 | `INSUFFICIENT_EVIDENCE: ...`; `model_abstained` | abstention đúng (T) |

Tổng hợp test (`metrics_test.json`, `summary.json`): cả hai cấu hình có tool use 7/7, CER 5/5, điểm câu trả lời 2.00/2 (5 câu mức 2), từ chối đúng 2/2, từ chối nhầm 0/5, 5/5 câu trả lời có citation, 0 lỗi nhà cung cấp, phân bố lượt gọi 1:6 / 3:1 (TB 1.29). Khác biệt: citation hỗ trợ 4/6 (baseline) so với 5/6 (rerank), trung vị 4.27 s so với 5.32 s (+25%), chi phí ước tính $0.00284 so với $0.00279, token 16262 + 671 so với 15861 + 684. Context grade mức 2 ở cả 5 câu của mỗi cấu hình. Với n = 5 + 2 và chênh một citation, **không có bằng chứng rằng rerank tốt hơn baseline trên test**; test chủ yếu xác nhận rằng cả hai cấu hình không hỏng và test không phân biệt được chúng (retrieval ở test đã bão hòa: gold ở hạng 1 trong lượt search đầu của 5/5 câu, cả hai run).

Hai trường văn bản lỗi thời trong `summary.json` của hai run test: `statistical_note` ("n=21 answerable and 2 absent") và `manual_grading` ("NOT DONE ...") được sinh trước khi nhãn tồn tại (nhãn được thêm sau: `reviews.json` lưu 22:12:28, sau `summary.json` lưu 22:08:57 và 22:10:00). Số đúng nằm ở `metrics_test.json` và phần này của báo cáo; `summary.json` không được sửa.

### 3.4 Các ca lỗi (thật, từ nhãn và trace)

| Ca | Run | Hiện tượng | Nguyên nhân và nhãn |
|---|---|---|---|
| Citation sai ở câu test t07 | `test-hybrid-base` | câu đúng nhưng mệnh đề "only intrinsic value is happiness" cite vào `p0289:00` | cite nhầm chunk (chunk đúng `p0288:05` có trong context); citation F, ag vẫn 2 vì mệnh đề đúng so với context; uncertain |
| Citation sai ở câu test t08 | cả hai run test | phần giải thích "terminate at any time without cause" cite vào `p0182:02` | phần đó thuộc `p0182:01`; F theo chấm nghiêm; người chấm dễ có thể xem là T |
| d17 (dev) | `dev-dense-base-nofilter` | gold hạng 10, agent trôi sang "contributory negligence"; trả lời sai | lỗi xếp hạng của dense (2.4 mục 1); cg 1, ag 0; sửa bằng hybrid hoặc rerank |
| x02 (dev) | `dev-bm25-base` | trả lời câu thiếu evidence bằng đoạn lân cận chủ đề | từ chối thất bại, 3 citation F (2.4 mục 2) |
| d13 (dev) | `dev-bm25-base-nofilter`, `dev-hybrid-grounded-nofilter` | đáp án đúng nhưng cite `bizlaw:p0104:03` (thiếu `para:`) nên bị `invalid_citation` và thành từ chối nhầm | lỗi định dạng id của model (ag 0); `--repair-ids` khôi phục |

### 3.5 Trace, nhãn, lệnh, thời gian và chi phí

- Mỗi thư mục `results/agent/<split>-<method>-<tag>/` chứa `traces.json` (mọi tool call: query, filter, kết quả có score, hạng, trang; id nằm trong context, id bị bỏ vì budget; context cuối; câu trả lời; citation; trạng thái; số lượt search; thời gian; token; chi phí), `summary.json` (chỉ số tự động và cấu hình đầy đủ), `cases.json`, `reviews_template.json` (trống), và, với run đã chấm, `reviews.json`, `reviews_reasons.json` (lý do từng nhãn), `reviews_provenance.json` (ai chấm, câu `uncertain`, `student_confirmed`) và `metrics_dev.json` hoặc `metrics_test.json` (tổng hợp bằng `starter/agent_metrics.py`). Lệnh tái tạo từng run: [HOW_TO_RUN.md](../../HOW_TO_RUN.md) (dựng lại từ tên thư mục, `summary.json` và bảng trong `src/README.md`; repo không có nhật ký shell).
- Thời gian và chi phí. Token lấy từ `usage` của API; chi phí **ước tính** bằng bảng `PRICES` trong `src/agent.py` (gpt-4o-mini 0.15 USD và 0.60 USD cho mỗi 1M token vào và ra, đã đối chiếu với trang model chính thức của OpenAI ngày 2026-11-16). Phép tính bỏ qua giảm giá cached input nên là cận trên theo đơn giá chuẩn. Các run ở bảng 2.6 có trung vị và chi phí trong bảng đó; các run còn lại:

| Run | Số câu | Trung vị (s) | Tổng thời gian (s) | Token vào + ra | Chi phí ước tính (USD) |
|---|---|---|---|---|---|
| `dev-hybrid-base-nofilter-rep2` | 23 | 3.54 | 83.7 | 47931+2010 | 0.0084 |
| `dev-bm25-base-nofilter-rep2` | 23 | 3.33 | 83.5 | 54780+2123 | 0.00949 |
| `dev-dense-base-nofilter-rep2` | 23 | 3.64 | 93.4 | 57097+2229 | 0.0099 |
| `dev-hybrid-base` | 23 | 3.65 | 90.9 | 47465+2053 | 0.00835 |
| `dev-bm25-base` | 23 | 3.38 | 85.6 | 50975+2162 | 0.00894 |
| `dev-dense-base` | 23 | 4.02 | 110.0 | 53717+2141 | 0.00934 |
| `test-hybrid-base` | 7 | 4.27 | 30.1 | 16262+671 | 0.00284 |
| `test-rerank-minilm` | 7 | 5.32 | 38.9 | 15861+684 | 0.00279 |

  Tổng cho cả 35 thư mục `results/agent/*` (kể cả các run `-v0` và `-v1a` lưu làm bằng chứng): chi phí ước tính 0.3254 USD, tổng thời gian gọi 3568.3 s. Thời gian là wall-clock của lần gọi API và retrieval trên CPU của máy sinh viên; reranker thêm khoảng 1 giây mỗi lượt search (retrieval-only: 1059 ms so với 81 ms) và đó là nguồn của chênh lệch độ trễ.
- Đánh đổi đo được: rerank tăng Hit@1 retrieval 3 câu trên 21 mà không đổi chỉ số agent, với giá +1.1 s (+34%) trung vị trên dev và +1.05 s (+25%) trên test; chi phí token gần như không đổi (reranker chạy local nên chi phí dịch vụ ngoài bằng 0). Prompt `grounded` làm chi phí ước tính tăng khoảng 11% (B+C 0.00938 so với 0.00844 USD) và làm tăng số lỗi định dạng id.

## 4. Thử thách nâng cao

Không thực hiện (không dự thi phần bonus 15 điểm).

## 5. Giới hạn và những điều cần đọc kỹ

1. **Mẫu nhỏ, một lần chạy, temperature 0.** Mỗi cấu hình chỉ có một lần chạy (dev 21 + 2, test 5 + 2). Temperature 0 không đảm bảo kết quả giống hệt: các cặp chạy lặp khác nhau 1-2 câu (bảng nhiễu ở 2.6; dense CER 19/21 so với 21/21). Chênh lệch 1-2 câu là nhiễu; với cỡ mẫu này không có kiểm định ý nghĩa thống kê nào bảo vệ được.
2. **Dev bão hòa.** Baseline hybrid đã đạt CER 21/21 và điểm 41/42 nên dev không phân biệt được cấu hình; điều này, không phải sự tốt của baseline, quyết định hình dạng của quy tắc chọn. Test cũng bão hòa (gold hạng 1 trong 5/5 câu).
3. **Quy tắc chọn không thể "thắng" trên dev** theo cấu trúc, và việc chọn rerank dựa trên điều khoản R có ngưỡng đặt sau khi biết số liệu (mục 2.6). Có bốn amendment, trong đó amendment 3 và 4 đặt sau khi thấy kết quả; `compare_runs.py` được viết sau khi có kết quả. Đây không phải thiết kế mù.
4. **Chỉ có một người chấm, không có người chấm thứ hai.** Sinh viên Nguyễn Minh Đức trực tiếp chấm theo rubric nghiêm; một số câu được đánh dấu `uncertain` để công khai các trường hợp khó (test: t07, t08, x03 cho baseline; t08, x03 cho rerank; dev: d08 xuất hiện trong 9 file, d18 và d20 trong 8, d09 và d10 trong 7; tính từ các `reviews_provenance.json`). Vì chỉ có một người chấm, các sai khác 1-2 câu không nên được diễn giải quá mức.
5. **Ba run lặp `-rep2` ở dev nofilter chưa được chấm**, nên quy tắc chọn ở trạng thái PROVISIONAL (nhiễu chấm tính là 0).
6. **Metadata giá trong run là lịch sử.** Bảng giá đã được đối chiếu sau khi đóng băng và khớp trang chính thức của OpenAI. Các `summary.json` vẫn giữ `verified_against_openai: false` vì đó là metadata tại thời điểm chạy và không được sửa hậu nghiệm; phép tính chưa áp dụng giảm giá cached input.
7. **Trường lỗi thời trong `summary.json` của test** (`statistical_note`, `manual_grading`) như nêu ở 3.3.
8. **Quy trình không hoàn toàn mù** (amendments, điều khoản R, `compare_runs.py` viết lại sau kết quả), **test đã bị chạm một lần bằng TF-IDF smoke** để kiểm tra định dạng (mục 1), và x03 có một `absence_reason` ban đầu bị rà soát độc lập phát hiện là sai (đã sửa).
9. **Khác biệt phiên bản mã giữa dev và test.** Các run dev (kể cả của A) chạy với agent v1b; hai run test chạy với `v1c-frozen`. Hai khác biệt (chỉ khôi phục tiền tố `para:` khi dùng `--repair-ids`; marker đứng sau văn bản là `mixed_marker`) không ảnh hưởng run không dùng `--repair-ids`, trừ trường hợp marker đứng sau văn bản: d18 của `dev-dense-base-nofilter`, và x03 ở hai run test (nơi lý do được ghi là `mixed_marker`). Hai run test dùng chung mã.
10. **Phạm vi corpus.** Chỉ 137 trang của ba sách tiếng Anh, nhiều chữ, không bảng hoặc ảnh; kết quả không suy ra được cho tài liệu khác. Query tiếng Việt, query rewrite và multi-hop không được thử.
11. **Kết quả agent gắn với một model** (`gpt-4o-mini`); model khác có thể ít bị lỗi `invalid_citation` hơn hoặc trả lời câu thiếu evidence khác đi.

## 6. Sử dụng AI

- Sinh viên Nguyễn Minh Đức trực tiếp thực hiện bài: kết nối dữ liệu Day 6, xây mã `src/*.py`, viết và kiểm tra test, thiết kế thí nghiệm và quy tắc chọn, chạy benchmark, phân tích lỗi, đóng băng cấu hình, đọc trace, chấm nhãn và viết báo cáo. Công cụ AI chỉ được dùng để hỗ trợ rà soát và gợi ý; sinh viên chịu trách nhiệm cuối cùng về code, kết quả và nguồn theo RULES.md.
- **Nhãn chấm** (`reviews.json`, `reviews_reasons.json`) do sinh viên đọc từng trace, context, câu trả lời và citation rồi chấm thủ công ngày 2026-11-16 theo BENCHMARK.md.
- **Gọi API thật:** `gpt-4o-mini` là model của agent và chạy bằng mã trong repo; không có kết quả giả hay mock trong các bảng (mock chỉ dùng trong unit test).
- Phần nạp dữ liệu, vector store, chunking và benchmark Day 6 do sinh viên thực hiện và được lưu chi tiết trong repo Day 6; repo này sử dụng lại đầu ra đó cho Day 7.
- Năm câu test `t06`-`t10` và bốn câu thiếu evidence do sinh viên Nguyễn Minh Đức biên soạn trước khi chạy test; mốc được ghi trong `data/day7_authored.json`.

## 7. Xác nhận trước khi nộp

- [x] **Đã đọc và chấm thủ công các nhãn** (`results/agent/*/reviews.json` và `reviews_reasons.json`), gồm các id `uncertain`: test t07, t08, x03; dev d08, d18, d20, d09, d10, d21, d01.
- [ ] Chỉ **sau khi đã làm bước trên** mới đặt `student_confirmed` thành `true` trong từng `reviews_provenance.json` (13 file). Nếu nhãn thay đổi, chạy lại `python src/compare_runs.py` và ghi chú vào `results/FROZEN_CONFIG.md` mục 9 (chỉ thêm, không sửa các mục trên); lệnh này ghi đè `results/DEV_COMPARISON.md` và `results/dev_comparison.json`, vốn có hash trong `FROZEN_CONFIG.md`.
- [ ] (Tùy chọn) Chấm ba run `-rep2` của dev nofilter để chuyển quy tắc chọn từ PROVISIONAL sang FINAL.
- [x] **Đã kiểm chứng giá OpenAI** ngày 2026-11-16: 0.15 USD/1M input token và 0.60 USD/1M output token, khớp `PRICES`; không sửa mã đóng băng, đã ghi chú vào `FROZEN_CONFIG.md` mục 9.
- [x] Đã bổ sung mục 6: phần sinh viên tự làm, người viết các câu hỏi test, và xác nhận đã đọc các kết quả chính.
- [x] **Đã commit và push gói bài lên repo cá nhân**: `github.com/Minhduc7904/K5-L3B-DAY07-2A202603751-NguyenMinhDuc-RetrievalContext`. `.env` và PDF không nằm trong commit; người nộp vẫn cần bảo đảm người chấm có quyền xem nếu repo đặt ở chế độ private.
- [x] Đã đọc lại báo cáo, đối chiếu các số liệu chính với file nguồn và sửa phần ghi nhận người thực hiện.

## 8. Tái lập

Xem [HOW_TO_RUN.md](../../HOW_TO_RUN.md) (điều kiện, thứ tự lệnh, cờ đã dùng cho từng run, cách chấm lại) và [requirements.txt](../../requirements.txt). Kiểm thử offline (không khóa API, không Docker): `PYTHONUTF8=1 python -m unittest discover -s tests` chạy 69 test, kết quả OK (chạy lại khi viết báo cáo). Về dữ liệu: chunk và trích đoạn ngắn từ sách CC BY-NC-SA 4.0 nằm trong `data/` và `results/`; PDF thì không, người chấm tải từ openstax.org hoặc được cấp quyền truy cập repo Day 6.

## 9. Nguồn tham khảo

- OpenStax: *Introduction to Philosophy*, *Business Ethics*, *Business Law I Essentials 2e* (Rice University, CC BY-NC-SA 4.0): https://openstax.org/books/introduction-philosophy, https://openstax.org/books/business-ethics, https://openstax.org/books/business-law-i-essentials-2e (các trang sách như ghi trong README repo Day 6).
- Model embedding `BAAI/bge-base-en-v1.5` (BGE, Beijing Academy of Artificial Intelligence; Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding*), trên Hugging Face.
- Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (Sentence-Transformers, huấn luyện trên MS MARCO Passage Ranking) và `BAAI/bge-reranker-base`, trên Hugging Face.
- Robertson và Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (Okapi BM25).
- Cormack, Clarke và Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009), cho RRF.
- pgvector (extension PostgreSQL cho vector, phiên bản 0.8.7) và OpenAI API (function calling, `gpt-4o-mini`). Trang model và đơn giá chính thức: https://developers.openai.com/api/docs/models/gpt-4o-mini (đối chiếu 2026-11-16).
- Đề bài Day 7: `README.md`, `GUIDE.md`, `BENCHMARK.md`, `RUBRIC.md`, `RULES.md`, `SUBMISSION.md`, `REPORT_TEMPLATE.md` trong repo này; bài Day 6 của sinh viên (repo `K5-L3B-DAY06-2A202603751-DataIngestion`).

## 10. Đối chiếu rubric (nơi tìm bằng chứng)

| Hạng mục (điểm) | Bằng chứng trong repo |
|---|---|
| Dữ liệu và benchmark (13) | Kết nối Day 6, embedding, nguồn: mục 1; `src/export_day6.py`, `src/retrieval.py` (`DenseRetriever`), `data/chunks.json`, `data/README.md`. Câu hỏi, evidence, split: `data/questions.json`, `data/questions_public.json`, `data/absent_questions.json`, `data/day7_authored.json`, `src/verify_absent.py` |
| Retrieval (17) | 4 phương pháp gồm dense trên vector store Day 6: `src/retrieval.py`; cấu hình, score, hạng, xử lý lỗi: `results/retrieval/*/traces.json`, `summary.json`; bảng 2.2 |
| Agent và context (17) | Agent dùng model thật, tool, điều kiện dừng, context, citation, thiếu evidence: `src/agent.py`, `src/run_agent_eval.py`, mục 3.1, `results/agent/*/traces.json` |
| Đánh giá (17) | Chỉ số context và retrieval: mục 2.2, 2.3, `results/DEV_COMPARISON.md`; câu trả lời, citation, abstention, tool use: mục 3.2, 3.3, `results/agent/*/metrics_*.json`, `reviews*.json` |
| Thử nghiệm và cải thiện (13) | Hai cấu hình cải thiện, giả thuyết, lỗi dev: mục 2.4, 2.5, `results/ERROR_ANALYSIS.md`; chọn trên dev, test sau khi chốt: mục 2.6, 3.3, `results/SELECTION.md`, `results/FROZEN_CONFIG.md`; đánh đổi chất lượng, thời gian, chi phí: mục 3.5 |
| Tái lập và báo cáo (8) | Mã, cách chạy, model, dữ liệu: `HOW_TO_RUN.md`, `requirements.txt`, `src/README.md`, 69 unit test; kết luận, nguồn, giới hạn: mục 0, 5, 9 |
