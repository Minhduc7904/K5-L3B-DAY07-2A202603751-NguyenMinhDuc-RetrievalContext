# Cách chạy lại bài Day 7 (Agentic RAG)

Hướng dẫn tái lập kết quả trong [report/individual/2A202603751_NguyenMinhDuc.md](report/individual/2A202603751_NguyenMinhDuc.md). Mọi lệnh chạy từ thư mục gốc của repo này, trừ khi ghi khác. Cờ dòng lệnh dưới đây đã được đối chiếu với `argparse` trong `src/*.py`. Các lệnh của từng lần chạy đã báo cáo được **dựng lại** từ tên thư mục kết quả, `summary.json` (`config`, `tag`, `method`) và bảng trong `src/README.md`; repo không có nhật ký lệnh shell.

Môi trường đã dùng: Windows 11 (10.0.22631), Python 3.12.6, Docker Desktop, chỉ CPU (không GPU).

## 1. Điều kiện cần

1. **Python 3.12** (đã chạy với 3.12.6).
2. **Docker** và repo Day 6 của sinh viên (https://github.com/Minhduc7904/K5-L3B-DAY06-2A202603751-DataIngestion): cần container pgvector và dữ liệu đã nạp. Image `pgvector/pgvector:0.8.7-pg16` (PostgreSQL 16, pgvector 0.8.7), cổng `127.0.0.1:55433`, tên container `day6_student_starter-postgres-1`.
3. **Gói Python**: `pip install -r requirements.txt` (phiên bản ghim; xem file đó).
4. **Mạng** cho lần chạy đầu: tải model từ Hugging Face (`BAAI/bge-base-en-v1.5` khoảng 440 MB, `cross-encoder/ms-marco-MiniLM-L-6-v2` khoảng 88 MB; `BAAI/bge-reranker-base` khoảng 1.1 GB chỉ cần cho dòng bge của bảng retrieval-only) và gọi OpenAI API.
5. **Khóa OpenAI** có quyền dùng `gpt-4o-mini` (snapshot đã thấy trong mọi `summary.json`: `gpt-4o-mini-2024-07-18`). Chi phí ước tính của cả 35 thư mục kết quả khoảng 0.33 USD theo bảng giá trong `src/agent.py`; mức 0.15 USD/1M input token và 0.60 USD/1M output token đã được đối chiếu với [trang model chính thức của OpenAI](https://developers.openai.com/api/docs/models/gpt-4o-mini) ngày 2026-11-16. Mỗi lần chạy dev 23 câu khoảng 0.008-0.010 USD ước tính và khoảng 80-110 giây gọi model.

### Biến môi trường (file `.env`, chỉ liệt kê tên; không commit file này)

Sao chép `.env.example` thành `.env` rồi điền. `.env` đã nằm trong `.gitignore`.

| Tên | Ý nghĩa |
|---|---|
| `OPENAI_API_KEY` | khóa OpenAI (bí mật; bắt buộc cho `run_agent_eval.py`) |
| `OPENAI_MODEL` | model của agent (mặc định `gpt-4o-mini`) |
| `DATABASE_URL` | chuỗi kết nối tới pgvector của Day 6 (xem `.env.example`) |
| `EMBEDDING_MODEL` | model embedding của query; phải là model đã dựng index (`BAAI/bge-base-en-v1.5`) |
| `EMBEDDING_DIMENSIONS` | số chiều embedding (768) |
| `CHUNK_STRATEGY` | chiến lược chunk của Day 6 dùng cho Day 7 (`B`, 662 dòng) |

Giá trị trong môi trường (biến shell) ghi đè giá trị trong `.env` đối với `DATABASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `CHUNK_STRATEGY`, `OPENAI_API_KEY` và `OPENAI_MODEL` (xem `load_env` trong `src/retrieval.py` và `load_llm_env` trong `src/agent.py`).

## 2. Dữ liệu và quyền sử dụng

Ba giáo trình OpenStax, giấy phép **CC BY-NC-SA 4.0** (© Rice University; "Access for free at openstax.org"), dùng phi thương mại cho bài tập; phải ghi nguồn và chia sẻ lại cùng giấy phép. **PDF không nằm trong repo này.** Tải bản PDF miễn phí (mục *Download a PDF* trên trang sách) và đặt vào `data/` của repo Day 6 với đúng tên file:

| `source_id` | Sách | Trang sách | Tên file |
|---|---|---|---|
| `phil` | Introduction to Philosophy | https://openstax.org/books/introduction-philosophy | `Introduction_to_Philosophy-WEB.pdf` (421 trang) |
| `bizethics` | Business Ethics | https://openstax.org/books/business-ethics | `business-ethics_-_WEB.pdf` (355 trang) |
| `bizlaw` | Business Law I Essentials 2e | https://openstax.org/books/business-law-i-essentials-2e | `Business_Law_I_Essentials_2e_-_WEB.pdf` (164 trang) |

Bản PDF đã dùng có ngày tạo 2026-04-20 (phil, bizethics) và 2026-06-11 (bizlaw), tải ngày 2026-11-14 (`data/manifest.csv` của repo Day 6). Nếu tải bản khác, số trang có thể lệch: số trang trong benchmark là số trang PDF vật lý đánh từ 1. `data/chunks.json` và `results/` chứa trích đoạn từ sách; chỉ chia sẻ với người chấm và giữ dòng ghi nguồn trên.

## 3. Dựng vector store Day 6 (một lần)

Trong thư mục `starter/` của repo Day 6 (PowerShell):

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait    # PostgreSQL 16 + pgvector 0.8.7 tại 127.0.0.1:55433
python scripts/check_setup.py                          # kỳ vọng: "Setup OK"
python -m src.run ingest                               # nạp 3 PDF: chunk A và B, embedding bge-base-en-v1.5 (khoảng 4 phút trên CPU)
```

Kiểm tra: bảng `chunks` có đúng 662 dòng với `strategy = 'B'` (`DenseRetriever` trong `src/retrieval.py` từ chối chạy nếu số dòng khác số chunk của `data/chunks.json`). Nếu container Day 6 đã chạy và đã nạp, chỉ cần `docker ps` thấy `day6_student_starter-postgres-1` ở trạng thái healthy.

## 4. Trình tự lệnh của repo Day 7

Cài gói và tạo `.env` (mục 1), rồi:

```powershell
$env:PYTHONUTF8 = "1"      # bash: export PYTHONUTF8=1
```

### 4.1 Xuất dữ liệu từ Day 6 (`src/export_day6.py`)

```bash
python src/export_day6.py --day6 <duong-dan-repo-Day6> --output data
```

Đọc bảng `chunks` (strategy B) qua `DATABASE_URL` (lấy từ biến môi trường hoặc `<day6>/starter/.env`) và `<day6>/benchmark/questions.json`; ghi `data/chunks.json`, `data/questions.json`, `data/questions_public.json`. Câu test Day 6 (t01-t05) được chuyển thành dev d17-d21. **Yêu cầu có sẵn** `data/day7_authored.json` (5 câu test t06-t10) và `data/absent_questions.json` (x01-x04). Lệnh **ghi đè** ba file đầu; để chỉ so sánh với bản đã nộp, dùng `--output` là một thư mục tạm có chép sẵn hai file đầu vào kia, rồi so SHA-256 với `results/FROZEN_CONFIG.md` mục 2 (các hash của `data/` chỉ khớp nếu dữ liệu Day 6 giống hệt).

### 4.2 Kiểm tra câu thiếu evidence (`src/verify_absent.py`)

```bash
python src/verify_absent.py --pdf-dir <duong-dan-repo-Day6>/data
```

Quét mọi regex `absence_check` trên 662 chunk và toàn bộ trang của 3 PDF (cần PyMuPDF). Kết quả đã quan sát khi viết báo cáo: `scanned=1602 texts; problems=0`, thoát mã 0. Cờ khác: `--chunks data/chunks.json`, `--absent data/absent_questions.json`, `--chunks-only` (chỉ quét chunk; script nói rõ là không đủ để khẳng định cả sách không có đáp án).

### 4.3 Benchmark retrieval-only (`src/eval_retrieval.py`)

Cờ: `--split {dev,test}`, `--filters {none,question}` (mặc định `none`), `--methods` (mặc định `dense,bm25,hybrid`), `--reranker` (`minilm`, `bge` hoặc tên model), `--pool` (30), `--output`, `--final` (bắt buộc cho test; test chưa được chạy retrieval-only). Các thư mục kết quả đã báo cáo:

```bash
python src/eval_retrieval.py --split dev --filters none                 # results/retrieval/dev-filters-none
python src/eval_retrieval.py --split dev --filters question             # results/retrieval/dev-filters-question
python src/eval_retrieval.py --split dev --filters none     --methods dense,bm25,hybrid,rerank --reranker minilm --pool 30 --output results/retrieval/dev-filters-none-rerank-minilm
python src/eval_retrieval.py --split dev --filters question --methods dense,bm25,hybrid,rerank --reranker minilm --pool 30 --output results/retrieval/dev-filters-question-rerank-minilm
python src/eval_retrieval.py --split dev --filters none     --methods dense,bm25,hybrid,rerank --reranker bge --pool 30 --output results/retrieval/dev-filters-none-rerank-bge
python src/eval_retrieval.py --split dev --filters question --methods dense,bm25,hybrid,rerank --reranker bge --pool 30 --output results/retrieval/dev-filters-question-rerank-bge
```

Chạy dev **ghi đè** thư mục `--output` nếu đã có; để không mất kết quả đã nộp, đặt `--output` sang thư mục mới. Độ trễ (ms) phụ thuộc máy nên không khớp từng chữ số; Hit@k và MRR là tất định.

### 4.4 Chạy agent (`src/run_agent_eval.py`)

Cờ: `--split {dev,test}`, `--method {dense,bm25,hybrid,rerank}` (bắt buộc), `--reranker`, `--pool`, `--repair-ids`, `--prompt {base,grounded}`, `--tag`, `--top-k` (5), `--budget` (10000 ký tự), `--max-searches` (3), `--filters {none,question}` (**mặc định `question`**), `--only`, `--max-cost` (1.50 USD), `--output`, `--overwrite`, `--final`. Kết quả vào `results/agent/<split>-<method>-<tag>/`; dev không ghi đè nếu chưa có `--overwrite`. Mọi lần chạy dùng top-k 5, budget 10000, `--max-searches 3` (mặc định).

Các run **dev** đã báo cáo (`--split dev` cho mọi dòng; agent v1b):

| Thư mục kết quả | Cờ còn lại |
|---|---|
| `dev-{dense,bm25,hybrid}-base` | `--method M --tag base` (filters = question) |
| `dev-{dense,bm25,hybrid}-base-rep2` | `--method M --tag base-rep2` |
| `dev-{dense,bm25,hybrid}-base-nofilter` | `--method M --tag base-nofilter --filters none` |
| `dev-{dense,bm25,hybrid}-base-nofilter-rep2` | `--method M --tag base-nofilter-rep2 --filters none` |
| `dev-rerank-minilm-nofilter` (A) | `--method rerank --reranker minilm --pool 30 --tag minilm-nofilter --filters none` |
| `dev-rerank-minilm` | `--method rerank --reranker minilm --pool 30 --tag minilm` |
| `dev-hybrid-grounded-nofilter` (B) | `--method hybrid --prompt grounded --tag grounded-nofilter --filters none` |
| `dev-hybrid-repair-nofilter` (C) | `--method hybrid --repair-ids --tag repair-nofilter --filters none` |
| `dev-hybrid-repair` | `--method hybrid --repair-ids --tag repair` |
| `dev-hybrid-grounded-repair-nofilter` (B+C) | `--method hybrid --prompt grounded --repair-ids --tag grounded-repair-nofilter --filters none` |
| `dev-hybrid-grounded-repair` | `--method hybrid --prompt grounded --repair-ids --tag grounded-repair` |
| `dev-rerank-minilm-grounded-repair-nofilter` (A+B+C) | `--method rerank --reranker minilm --pool 30 --prompt grounded --repair-ids --tag minilm-grounded-repair-nofilter --filters none` |
| `dev-rerank-minilm-grounded-repair` | `--method rerank --reranker minilm --pool 30 --prompt grounded --repair-ids --tag minilm-grounded-repair` |

Ví dụ đầy đủ: `python src/run_agent_eval.py --split dev --method hybrid --tag base-nofilter --filters none`.

Các thư mục `*-v0` và `*-v1a` là kết quả của các phiên bản agent cũ (v0, v1a) được giữ làm bằng chứng; mã của các phiên bản đó **không còn trong repo** nên không tái tạo được, và `compare_runs.py` bỏ qua chúng. Chạy lại dev bằng mã hiện tại (`v1c-frozen`) sẽ cho kết quả hơi khác bản đã nộp: model API không tất định ở temperature 0 (các cặp chạy lặp khác nhau 1-2 câu), và hai thay đổi của `v1c-frozen` (chỉ khôi phục tiền tố `para:`; marker `INSUFFICIENT_EVIDENCE` đứng sau văn bản là `mixed_marker`).

### 4.5 Hai lần chạy test (chỉ chạy một lần, sau khi đóng băng)

Lệnh trong `results/FROZEN_CONFIG.md` mục 6:

```bash
python src/run_agent_eval.py --split test --final --method hybrid --tag base --filters none --max-cost 0.50
python src/run_agent_eval.py --split test --final --method rerank --reranker minilm --pool 30 --tag minilm --filters none --max-cost 0.50
```

Chương trình **từ chối** chạy test nếu thiếu `--final`, nếu dùng `--only` hoặc `--output`, nếu thư mục kết quả đã tồn tại (`--overwrite` không cứu được), hoặc nếu đã có từ 2 thư mục `results/agent/test-*` (trần chung cho mọi phương pháp). Vì repo nộp đã có hai thư mục đó, lệnh sẽ bị từ chối; muốn chạy lại hãy làm trong một bản sao của repo và xóa hai thư mục `results/agent/test-*` ở bản sao đó (không làm trên bản nộp). Sau mỗi lần chạy, đối chiếu `summary.json`: `agent_version` = `v1c-frozen`, `config.agent_code_sha256`, `system_prompt_sha256`, `chunks_sha256`, `questions_public_sha256` khớp `results/FROZEN_CONFIG.md` mục 2.

### 4.6 Tổng hợp chỉ số tự động và so sánh các run dev (`src/compare_runs.py`)

```bash
python src/compare_runs.py
```

Không có cờ. Đọc mọi `results/agent/*/summary.json` (bỏ qua `*-v0`, `*-v1a`), `reviews.json` nếu có, và `results/retrieval/`; ghi `results/DEV_COMPARISON.md` và `results/dev_comparison.json` (ghi đè; hai file này có SHA-256 trong `results/FROZEN_CONFIG.md`, nên nếu chạy lại hãy ghi chú vào mục 9 của file đó) và in kết quả áp dụng quy tắc của `results/SELECTION.md`. Không gọi API, không đọc test.

### 4.7 Dấu vân tay mã và dữ liệu (`src/freeze_info.py`)

```bash
python src/freeze_info.py          # thêm --json để in JSON
```

In SHA-256 của `src/agent.py`, `src/run_agent_eval.py`, `src/retrieval.py`, `src/compare_runs.py`, `results/SELECTION.md`, `data/chunks.json`, `data/questions_public.json`, `data/questions.json`, `data/absent_questions.json`, hai prompt, phiên bản gói và git HEAD. Không gọi API. Khớp với bảng của `results/FROZEN_CONFIG.md` mục 2 tại thời điểm viết hướng dẫn này.

### 4.8 Kiểm thử đơn vị

```bash
python -m unittest discover -s tests -v
```

Offline, không cần khóa API và Docker. 69 test, kết quả `OK` (đã chạy khi viết hướng dẫn này).

## 5. Chấm lại (regrade)

Mỗi lần chạy agent tạo `reviews_template.json` **trống có chủ đích**: không script hay LLM nào tự chấm. Nhãn hiện có (`reviews.json`, `reviews_reasons.json`) do sinh viên Nguyễn Minh Đức đọc trace và chấm thủ công theo BENCHMARK.md; metadata xác nhận trong `reviews_provenance.json` được chủ repo quản lý riêng.

1. Đọc `traces.json` và `reviews_template.json` của run; với từng câu, đối chiếu đoạn được cite với `answer`/`model_text` (với câu bị `abstain`, đọc cả `model_text` vì `answer` chỉ là câu thay thế của harness).
2. Điền `context_grade` (0/1/2), `answer_grade` (0/1/2), `citation_support` (danh sách true/false cho từng citation, theo thứ tự) và, với câu thiếu evidence, `abstention_correct`; lưu thành `reviews.json` cùng thư mục (chấm nghiêm theo BENCHMARK.md mục 2 và 3).
3. Tổng hợp:

```bash
python starter/agent_metrics.py --cases results/agent/<run>/cases.json --traces results/agent/<run>/traces.json --reviews results/agent/<run>/reviews.json --split dev --output results/agent/<run>/metrics_dev.json
```

Với test dùng `--split test` và `--output .../metrics_test.json`. Cả bốn cờ `--cases`, `--traces`, `--reviews`, `--split`, `--output` đều bắt buộc.
4. Chạy lại `python src/compare_runs.py` để quy tắc chọn dùng nhãn mới (trạng thái FINAL chỉ khi mọi run so sánh và cặp `base-nofilter` / `-rep2` của mọi phương pháp liên quan đều có nhãn).
5. Sau khi hoàn tất lần kiểm tra cuối, chủ repo cập nhật `student_confirmed` trong `reviews_provenance.json`; bước metadata này không làm thay đổi nội dung nhãn.

## 6. Khắc phục sự cố

- `cannot connect to the Day 6 pgvector store ... is Docker up?`: khởi động container Day 6 (mục 3) và kiểm tra `DATABASE_URL`.
- `pgvector holds N 'B' chunks but chunks.json has 662`: chạy `python -m src.run ingest` ở Day 6 hoặc `python src/export_day6.py` để xuất lại chunk.
- `OPENAI_API_KEY is missing`: điền `.env` hoặc đặt biến môi trường.
- `cannot load embedding model ...` / `cannot load cross-encoder ...`: cần mạng lần đầu để tải model từ Hugging Face; không có phương án dự phòng.
- Lỗi mã hóa trên Windows: đặt `PYTHONUTF8=1`.
