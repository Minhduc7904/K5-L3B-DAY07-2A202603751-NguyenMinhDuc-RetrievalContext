# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Dữ liệu của bài này (Day 6 → Day 7)

| File | Nội dung | Ai được đọc |
|---|---|---|
| `chunks.json` | 662 chunk cấu hình B của Day 6 (xuất bằng `src/export_day6.py`) | agent, retriever |
| `questions_public.json` | 30 câu hỏi chỉ gồm `id`, `split`, `question`, `filter` | **agent, retriever** |
| `questions.json` | 26 câu có đáp án (21 dev + 5 test): `gold`, `expected_answer`, `type`, `notes` | chỉ code đánh giá |
| `day7_authored.json` | 5 câu test mới viết cho Day 7 (t06–t10), dạng thô trước khi chuyển đổi | chỉ code xuất |
| `absent_questions.json` | 4 câu thiếu evidence: `absence_reason`, `absence_check`, `grader_note`, `related_topic` | chỉ code đánh giá |

- `type`, `notes`, `related_topic`, `absence_reason`, `absence_check`, `decoy_check`, `grader_note` chỉ để phân tích và chấm; không đưa vào prompt hay retriever.
- Số trang là **số trang PDF vật lý (đánh từ 1)**, không phải số in ở chân trang sách.
- `evidence_original` là trích dẫn gốc của Day 6; `evidence` là đoạn y nguyên trong chunk (chỉ khác dấu nháy cong/thẳng).
- Kiểm tra câu thiếu evidence: `python src/verify_absent.py --pdf-dir <thư mục PDF Day 6>`. Câu x03 có một decoy đã biết (Jimmy John's "two years", PDF tr.195) được khai báo trong `decoy_check`.
- **Chia dev/test:** 5 câu test của Day 6 (t01–t05) đã được đánh giá và bàn luận trong báo cáo Day 6, nên không còn độc lập. Chúng được chuyển sang **dev** thành `d17`–`d21` (giữ `day6_id`, `day6_split`). **Test của Day 7 là 5 câu mới `t06`–`t10`** (viết ngày 2026-11-16, trước khi chạy bất kỳ thí nghiệm Day 7 nào) cộng 2 câu thiếu evidence `x03`, `x04`. Dev gồm 21 câu có đáp án và `x01`, `x02`.
- Câu `d21` (Day 6 `t05`) chỉ trả lời được khi có `filter` (`bizethics`); `baseline.py` bỏ qua filter nên kết quả của nó lẫn lỗi retrieval với phạm vi sách.
- Câu `t09` có decoy: `bizethics` tr.175–176 nêu số liệu 1973–2016 gần giống, câu hỏi nêu rõ 1973–2013.
- Đã từng chạy TF-IDF baseline một lần trên 5 câu test mới để kiểm tra định dạng (3/5 Hit@1, 5/5 Hit@3); không câu nào bị sửa theo kết quả đó. Cần nêu điều này trong báo cáo.
