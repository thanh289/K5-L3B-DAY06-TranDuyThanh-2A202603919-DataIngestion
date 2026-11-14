# Báo cáo cá nhân — Day 06

- Họ và tên: Trần Duy Thành
- MSSV: 2A202603919
- Lớp: L3B
- Link repo: https://github.com/thanh289/K5-L3B-DAY06-TranDuyThanh-2A202603919-DataIngestion.git
- Commit chốt bài: final_submit

## Dữ liệu và pipeline

Corpus gồm 37 trang nội dung thuộc hai chương của *Digital Progress and Trends Report 2023* (World Bank, 2024, CC BY 3.0 IGO). Lấy các dải 29–45 và 49–68, bỏ phần đầu tài liệu, trang phân cách và tài liệu tham khảo ngoài phạm vi đã chọn. [Manifest](../../data/manifest.json)

Pipeline dùng `pypdf` để trích văn bản theo trang, chuẩn hóa Unicode NFKC và khoảng trắng, giữ xuống dòng để phục vụ chunking. Manifest đếm ký tự bằng PyMuPDF theo quy tắc khoảng trắng khác, còn pipeline dùng `pypdf` và đếm 116.359 ký tự trước khi chia chunk.

Mỗi chunk giữ `document_id`, `source`, `page` và `section`. `page` là số trang vật lý và `printed_page` là số in trên báo cáo. Trường `section` lấy từ ranh giới chương trong PDF để giới hạn phạm vi khi truy xuất.

Mô hình OpenAI `text-embedding-3-small` tạo vector 1.536 chiều. Cấu hình Docker dùng PostgreSQL 16 và pgvector 0.8.7, truy xuất theo cosine chính xác. Môi trường có Python 3.11, OpenAI SDK 3.50.0, pypdf 6.21.4 và psycopg 3.3.8.


## Cấu hình và benchmark

A chia cửa sổ 500 ký tự, overlap 50. B chia theo dấu hiệu đoạn/câu, tối đa 500 ký tự. C dùng chung chunk và vector của B, áp dụng lọc `section` nếu câu hỏi có điều kiện lọc. Ba cấu hình dùng cùng corpus, model và câu hỏi. Benchmark có 15 dev, 5 test và 7 câu có filter (2 câu test). Không có trang gold chung giữa hai tập. Kiểm tra offline tìm thấy tất cả gold evidence trong trang tương ứng sau chuẩn hóa. Các artifact cho thấy dev/test đã được khai báo khi chấm. [Bộ câu hỏi](../../benchmark/questions.json)

## Kết quả

| Cấu hình | Dev Hit@1 | Dev Evidence Hit@3 | Test Hit@1 | Test Evidence Hit@3 | Số chunk | Truy xuất 20 câu |
|---|---:|---:|---:|---:|---:|---:|
| A | 5/15 | 9/15 | 1/5 | 2/5 | 274 | 0,641 giây |
| B | 7/15 | 10/15 | 2/5 | 3/5 | 329 | 0,666 giây |
| C | 7/15 | 10/15 | 2/5 | 3/5 | Dùng chung 329 chunk của B | 0,674 giây |

| Cấu hình | Coherence /2 (10 vị trí) | Metadata đúng đủ (test) | Tuân thủ filter | Grounding (số câu điểm 0/1/2) | Grounding trung bình /2 |
|---|---:|---:|---:|---:|---:|
| A | 1,00 | 15/15 | N/A | 0 / 2 / 3 | 1,60 |
| B | 1,40 | 15/15 | N/A | 1 / 1 / 3 | 1,40 |
| C | 1,40 | 15/15 | 7/7 cả bộ, 2/2 test | 1 / 1 / 3 | 1,40 |

Step 6 ghi 41,04 giây cho lượt nạp kèm smoke test, 603 vector tài liệu mới và 57.214 token API. Step 7 ghi 28,235 giây cho toàn lượt, dùng lại vector tài liệu và tiêu thụ 1.218 token query. Chi phí API ước tính lần lượt là 0,00114428 USD và 0,00002436 USD, tính theo 0,02 USD/1 triệu token đầu vào. Log gộp số đo theo lượt nên chưa tách riêng chi phí nạp cho A và B. [Step 6](../../starter/results/step6_smoke_test.json) · [Step 7](../../starter/results/step7_summary.json) · [Đơn giá OpenAI](https://developers.openai.com/api/docs/models/text-embedding-3-small)

Chunk được upsert theo chiến lược, model và chunk ID. Khi đồng bộ, code xóa ID cũ của tài liệu được nạp trong cùng transaction. Log Step 6 hiện có ghi lượt nạp mới; số đo của lượt nạp lại cần thu từ `step6_reload.json` theo hướng dẫn tái lập.

## Phân tích và kết luận

Ưu tiên Evidence Hit@3 rồi Hit@1 trên dev, B là lựa chọn phù hợp cho lần chạy này. B giữ trọn đoạn gold ở 19/20 câu so với 17/20 của A, đổi lại lưu thêm 55 vector. C thu hẹp đúng phạm vi, nhưng các hit của B đã nằm trong chương được yêu cầu và B/C trả cùng top 3 cho cả 20 câu. Metadata vẫn có ích để truy lại PDF và lọc theo ngữ cảnh.

Ở `q20` (PDF 67), A cắt câu về Cốc Cốc qua hai chunk. Hai chunk đủ để lập câu trả lời khi đọc chung, còn B đưa đủ bằng chứng vào ngay hạng 1. Ở `q19` (PDF 63), B đưa đúng trang vào hạng 2 nhưng thiếu vị trí đứng đầu của thương mại điện tử và mức tăng gần 6 điểm phần trăm. Chunk đầy đủ có trong index, nên lỗi này thuộc về xếp hạng.

Test chỉ có 5 câu và mẫu coherence chỉ gồm 10 vị trí đầu chương 1. [Trace top 3](../../starter/results/step7_traces.json) · [Audit Step 8](../../starter/results/step8_audit.md)
