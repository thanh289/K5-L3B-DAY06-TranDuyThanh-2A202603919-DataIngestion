#  Đánh giá và phân tích kết quả

## Phạm vi và phương pháp


Cả ba cấu hình dùng cùng 20 câu hỏi tiếng Việt (15 dev, 5 test), cùng model OpenAI `text-embedding-3-small` (1.536 chiều) và tìm kiếm cosine chính xác trong PostgreSQL + pgvector. A dùng cửa sổ 500 ký tự, overlap 50. B nhóm đoạn theo dấu hiệu cấu trúc, tối đa 500 ký tự. C dùng chính index B và thêm filter `section` khi câu hỏi khai báo. Bộ benchmark có 7 câu khai báo filter, trong đó 2 câu test.


## 1. Độ chính xác truy xuất

Tính lại các chỉ số từ 60 trace cho kết quả khớp với Step 7. Một hit tự động yêu cầu đúng nguồn, đúng trang và chứa nguyên văn đoạn gold đã chuẩn hóa trong một chunk.

| Cấu hình | Dev Hit@1 | Dev Evidence Hit@3 | Test Hit@1 | Test Evidence Hit@3 | Đúng trang trong top 3 (dev / test) |
|---|---:|---:|---:|---:|---:|
| A | 5/15 | 9/15 | 1/5 | 2/5 | 14/15 · 5/5 |
| B | 7/15 | 10/15 | 2/5 | 3/5 | 14/15 · 5/5 |
| C | 7/15 | 10/15 | 2/5 | 3/5 | 14/15 · 5/5 |

Kiểm tra index trước truy xuất cho thấy A có chunk chứa trọn đoạn gold ở 17/20 câu. Ba câu thiếu là `q03`, `q11`, `q20`. B và C có độ phủ 19/20, thiếu `q07`. Mỗi cấu hình có thêm 6 câu trượt top 3 dù index có chunk chứa trọn gold. Riêng `q10`, cả ba cấu hình đều bỏ lỡ trang gold 49 trong top 3.

## 2. Tính mạch lạc của chunk

Quy tắc lấy mẫu áp dụng trước khi gán nhãn coherence. Chọn 10 trang đầu theo thứ tự page ID, từ PDF 29 đến 38. Mỗi vị trí là đầu trang trong văn bản trích xuất. Ở mỗi cấu hình, chấm chunk đầu tiên bao phủ vị trí đó.  Điểm 0 là không hiểu được ý, 1 là hiểu một phần hoặc thiếu ngữ cảnh, 2 là đủ ngữ cảnh và giữ trọn ý.

| Trang PDF | Vị trí đầu trang | A | Lý do A | B | C | Lý do B/C |
|---:|---|---:|---|---:|---:|---|
| 29 | Tiêu đề chương và các thông điệp chính | 1 | Thông điệp đầu đủ ý, thông điệp về khả năng chi trả bị cắt giữa từ. | 2 | 2 | Tiêu đề và thông điệp đầu về tăng người dùng Internet và chênh lệch thu nhập đều trọn ý. |
| 30 | Đoạn mở về khoảng cách số | 1 | Ví dụ châu Phi rõ, câu tiếp theo về rào cản sử dụng bị cắt giữa ý. | 2 | 2 | Đoạn về khoảng cách số và ví dụ châu Phi đọc được độc lập. |
| 31 | Chú thích Hình 1.1 | 1 | Tiêu đề rõ, các giá trị cuối chunk rời khỏi nhãn biểu đồ. | 1 | 1 | Có tiêu đề và nguồn của hình, thiếu dữ liệu để đọc xu hướng. |
| 32 | Chú thích Hình 1.2 | 1 | Biết hình nói về khoảng cách số, khó gắn dãy số với từng nhóm. | 1 | 1 | Có tiêu đề, chú giải và dãy số, vẫn thiếu liên kết giữa số và nhóm. |
| 33 | Chú thích Hình 1.3 | 1 | Nhận ra hai loại băng rộng, các trục bị làm phẳng và chú giải bị cắt. | 1 | 1 | Giữ tiêu đề, nhãn và nguồn, khó nối các giá trị với từng chuỗi. |
| 34 | Đoạn tiếp nối về điện thoại thông minh | 1 | Mở đầu thiếu chủ ngữ từ trang trước, cuối chunk cắt giữa từ. | 1 | 1 | Giữ được các câu so sánh tiếp theo, câu mở đầu vẫn thiếu chủ ngữ. |
| 35 | Đề mục về tốc độ Internet | 1 | Ý chính rõ nhưng câu chuyển sang băng thông quốc tế bị cắt. | 2 | 2 | Giữ đề mục, mốc đại dịch và toàn bộ so sánh giữa các nhóm thu nhập. |
| 36 | Chú thích Hình 1.9 | 1 | Nhận ra hai loại lưu lượng nhưng trục và nhãn rời nhau, dòng nguồn bị cắt. | 1 | 1 | Giữ được tên đại lượng và đơn vị, các số không còn gắn với chuỗi. |
| 37 | Chú thích Hình 1.10 | 1 | Tiêu đề và ghi chú rõ, chuỗi số và tháng ở cuối bị tách khỏi biểu đồ. | 1 | 1 | Có chủ đề và nguồn, thiếu dữ liệu thể hiện ứng dụng nào tăng. |
| 38 | Đoạn về ứng dụng trong thời gian phong tỏa | 1 | Đọc được xu hướng chính, câu tiếp theo bị cắt giữa cụm từ. | 2 | 2 | Hai nhận xét về tăng trưởng lượt tải và mức nghiêm ngặt của phong tỏa đều đủ ý. |

**Điểm trung bình** A **1,00/2** (10 mẫu). B và C **1,40/2** mỗi cấu hình (10 mẫu). Mẫu tập trung ở đầu chương 1 và có nhiều hình. Kết quả phản ánh cách hai phương pháp xử lý các vị trí này. Với cả A và B, biểu đồ bị làm phẳng khi trích PDF, khiến các giá trị khó diễn giải từ chunk.

## 3. Tính hữu dụng của metadata

Kiểm tra đủ 45 hit test. Mỗi hit được nối theo chunk ID với chunk tái tạo từ PDF, so sánh nội dung, `document_id`, `source`, `page`, `section`, `printed_page` và `page_id`. Đối chiếu thêm số trang in với đầu trang trong PDF, và chương với tiêu đề và phạm vi hai chương. Các số dưới đây theo thứ tự hạng 1 đến 3. Tất cả hit trong bảng đều đạt kiểm tra metadata.

| Câu test | A · trang PDF | B · trang PDF | C · trang PDF |
|---|---|---|---|
| q16 | 43 / 37 / 39 | 43 / 66 / 37 | 43 / 66 / 37 |
| q17 | 43 / 44 / 63 | 43 / 43 / 44 | 43 / 43 / 44 |
| q18 | 60 / 55 / 60 | 60 / 60 / 53 | 60 / 60 / 53 |
| q19 | 64 / 64 / 63 | 62 / 63 / 62 | 62 / 63 / 62 |
| q20 | 67 / 67 / 59 | 67 / 59 / 51 | 67 / 59 / 51 |

A, B và C đều đạt **15/15** hit test có metadata đầy đủ và đúng. C trả toàn bộ top 3 trong phạm vi lọc ở **2/2 câu test** và **7/7 câu trên cả bộ**. A và B không áp dụng filter, nên chỉ số tuân thủ filter của hai cấu hình là N/A.

`source` và `page` giúp mở đúng tài liệu và vị trí để kiểm tra câu trả lời. `section` giới hạn phạm vi theo chương, nhưng trong lần chạy này cả 7 câu lọc đã có top 3 của B nằm sẵn trong chương được yêu cầu. Trường này cũng không phân biệt được các đoạn cạnh tranh trong cùng chương, như ở `q19`.

## 4. Chất lượng grounding

Đọc toàn bộ top 3 của từng câu test và cho phép kết hợp bằng chứng giữa các chunk. Điểm 0 là không có bằng chứng cho câu trả lời, 1 là có bằng chứng liên quan nhưng thiếu ý hoặc điều kiện quan trọng, 2 là đủ để trả lời và xác định nguồn.

| Câu | A | Lý do A | B | C | Lý do B/C |
|---|---:|---|---:|---:|---|
| q16 | 1 | Hạng 2 (PDF 37) có ví dụ về làm việc từ xa, học trực tuyến, khám từ xa và mua sắm, nhưng danh sách nhóm ứng dụng bị cắt. | 0 | 0 | Hạng 3 (PDF 37) chỉ có tiêu đề và nguồn của hình; hai hit khác nói về doanh nghiệp hoặc lệnh cấm ứng dụng. Không hit nào cho biết các nhóm ứng dụng tăng sử dụng. |
| q17 | 2 | Hạng 2 (PDF 44) nêu mức giảm doanh số nhỏ hơn ở nhóm đầu tư giải pháp số, trên mọi mức sẵn sàng số trước đại dịch. | 2 | 2 | Hạng 3 (PDF 44) có đủ so sánh và điều kiện. |
| q18 | 2 | Hạng 1 (PDF 60) có đủ 19% và mốc nhanh nhất kể từ năm 2008. | 2 | 2 | Hạng 1 (PDF 60) có đủ cả hai dữ kiện. |
| q19 | 1 | Hạng 3 (PDF 63) nhắc thương mại điện tử trong nhóm dẫn đầu, thiếu vị trí đứng đầu ở cả hai kỳ và mức tăng gần 6 điểm phần trăm. | 1 | 1 | Hạng 1–2 (PDF 62–63) có ngữ cảnh về thương mại điện tử và VC, nhưng thiếu cùng hai dữ kiện. |
| q20 | 2 | Hạng 1 và 2 (cùng PDF 67) ghép được chủ thể Cốc Cốc với đủ ba cách bản địa hóa. | 2 | 2 | Hạng 1 (PDF 67) tự chứa đủ ba cách bản địa hóa và chủ thể. |

| Cấu hình | Số câu điểm 0 | Số câu điểm 1 | Số câu điểm 2 | Trung bình /2 |
|---|---:|---:|---:|---:|
| A | 0 | 2 | 3 | 1,60 |
| B | 1 | 1 | 3 | 1,40 |
| C | 1 | 1 | 3 | 1,40 |

A có cùng 3/5 câu đủ bằng chứng như B và C, và hơn 0,20 điểm trung bình do có bằng chứng một phần ở `q16`. Ở `q20`, A được 2 điểm grounding dù trượt Evidence Hit@3 vì câu gold nằm qua hai chunk. Tương tự trên dev, chunk hạng 1 của B ở `q07` chứa tỷ lệ khoảng 13% và đủ ba quốc gia được hỏi, nhưng thiếu phần cuối câu gold dẫn tới hình. Điểm truy xuất tự động vẫn giữ nguyên.

## 5. Tác động của chiến lược và đánh đổi

| Cấu hình | Số chunk trong index phục vụ cấu hình | Tổng ký tự trong chunk | Độ dài trung bình | Thời gian truy xuất 20 câu |
|---|---:|---:|---:|---:|
| A | 274 | 128.209 | 467,92 | 0,641 giây |
| B | 329 | 116.067 | 352,79 | 0,666 giây |
| C | 329, dùng chung B | 116.067, dùng chung B | 352,79 | 0,674 giây |

**A–B.** B giữ trọn đoạn gold ở thêm hai câu trên toàn bộ index và tăng dev Hit@3 từ 9/15 lên 10/15. Coherence cao hơn trong mẫu đã chấm. B lưu thêm 55 vector (khoảng 20%) dù tổng ký tự trong chunk giảm khoảng 9,5%. B vẫn có thể xếp một chunk đúng trang nhưng thiếu chi tiết lên trước chunk đủ bằng chứng, như ở `q19`. Điểm grounding test cho thấy A vẫn đáng so sánh nếu hệ thống đọc chung nhiều chunk.

**B–C.** Hai cấu hình trả cùng chunk theo cùng thứ tự ở cả 20 câu. Filter có giá trị khi cần giới hạn phạm vi theo yêu cầu có sẵn, song lần chạy này chưa quan sát được lợi ích về thứ hạng. C dùng lại vector của B. Chênh lệch 0,008 giây là tổng thời gian truy xuất 20 câu trong một lần chạy, bao gồm chi phí gọi cơ sở dữ liệu, nên chưa đủ để kết luận về tốc độ.


**Lựa chọn cấu hình.** Dựa trên dev Hit@3 rồi Hit@1, B là cấu hình mặc định hợp lý trong lần chạy này. C phù hợp khi đầu vào yêu cầu phạm vi theo chương. Đây là lựa chọn khi phân tích các kết quả đã có; các tệp Step 7 chưa cho thấy tiêu chí đã được ghi trước khi xem test. Nếu mở rộng thí nghiệm, ưu tiên kiểm tra thứ hạng trên dev và giữ riêng giá trị chẩn đoán của việc đọc nhiều chunk. Dùng một tập test mới để đánh giá độc lập nếu đã tinh chỉnh dựa vào các lỗi test nêu trên.
