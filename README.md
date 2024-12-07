1. Cách xử lý khi chạy chương trình:
- Có thể bật các dòng print để debug
- Nếu thấy xử lý 1 câu bị lâu, buffer lớn -> Khả năng cao bị lỗi ở dòng nào đó, kiểm tra dòng số bao nhiêu và check ở file excel:
  - Nếu là kết quả OCR không chính xác (bị thừa chữ, đọc ở hình ảnh): Xóa ô OCR text tại dòng đó (None) hoặc đưa từ này vào black_list
  
2. Cách xử lý file sau khi chạy lần đầu:
- Kiểm tra các chữ màu xanh lá, xanh dương:
  - Nếu đúng có thể bỏ qua, nếu sai thì kiểm tra thêm 2 ô gần nhất để điều chỉnh kết quả OCR text cho hợp lý

