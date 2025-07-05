# Hand Gesture Capture

Ứng dụng nhận diện cử chỉ tay và chụp ảnh tự động bằng Python, sử dụng OpenCV, MediaPipe và PyQt6.

## Tính năng

- Nhận diện số ngón tay giơ lên bằng MediaPipe.
- Chụp ảnh tự động khi phát hiện cử chỉ tay phù hợp.
- Thêm logo, khung ảnh vào ảnh chụp.
- Xem, tải về, xoá ảnh đã chụp ngay trong ứng dụng.
- Giao diện trực quan, dễ sử dụng với PyQt6.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Windows, Linux hoặc macOS

## Cài đặt

1. **Clone dự án:**
    ```sh
    git clone https://github.com/ntdung2212/hand-gesture-capture.git
    cd hand-gesture-capture
    ```

2. **Cài đặt các thư viện cần thiết:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Chạy ứng dụng:**
    ```sh
    python app.py
    ```

## Hướng dẫn sử dụng

- **Tab Camera:** Hiển thị camera, nhận diện cử chỉ tay và chụp ảnh tự động khi giơ 2 ngón tay, dừng khi giơ 4 ngón tay. Có thể chụp thủ công bằng nút "Chụp ngay".
- **Tab Ảnh đã chụp:** Xem, tải về hoặc xoá ảnh đã chụp.
- **Tab Cài đặt:** Tuỳ chỉnh độ phân giải, logo, khung ảnh, delay chụp, số tay nhận diện, độ tin cậy...

## Đóng góp

Mọi đóng góp, báo lỗi hoặc ý tưởng mới đều được hoan nghênh! Hãy tạo issue hoặc pull request.

## Bản quyền

MIT License. Vui lòng ghi nguồn khi sử dụng lại mã nguồn này.
