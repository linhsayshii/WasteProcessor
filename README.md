# WasteProcessor - Hệ Thống Phân Loại Rác Thải Thông Minh ♻️

Dự án ứng dụng Trí tuệ nhân tạo (AI/Deep Learning) để tự động nhận diện và phân loại rác thải qua hình ảnh. Hệ thống giúp người dùng dễ dàng xác định loại rác (Tái chế, Nguy hại, Hữu cơ, v.v.) để xử lý đúng cách, góp phần bảo vệ môi trường.

## ✨ Tính Năng Nổi Bật

*   **🧠 Nhận diện đa chủng loại**: Hỗ trợ phân loại 9 nhóm rác thải phổ biến:
    *   Pin / Ắc quy (Rác nguy hại)
    *   Rác điện tử (E-waste)
    *   Thủy tinh, Kim loại, Giấy, Nhựa (Rác tái chế)
    *   Bóng đèn (Rác nguy hại)
    *   Rác hữu cơ
    *   Phụ tùng ô tô/xe máy
*   **💡 Tư vấn thông minh**: Không chỉ định danh, hệ thống còn cung cấp thông tin loại rác (Vd: Đây là rác tái chế hay rác nguy hại) bằng tiếng Việt.
*   **🖥️ Giao diện trực quan**: Tích hợp giao diện Web (Gradio) dễ sử dụng với 2 chế độ:
    1.  **Phân tích ảnh tĩnh**: Tải ảnh lên từ máy tính.
    2.  **Webcam trực tiếp**: Nhận diện rác theo thời gian thực (Real-time).
*   **🚀 Tối ưu hóa phần cứng**: Tự động sử dụng GPU (Metal/MPS trên MacOS hoặc CUDA trên Windows) để tăng tốc độ xử lý.

## 🛠 Yêu Cầu Cài Đặt

Đảm bảo bạn đã cài đặt **Python 3.8+**.

Các thư viện cần thiết:
*   `torch`, `torchvision`: Framework Deep Learning.
*   `gradio`: Xây dựng giao diện web.
*   `pillow`, `numpy`: Xử lý ảnh.
*   `tqdm`: Hiển thị thanh tiến trình khi train.

Cài đặt nhanh bằng lệnh:

```bash
pip install torch torchvision torchaudio gradio tqdm pillow numpy
```

## 📂 Cấu Trúc Thư Mục Dữ Liệu

Để hệ thống hoạt động hoặc huấn luyện lại, cấu trúc thư mục `dataset/` cần được tổ chức như sau (đúng tên thư mục tiếng Anh):

```text
WasteProcessor/
├── dataset/
│   ├── train/                  # Dữ liệu để học
│   │   ├── automobile wastes/
│   │   ├── battery waste/
│   │   ├── E-waste/
│   │   ├── glass waste/
│   │   ├── light bulbs/
│   │   ├── metal waste/
│   │   ├── organic waste/
│   │   ├── paper waste/
│   │   ├── plastic waste/
│   ├── test/                   # Dữ liệu để kiểm thử
│   │   ├── ... (Tương tự như train)
```

## 🚀 Hướng Dẫn Sử Dụng

### 1. Huấn luyện mô hình (Training)
Nếu bạn chưa có file model `waste_classifier.pth` hoặc muốn train lại với dữ liệu mới:

```bash
python train_model.py
```
*   Quá trình train sẽ tự động lưu model tốt nhất vào file `waste_classifier.pth`.
*   Hỗ trợ Resume (chạy lại) và thanh tiến trình trực quan.

### 2. Chạy ứng dụng (Inference)
Sau khi có model, khởi động ứng dụng web:

```bash
python main.py
```
*   Truy cập vào đường dẫn local (thường là `http://127.0.0.1:7860`) trên trình duyệt.
*   Chọn tab **"Phân Tích Ảnh"** hoặc **"Camera Trực Tiếp"** để trải nghiệm.

## 🧩 Thành Phần Mã Nguồn

*   **`ai_processor.py`**:
    *   `AIModel`: Class quản lý việc load model ResNet18 và dự đoán ảnh.
    *   `WasteExpert`: "Chuyên gia" ánh xạ kết quả từ AI sang thông tin rác tiếng Việt.
*   **`train_model.py`**: Script huấn luyện, sử dụng Transfer Learning từ ResNet18.
*   **`main.py`**: Mã nguồn giao diện chính, kết nối AI với giao diện Gradio.
*   **`requirements.txt`**: Danh sách thư viện phụ thuộc.

## 📝 Ghi Chú
*   Mô hình sử dụng **ResNet18** (Pre-trained) nên có độ chính xác khá tốt ngay cả với dữ liệu vừa phải.
*   Nếu không có GPU, hệ thống vẫn chạy được trên CPU (sẽ chậm hơn một chút khi train).

---
*Dự án Green AI - Vì một môi trường xanh sạch đẹp! 🌿*
