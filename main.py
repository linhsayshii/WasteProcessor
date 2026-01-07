import gradio as gr
from ai_processor import AIModel, WasteExpert
import time
from PIL import Image
import numpy as np
import os

# --- 1. KHỞI TẠO HỆ THỐNG (OOP) ---
print("⏳ Đang khởi động hệ thống Green AI...")

# Kiểm tra xem file model có tồn tại không trước khi load
model_filename = 'waste_classifier.pth' 

if not os.path.exists(model_filename):
    print(f"⚠️ CẢNH BÁO: Không tìm thấy file '{model_filename}'!")
    print("👉 Bạn cần chạy file 'train_model.py' trước để tạo ra file này.")
    # Tạo object rỗng để app không bị crash ngay lập tức, nhưng sẽ báo lỗi khi dùng
    ai_bot = None
else:
    try:
        ai_bot = AIModel(model_path=model_filename)
        print("✅ Hệ thống AI đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo AI: {e}")
        ai_bot = None

expert = WasteExpert()

# --- 2. HÀM XỬ LÝ LOGIC ---

def analyze_static(image):
    """Xử lý ảnh tĩnh (Tab 1)"""
    if ai_bot is None:
        return "Lỗi Model", "...", "Chưa load được AI"
        
    if image is None:
        return None, None, None

    # B1: AI dự đoán
    label_en, conf = ai_bot.predict(image)
    
    # B2: Lấy thông tin (Chỉ có name và type)
    info = expert.consult(label_en)

    # B3: Trả về kết quả (3 giá trị tương ứng với 3 ô output bên dưới)
    vn_name = info['name'].upper()
    accuracy = f"Độ chính xác: {conf:.1%}"
    waste_type = info['type']
    
    return vn_name, accuracy, waste_type

def analyze_stream(image):
    """Xử lý Webcam Live (Tab 2)"""
    if ai_bot is None:
        return "Lỗi Model", "..."

    if image is None:
        return "Đang chờ camera...", "..."

    # Chuyển đổi ảnh từ Webcam (Numpy -> PIL)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # B1: AI dự đoán
    label_en, conf = ai_bot.predict(image)
    
    # Chống nhấp nháy khi độ tin cậy thấp
    if conf < 0.5:
        return "🔍 Đang tìm...", "..."

    # B2: Lấy thông tin
    info = expert.consult(label_en)

    # B3: Trả về kết quả (2 giá trị)
    # Gộp tên và độ tin cậy vào 1 dòng cho gọn
    display_name = f"{info['name']} ({conf:.0%})"
    waste_type = info['type']
    
    return display_name, waste_type

# --- 3. GIAO DIỆN (UI) ---

custom_css = """
.header {text-align: center; color: #2e7d32; margin-bottom: 10px;}
.result-box {
    background-color: #e8f5e9; 
    border-radius: 8px; 
    padding: 10px; 
    font-weight: bold; 
    text-align: center;
    font-size: 20px;
    color: #1b5e20;
}
.type-box {
    background-color: #fff3e0;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    color: #e65100;
}
"""

with gr.Blocks(css=custom_css, title="WASTE CLASSIFIER") as demo:
    
    gr.Markdown("# 🌿 PHÂN LOẠI RÁC THÔNG MINH", elem_classes="header")
    
    with gr.Tabs():
        # === TAB 1: ẢNH TĨNH ===
        with gr.TabItem("📸 Phân Tích Ảnh"):
            with gr.Row():
                with gr.Column():
                    input_static = gr.Image(label="Ảnh đầu vào", sources=["upload", "webcam"], type="pil", height=300)
                    btn_analyze = gr.Button("🔍 KIỂM TRA", variant="primary")
                
                with gr.Column():
                    # Chỉ còn 3 ô output: Tên, Độ chính xác, Loại rác
                    out_name = gr.Textbox(label="Vật thể phát hiện", elem_classes="result-box")
                    out_acc = gr.Label(label="Độ chính xác AI")
                    out_type = gr.Textbox(label="Phân loại rác", elem_classes="type-box")
            
            # Kết nối hàm xử lý
            btn_analyze.click(analyze_static, [input_static], [out_name, out_acc, out_type])

        # === TAB 2: WEBCAM LIVE ===
        with gr.TabItem("🎥 Quét Live"):
            with gr.Row():
                with gr.Column():
                    input_stream = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy", height=400)
                
                with gr.Column():
                    # Output rút gọn cho Live
                    live_name = gr.Textbox(label="Vật thể", elem_classes="result-box")
                    live_type = gr.Textbox(label="Phân loại", elem_classes="type-box")

            # Kết nối hàm xử lý (Thay đổi liên tục)
            input_stream.change(analyze_stream, [input_stream], [live_name, live_type], show_progress=False)

# --- 4. CHẠY APP ---
if __name__ == "__main__":
    demo.launch(share=False)