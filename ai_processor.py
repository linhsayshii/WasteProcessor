import torch
from torchvision import models, transforms
from PIL import Image
import os

# --- CLASS 1: AI MODEL (Bộ não nhận diện hình ảnh) ---
class AIModel:
    def __init__(self, model_path='waste_classifier.pth'):
        # 1. Tự động nhận diện thiết bị (Quan trọng cho Mac M2)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 AI Processor: Đang chạy trên Apple M2 GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("⚠️ AI Processor: Đang chạy trên CPU (Chậm)")
            
        self.loaded = False
        
        # 2. Load Model
        if os.path.exists(model_path):
            try:
                # Mẹo: Load vào CPU trước để tránh lỗi bộ nhớ, sau đó mới đẩy sang GPU
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                self.class_names = checkpoint['class_names']
                
                # Khởi tạo kiến trúc ResNet18
                self.model = models.resnet18(weights=None)
                # Sửa đầu ra cho khớp số lượng class (9 loại)
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_ftrs, len(self.class_names))
                
                # Nạp trọng số (Weights)
                self.model.load_state_dict(checkpoint['model_state'])
                
                # Đẩy model sang thiết bị xử lý (GPU/MPS)
                self.model.to(self.device)
                self.model.eval() # Chế độ dự đoán
                
                self.loaded = True
                print(f"✅ Đã load model thành công! Nhận diện được: {len(self.class_names)} loại.")
            except Exception as e:
                print(f"❌ Lỗi khi load model: {e}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file '{model_path}'. Hãy kiểm tra lại tên file!")

        # 3. Bộ xử lý ảnh (Giống hệt lúc train)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        Input: Ảnh PIL
        Output: Tên nhãn (Tiếng Anh), Độ tin cậy (0.0 - 1.0)
        """
        if not self.loaded:
            return "Error", 0.0

        # Chuyển ảnh sang Tensor và đẩy vào GPU M2
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Tính xác suất (Softmax)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, idx = probs.topk(1, dim=1)
        
        label_en = self.class_names[idx.item()]
        confidence = conf.item()
        
        return label_en, confidence

# --- CLASS 2: WASTE EXPERT (Chuyên gia tư vấn rác - Rút gọn) ---
class WasteExpert:
    def __init__(self):
        # Cấu hình kiến thức (Chỉ giữ lại Name và Type theo yêu cầu của bạn)
        self.knowledge_base = {
            "automobile wastes": {
                "vn_name": "Phụ tùng Ô tô / Xe máy",
                "type": "Rác Công Nghiệp"
            },
            "battery waste": {
                "vn_name": "Pin / Ắc quy cũ",
                "type": "Rác Nguy Hại (Hazardous)"
            },
            "E-waste": {
                "vn_name": "Rác thải Điện tử (Linh kiện)",
                "type": "Rác Điện Tử"
            },
            "glass waste": {
                "vn_name": "Thủy tinh (Chai/Lọ/Kính)",
                "type": "Rác Tái Chế"
            },
            "light bulbs": {
                "vn_name": "Bóng đèn (Huỳnh quang/Sợi đốt)",
                "type": "Rác Nguy Hại"
            },
            "metal waste": {
                "vn_name": "Kim loại (Vỏ lon/Sắt vụn)",
                "type": "Rác Tái Chế"
            },
            "organic waste": {
                "vn_name": "Rác Hữu cơ (Thức ăn/Rau củ)",
                "type": "Rác Hữu Cơ"
            },
            "paper waste": {
                "vn_name": "Giấy / Bìa Carton",
                "type": "Rác Tái Chế"
            },
            "plastic waste": {
                "vn_name": "Nhựa (Chai/Lọ/Túi)",
                "type": "Rác Tái Chế"
            },
            "unknown": {
                "vn_name": "Chưa xác định",
                "type": "N/A"
            }
        }

    def consult(self, label_en):
        """
        Nhận vào tên tiếng Anh (từ AI), trả về thông tin rút gọn.
        """
        # Lấy thông tin, nếu không có key thì trả về 'unknown'
        info = self.knowledge_base.get(label_en, self.knowledge_base["unknown"])
        
        return {
            "name": info['vn_name'],
            "type": info['type']
        }