import torch
from torchvision import models, transforms
from PIL import Image
import os

# --- CLASS 1: AI MODEL (Đã nâng cấp lên ResNet50) ---
class AIModel:
    def __init__(self, model_path='waste_classifier_pro.pth'):
        # 1. Tự động nhận diện thiết bị
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.loaded = False
        
        # 2. Load Model
        if os.path.exists(model_path):
            try:
                # Load thông tin saved
                # map_location='cpu' để an toàn, sau đó mới đẩy vào GPU
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                self.class_names = checkpoint['class_names']
                
                # --- SỬA Ở ĐÂY: Đổi resnet18 thành resnet50 ---
                self.model = models.resnet50(weights=None)
                
                # Tự động lấy số lượng đầu vào của lớp cuối cùng (ResNet50 là 2048)
                num_ftrs = self.model.fc.in_features 
                self.model.fc = torch.nn.Linear(num_ftrs, len(self.class_names))
                
                # Nạp trọng số (Weights)
                self.model.load_state_dict(checkpoint['model_state'])
                
                # Đẩy model sang thiết bị xử lý
                self.model.to(self.device)
                self.model.eval() # Chế độ dự đoán
                
                self.loaded = True
                print(f"✅ Đã load thành công! Nhận diện được: {len(self.class_names)} loại.")
            except Exception as e:
                print(f"❌ Lỗi khi load model: {e}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file '{model_path}'.")

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
            return "Error: Model not loaded", 0.0

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

# --- CLASS 2: WASTE EXPERT ---
class WasteExpert:
    def __init__(self):
        self.knowledge_base = {
            "automobile wastes": {"vn_name": "Phụ tùng Ô tô / Xe máy", "type": "Rác Công Nghiệp"},
            "battery waste": {"vn_name": "Pin / Ắc quy cũ", "type": "Rác Nguy Hại (Hazardous)"},
            "E-waste": {"vn_name": "Rác thải Điện tử", "type": "Rác Điện Tử"},
            "glass waste": {"vn_name": "Thủy tinh (Chai/Lọ)", "type": "Rác Tái Chế"},
            "light bulbs": {"vn_name": "Bóng đèn vỡ/hỏng", "type": "Rác Nguy Hại"},
            "metal waste": {"vn_name": "Kim loại (Vỏ lon/Sắt)", "type": "Rác Tái Chế"},
            "organic waste": {"vn_name": "Rác Hữu cơ (Thức ăn)", "type": "Rác Hữu Cơ"},
            "paper waste": {"vn_name": "Giấy / Bìa Carton", "type": "Rác Tái Chế"},
            "plastic waste": {"vn_name": "Nhựa (Chai/Lọ/Túi)", "type": "Rác Tái Chế"},
            "unknown": {"vn_name": "Chưa xác định", "type": "N/A"}
        }

    def consult(self, label_en):
        info = self.knowledge_base.get(label_en, self.knowledge_base["unknown"])
        return {"name": info['vn_name'], "type": info['type']}