import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy
from tqdm import tqdm

def train_model():
    # --- CẤU HÌNH ---
    DATA_DIR = './dataset'
    SAVE_PATH = 'waste_classifier.pth'
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    NUM_WORKERS = 3
    LEARNING_RATE = 0.0001

    # 1. Setup GPU engine
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 2. Xử lý ảnh (Augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                      for x in ['train', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                                 shuffle=True, num_workers=NUM_WORKERS) 
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"📂 Đã load xong: {dataset_sizes['train']} ảnh train | {dataset_sizes['test']} ảnh test")
    print(f"🏷️ Labels: {class_names}")

    # 3. Load Model ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # --- QUAN TRỌNG: MỞ KHÓA NÃO (UNFREEZE) ---
    for param in model.parameters():
        param.requires_grad = True 

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # Dùng Adam với Learning Rate thấp
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("\n🔥 BẮT ĐẦU TRAIN...")
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')

        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            pbar = tqdm(dataloaders[phase], unit="batch", leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_description(f"{phase} Loss: {loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'   👉 {phase} Accuracy: {epoch_acc:.4f}')

            # Lưu model tốt nhất (kiểm tra trên tập test)
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("      🌟 Kỷ lục mới! Đã lưu tạm.")

    print(f'\n🏆 Kết quả cuối cùng: {best_acc:.4f}')
    
    # Save Final Model
    model.load_state_dict(best_model_wts)
    torch.save({
        'model_state': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)
    print(f"💾 Đã lưu model tại: {SAVE_PATH}")

if __name__ == "__main__":
    train_model()