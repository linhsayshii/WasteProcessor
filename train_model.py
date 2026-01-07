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
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # 1. Setup GPU engine
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 2. Xử lý ảnh (Augmentation nhẹ để tránh Overfitting)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
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
    
    # num_workers=2 để CPU load ảnh song song với việc GPU train
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                                 shuffle=True, num_workers=NUM_WORKERS) 
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"📂 Đã load xong: {dataset_sizes['train']} ảnh train | {dataset_sizes['test']} ảnh test")
    print(f"🏷️ Labels: {class_names}") # Sẽ in ra ['non_recyclable', 'recyclable']

    # 3. Load Model ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 4. Training Loop (Có lưu Best Model)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n🏁 BẮT ĐẦU RACE!")
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Sử dụng tqdm để hiện thanh loading
            pbar = tqdm(dataloaders[phase], unit="batch")
            
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
                
                # Update thanh loading
                pbar.set_description(f"{phase} Loss: {loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'   👉 {phase} Accuracy: {epoch_acc:.4f}')

            # Deep Copy model nếu nó tốt hơn các vòng trước
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("   🌟 (New Best Model Found!)")

    time_elapsed = time.time() - since
    print(f'\n✨ Train hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best Val Acc: {best_acc:.4f}')

    # Load lại weight tốt nhất và lưu
    model.load_state_dict(best_model_wts)
    torch.save({
        'model_state': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)
    print(f"💾 Đã lưu model tại: {SAVE_PATH}")

if __name__ == '__main__':
    train_model()