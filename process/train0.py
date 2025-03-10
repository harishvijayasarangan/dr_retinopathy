"""import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler  

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure the image filename has the correct extension
        filename = str(self.data.iloc[idx, 0])
        if not filename.endswith(".jpeg"):  
            filename += ".jpeg" 

        img_path = os.path.join(self.root_dir, filename)

        # Check if file exists (for debugging)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__": 
    print(f"🔥 Using device: {device}")
    if device.type == "cuda":
        print(f"⚡ GPU: {torch.cuda.get_device_name(0)}")

    # Paths (Ensure these files exist)
    train_csv = r"D:\retina train\diabetic-retinopathy-detection\trainLabels.csv\trainLabels.csv"
    val_csv = r"D:\retina train\diabetic-retinopathy-detection\trainLabels.csv\val.csv"
    image_dir = r"D:\retina train\diabetic-retinopathy-detection\train\train"

    # Verify CSV files exist before proceeding
    for file in [train_csv, val_csv]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"CSV file not found: {file}")

    # Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Data
    train_dataset = RetinopathyDataset(csv_file=train_csv, root_dir=image_dir, transform=transform)
    val_dataset = RetinopathyDataset(csv_file=val_csv, root_dir=image_dir, transform=transform)

    # Optimized DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # Load DenseNet Model
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(1024, 5)
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable Mixed Precision Training
    scaler = GradScaler()

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with autocast("cuda"): 
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # Print epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f" Epoch {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save Model
    torch.save(model.state_dict(), "densenet_retinopathy.pth")
    print("🎉 Model training complete. Saved as densenet_retinopathy.pth")
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler  
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = str(self.data.iloc[idx, 0])
        if not filename.endswith(".jpeg"):  
            filename += ".jpeg" 

        img_path = os.path.join(self.root_dir, filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__": 
    print(f"🔥 Using device: {device}")
    if device.type == "cuda":
        print(f"⚡ GPU: {torch.cuda.get_device_name(0)}")

    train_csv = r"D:\retina train\diabetic-retinopathy-detection\trainLabels.csv\trainLabels.csv"
    val_csv = r"D:\retina train\diabetic-retinopathy-detection\trainLabels.csv\val.csv"
    image_dir = r"D:\retina train\diabetic-retinopathy-detection\train\train"

    for file in [train_csv, val_csv]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"CSV file not found: {file}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.3203, 0.2244, 0.1610], std=[0.2972, 0.2135, 0.1682])
    ])

    train_dataset = RetinopathyDataset(csv_file=train_csv, root_dir=image_dir, transform=transform)
    val_dataset = RetinopathyDataset(csv_file=val_csv, root_dir=image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(1024, 5)
    model = model.to(device)

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    scaler = GradScaler()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="best_model", save_top_k=1, monitor="val_loss", mode="min"
    )
    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback])

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


import os
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/best_model.ckpt")
