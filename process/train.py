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
from torch.cuda.amp import autocast, GradScaler  
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == "__main__":


    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")


    train_csv = r"C:\Users\STIC-11\Desktop\class2\train.csv"
    val_csv = r"C:\Users\STIC-11\Desktop\class2\val.csv"
    image_dir = r"C:\Users\STIC-11\Desktop\sk1\data\square\train"


    for file in [train_csv, val_csv]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"CSV file not found: {file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError("Image directory not found.")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5440180897712708, 0.36822232604026794, 0.25577110052108765], std=[0.20675460994243622, 0.15600024163722992, 0.13932974636554718])
    ])


    class RetinopathyDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.data = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
            self.valid_indices = []
            print("Validating dataset files...")
            for idx in tqdm(range(len(self.data))):
                filename = str(self.data.iloc[idx, 0])
                if not filename.endswith(".jpeg"):
                    filename += ".jpeg"
                img_path = os.path.join(self.root_dir, filename)
                
                if os.path.exists(img_path):
                    self.valid_indices.append(idx)
                
            print(f"Found {len(self.valid_indices)} valid images out of {len(self.data)} entries")

            self.data = self.data.iloc[self.valid_indices].reset_index(drop=True)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            filename = str(self.data.iloc[idx, 0])
            if not filename.endswith(".jpeg"):
                filename += ".jpeg"
            
            img_path = os.path.join(self.root_dir, filename)
            try:
                image = Image.open(img_path).convert("RGB")
                label = int(self.data.iloc[idx, 1])

                if self.transform:
                    image = self.transform(image)

                return image, label
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                raise


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
    try:
        print("Initializing training dataset...")
        train_dataset = RetinopathyDataset(csv_file=train_csv, root_dir=image_dir, transform=transform)
        print("Initializing validation dataset...")
        val_dataset = RetinopathyDataset(csv_file=val_csv, root_dir=image_dir, transform=transform)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Dataset is empty after filtering invalid images")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=16, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
    except Exception as e:
        print(f"Error initializing datasets: {str(e)}")
        raise

    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)

  
    model.classifier = nn.Linear(1024, 5)
    model = model.to(device)

    
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scaler = GradScaler()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="best_model4", save_top_k=1, monitor="val_loss", mode="min"
    )
    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback])


    num_epochs = 10
    best_val_acc = 0.0
    
    for epoch in range(n):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with autocast(dtype=torch.float16): 
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


        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train -> Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"   Val   -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")


        model_path = f"checkpoints/model4_epoch_{epoch+1}.ckpt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully for epoch {epoch+1}!")

        torch.save(model.state_dict(), "checkpoints/best_model4.ckpt")

    print("Training completed!")
