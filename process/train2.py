import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
os.makedirs("checkpoints", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")
if device.type == "cuda":
    print(f" GPU: {torch.cuda.get_device_name(0)}")
train_csv = r"C:\Users\STIC-11\Desktop\class2\train.csv"
val_csv = r"C:\Users\STIC-11\Desktop\class2\val.csv"
image_dir = r"C:\Users\STIC-11\Desktop\sk1\mild_no_dr"

for file in [train_csv, val_csv]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"CSV file not found: {file}")
if not os.path.exists(image_dir):
    raise FileNotFoundError("Image directory not found.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5469, 0.3678, 0.2555], std=[0.2097, 0.1599, 0.1447])  
])

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

        
        label = torch.tensor(label, dtype=torch.float32)  

        if self.transform:
            image = self.transform(image)

        return image, label


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean', dataset=None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')


        if dataset:
            class_counts = dataset.data.iloc[:, 1].value_counts()
            total = class_counts.sum()
            self.alpha = torch.tensor([
                class_counts.get(0, 1) / total, 
                class_counts.get(1, 1) / total  
            ]).to(device)
        else:
            self.alpha = torch.tensor([0.25, 0.75]).to(device) 

    def forward(self, inputs, targets):
        targets = targets.view(-1)  
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) 
        alpha_factor = self.alpha[targets.long()] 
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


train_dataset = RetinopathyDataset(csv_file=train_csv, root_dir=image_dir, transform=transform)
val_dataset = RetinopathyDataset(csv_file=val_csv, root_dir=image_dir, transform=transform)


labels = train_dataset.data.iloc[:, 1].values
class_counts = pd.value_counts(labels)
class_weights = 1.0 / class_counts 
sample_weights = pd.Series(labels).map(class_weights).values
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)


weights = DenseNet121_Weights.IMAGENET1K_V1
model = densenet121(weights=weights)


model.classifier = nn.Linear(1024, 1) 
model = model.to(device)


criterion = BinaryFocalLoss(dataset=train_dataset)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scaler = GradScaler()


checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints", filename="best_model3", save_top_k=1, monitor="val_loss", mode="min"
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
        
        with autocast(dtype=torch.float16): 
            outputs = model(inputs).squeeze(1) 
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        predicted = torch.sigmoid(outputs) > 0.5  
        total += labels.size(0)
        correct += predicted.eq(labels.bool()).sum().item()

        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), "checkpoints/best_model3.ckpt")
    print(" Model saved successfully!")
