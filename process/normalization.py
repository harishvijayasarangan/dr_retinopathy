import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.data['image'] = self.data['image'].astype(str) + ".jpeg"

        self.data = self.data[self.data['image'].apply(lambda x: os.path.exists(os.path.join(self.root_dir, x)))]

        print(f"✅ Total valid images: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, filename)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def calculate_mean_std(dataloader, dataset_size):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0
    for images in tqdm(dataloader, desc="Processing Images", total=dataset_size):
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(dim=[0, 2]) * batch_samples
        std += images.std(dim=[0, 2]) * batch_samples
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    print(f"✅ Total images actually processed: {num_samples}")
    return mean, std

if __name__ == "__main__":
    train_csv = r"C:\Users\STIC-11\Desktop\sk1\train.csv"
    image_dir = r"C:\Users\STIC-11\Desktop\sk1\data\square\train"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = RetinopathyDataset(csv_file=train_csv, root_dir=image_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    mean, std = calculate_mean_std(dataloader, dataset_size=len(dataset))

    print(f"✅ Calculated Mean: {mean.tolist()}")
    print(f"✅ Calculated Std: {std.tolist()}")
