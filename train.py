import os
import zipfile
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DISEASES = ["COVID", "Viral Pneumonia", "Lung_Opacity", "Normal"]
EPOCHS = 5

def prepare_data():
    # Assumes kaggle.json is configured and datasets are downloaded
    # Download dataset manually or via kaggle API
    print("Preparing data...")
    if not os.path.exists("images"):
        os.makedirs("images")
        base = "COVID-19_Radiography_Dataset"
        if os.path.exists(base):
            for cls in DISEASES:
                cls_img_dir = os.path.join(base, cls, "images")
                if not os.path.exists(cls_img_dir):
                    cls_img_dir = os.path.join(base, cls)
                if os.path.exists(cls_img_dir):
                    for fname in os.listdir(cls_img_dir):
                        if fname.lower().endswith(".png"):
                            src = os.path.join(cls_img_dir, fname)
                            dst = os.path.join("images", f"{cls.replace(' ', '_')}_{fname}")
                            shutil.copy2(src, dst)
    return "images"

class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform, diseases_list):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.diseases = diseases_list

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            image = self.transform(image=image)["image"]

        labels = torch.tensor(row[self.diseases].values.astype("float32"))
        return image, labels

def main():
    prepare_data()
    # Create DF
    records = []
    if os.path.exists("images"):
        for fname in os.listdir("images"):
            if fname.lower().endswith(".png"):
                label = "Normal"
                for cls in DISEASES:
                    if fname.startswith(cls.replace(" ", "_")):
                        label = cls
                        break
                records.append({"Image Index": fname, "Finding Labels": label})
                
    df = pd.DataFrame(records)
    for disease in DISEASES:
        df[disease] = df["Finding Labels"].apply(lambda x: 1 if disease == x else 0)
        
    if len(df) == 0:
        print("No images found. Please download the dataset!")
        return

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    train_dataset = ChestXrayDataset(train_df, "images", train_transform, DISEASES)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 4))
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1} Loss: {train_loss:.4f}")
        
    torch.save(model.state_dict(), 'best_model.pth')
    print("Training complete! Saved best_model.pth")

if __name__ == "__main__":
    main()
