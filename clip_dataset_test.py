import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP and BLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()


class MultiLabelImageDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None, num_classes=4):
        """
        Args:
            images_dir (str): 圖片的資料夾路徑
            labels_file (str): 標籤的 txt 文件路徑
            transform (callable, optional): 圖片的轉換 (預處理)
            num_classes (int): 標籤的類別數 (例如 fire, smoke 等)
        """
        self.images_path = []
        self.images_labels = []
        self.transform = transform
        self.num_classes = num_classes
        self.label_map = {"fire": 0, "smoke": 1, "cloud": 2, "none": 3}  # 根據您的標籤映射
        
        # 讀取標籤文件並解析每一行
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split()  # 解析出圖片名稱和多標籤
                img_name = parts[0]
                labels = parts[1:]
                
                # 將圖片完整路徑加入到 images_path 列表
                img_path = os.path.join(images_dir, img_name)
                self.images_path.append(img_path)
                self.images_labels.append(labels)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        labels = self.images_labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        # 將標籤轉換為二進制向量
        label_vector = torch.zeros(self.num_classes)
        for label in labels:
            if label in self.label_map:
                label_vector[self.label_map[label]] = 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_vector
    

# Model definition
class Smoke_model(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(Smoke_model, self).__init__()
        self.l1 = nn.Linear(512, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text_embedding, image_embedding):
        combined_embedding = text_embedding * image_embedding
        output = self.l1(combined_embedding)
        output = self.relu(output)
        output = self.l2(output)
        return output


# Data Loading
images_dir = 'C:/Users/User/Documents/Kuan-wu Chu/final_smoke_datasets'  # Specify your directory
labels_file = 'C:/Users/User/Documents/Kuan-wu Chu/final_smoke_datasets_label/labels.txt'  # Specify your label file
dataset = MultiLabelImageDataset(images_dir=images_dir, labels_file=labels_file,transform=clip_preprocess)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # smoke_model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device),  labels.to(device)