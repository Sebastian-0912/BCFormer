import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


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
    
    