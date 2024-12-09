from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import torch


class MultiLabelImageDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None, num_classes=3):
        """
        Args:
            images_dir (str): Path to the images directory.
            labels_file (str): Path to the labels text file.
            transform (callable, optional): Transformations for preprocessing images.
            num_classes (int): Number of label classes (e.g., fire, smoke, cloud).
        """
        self.images_path = []
        self.images_labels = []
        self.transform = transform
        self.num_classes = num_classes
        self.label_map = {"fire": 0, "smoke": 1, "cloud": 2}  # Customize as per your labels
        
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                img_name = parts[0]
                labels = parts[1:]
                
                img_path = os.path.join(images_dir, img_name)
                self.images_path.append(img_path)
                
                # Handle "none" case by checking if there are valid labels
                if labels == ["none"]:
                    label_vector = torch.zeros(self.num_classes)
                else:
                    label_vector = torch.zeros(self.num_classes)
                    for label in labels:
                        if label in self.label_map:
                            label_vector[self.label_map[label]] = 1
                
                self.images_labels.append(label_vector)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label_vector = self.images_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label_vector