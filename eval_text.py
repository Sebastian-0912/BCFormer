import os
import torch
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm  # 進度條
from dataset import MultiLabelImageDataset
from vit_model import vit_base_patch32_224_in21k as create_model
# from utils import train_one_epoch, evaluate, read_split_data

from torch.utils.data import DataLoader, random_split

label_map = {"fire": 0, "smoke": 1, "cloud": 2, "none": 3}
labels = list(label_map.keys())

# 計算多標籤混淆矩陣
y_true = np.array([[1, 1, 0, 0]])  # 真實標籤
y_pred = np.array([[0, 0, 1, 1]])  # 預測標籤

# 初始化4x4的混淆矩陣
confusion_matrix_aggregate = np.zeros((4, 4), dtype=int)

# # 將所有的真實標籤和預測標籤疊加到4x4矩陣中
# for true, pred in zip(y_true, y_pred):
#     for i in range(4):
#         if true[i] == 1:  # 行代表真實標籤
#             for j in range(4):
#                 if pred[j] == 1:  # 列代表預測標籤
#                     confusion_matrix_aggregate[i, j] += 1
# # 可視化整合後的混淆矩陣
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(confusion_matrix_aggregate, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=labels, yticklabels=labels)

# # 標題和軸標籤，顯示準確率

# ax.set_xlabel("Predicted Labels")
# ax.set_ylabel("True Labels")
# plt.show()

mcm = multilabel_confusion_matrix(y_true, y_pred)

# 可視化每個類別的混淆矩陣
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, (ax, label) in enumerate(zip(axes, labels)):
    cm = mcm[i]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not " + label, label],
                yticklabels=["Not " + label, label], ax=ax)
    # ax.set_title(f"Confusion Matrix for {label} (Epoch {epoch})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()
