import os
import shutil
import random

# 設定原始資料夾和分割比例
original_data_dir = 'final_smoke_datasets'  # 原始數據夾
train_dir = 'dataset/train'                 # 訓練集資料夾
val_dir = 'dataset/val'                     # 驗證集資料夾
train_ratio = 0.8                           # 訓練集比例
label_file = 'final_smoke_datasets_label/new_label.txt'  # 原始標籤檔案
train_label_file = 'dataset/train_labels.txt'  # 訓練集標籤檔案
val_label_file = 'dataset/val_labels.txt'      # 驗證集標籤檔案

# 創建訓練和驗證資料夾
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 讀取標籤檔案並構建字典 {檔名: 標籤}
with open(label_file, 'r') as f:
    label_dict = {line.split()[0]: line.strip() for line in f}

# 獲取所有數據文件名並打亂順序
all_files = list(label_dict.keys())
random.seed(42)  # 固定隨機種子以保持一致
random.shuffle(all_files)

# 分割文件列表
train_size = int(train_ratio * len(all_files))
train_files = sorted(all_files[:train_size], key=lambda x: int(x.split('.')[0]))
val_files = sorted(all_files[train_size:], key=lambda x: int(x.split('.')[0]))
print(train_files)
# 初始化標籤檔案
with open(train_label_file, 'w') as train_f, open(val_label_file, 'w') as val_f:
    # 將訓練集文件複製並寫入標籤
    for file_name in train_files:
        src_path = os.path.join(original_data_dir, file_name)
        dst_path = os.path.join(train_dir, file_name)
        shutil.copy(src_path, dst_path)
        train_f.write(label_dict[file_name] + '\n')

    # 將驗證集文件複製並寫入標籤
    for file_name in val_files:
        src_path = os.path.join(original_data_dir, file_name)
        dst_path = os.path.join(val_dir, file_name)
        shutil.copy(src_path, dst_path)
        val_f.write(label_dict[file_name] + '\n')

print("數據及標籤已成功分割並複製到訓練和驗證資料夾。")
