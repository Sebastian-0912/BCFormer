import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MultiLabelImageDataset
import os
import matplotlib.pyplot as plt
# 印出當前目錄
current_directory = os.getcwd()
print("Current Directory:", current_directory)


# images_labels = r'C:/Users/User/Documents/Kuan-wu Chu/final_smoke_datasets/labels.txt'
# # 圖片資料夾路徑
# images_folder = r'C:/Users/User/Documents/Kuan-wu Chu/final_smoke_datasets'

images_folder = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val' 
images_labels = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val/val_labels.txt' 
# 獲取資料夾中的所有圖片檔案
images_path = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# 定義轉換（這裡用簡單的轉換示例，可根據需要修改）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 建立資料集和 DataLoader
dataset = MultiLabelImageDataset(images_dir=images_folder, labels_file=images_labels, transform=transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 建立標籤名稱對應的映射表
label_map = {0: "fire", 1: "smoke", 2: "cloud", 3: "none"}

# 測試資料集並顯示圖片和標籤
for idx, (image, label) in enumerate(dataloader):
    # 找到標籤中值為 1 的位置，並轉換為對應的標籤名稱
    label_names = [label_map[i] for i, val in enumerate(label[0].tolist()) if val == 1]
    
    print(f"Image {idx+1}:")
    print("Label:", label_names if label_names else ["none"])
    # 如果想要展示圖片，可以取消下面兩行的註解
    import matplotlib.pyplot as plt
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.show()

    # Break after printing the first few examples for brevity
    if idx >= 4:
        break
