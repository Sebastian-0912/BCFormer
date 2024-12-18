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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    # 設置模型路徑和結果保存路徑
    model_dir = "./weights"
    save_dir = "./inference_results"
    os.makedirs(save_dir, exist_ok=True)

    # 獲取所有模型文件（支持 model-*.pth 和 best_model-*.pth）
    model_paths = glob.glob(os.path.join(model_dir, "model-*.pth")) + \
                glob.glob(os.path.join(model_dir, "best_model-*.pth"))

    # 提取 epoch 數字並排序
    model_paths = sorted(model_paths, key=lambda x: int(x.split('-')[-1].split('.')[0]))

    # 標籤對應字典
    label_map = {"fire": 0, "smoke": 1, "cloud": 2, "none": 3}
    labels = list(label_map.keys())
    val_images_dir = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val' 
    val_labels_file = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val/val_labels.txt' 
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    val_dataset = MultiLabelImageDataset(images_dir=val_images_dir, labels_file=val_labels_file, transform=data_transform["val"])
    val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=nw)

    model = create_model(num_classes=4, has_logits=False).to(device)
    # 模型推論和混淆矩陣繪製
    for model_path in model_paths:
        # 提取 epoch 值
        epoch = int(model_path.split('-')[-1].split('.')[0])
        print(f"Loading model from epoch {epoch}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

        model.eval()

        # 初始化變量以累積整個 DataLoader 的推論結果
        all_true_labels = []
        all_predicted_labels = []
        accu_num = 0  # 累積正確預測數量
        sample_num = 0  # 累積樣本數

        # DataLoader 推論
        # DataLoader 推論
        with torch.no_grad():
            for step, (images, batch_labels) in enumerate(tqdm(val_loader, desc=f"Inference Epoch {epoch}")):
                # 更新樣本數
                sample_num += images.shape[0]
                
                # 模型預測
                pred = model(images.to(device))
                
                # 將預測結果轉為二進制格式並移至 CPU
                pred_classes = (torch.sigmoid(pred) > 0.5).cpu()
                true_labels = batch_labels.cpu()

                # 計算準確度 (逐元素比較並累計符合數量)
                accu_num += torch.eq(pred_classes, true_labels).sum().item()

                # 累積所有樣本的真實標籤和預測標籤
                all_true_labels.extend(true_labels.numpy())
                all_predicted_labels.extend(pred_classes.numpy())

        # 計算平均準確率
        avg_accuracy = accu_num / (sample_num * len(label_map))

        # 計算多標籤混淆矩陣
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predicted_labels)

        # 初始化4x4的混淆矩陣
        confusion_matrix_aggregate = np.zeros((4, 4), dtype=int)

        # 將所有的真實標籤和預測標籤疊加到4x4矩陣中
        for true, pred in zip(y_true, y_pred):
            for i in range(4):
                if true[i] == 1:  # 行代表真實標籤
                    for j in range(4):
                        if pred[j] == 1:  # 列代表預測標籤
                            confusion_matrix_aggregate[i, j] += 1

        # 可視化整合後的混淆矩陣
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(confusion_matrix_aggregate, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels)

        # 標題和軸標籤，顯示準確率
        ax.set_title(f"Aggregated Confusion Matrix (Epoch {epoch})\nAccuracy: {avg_accuracy:.2%}")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")

        # 儲存結果
        result_path = os.path.join(save_dir, f"aggregated_confusion_matrix_epoch_{epoch}.png")
        plt.savefig(result_path)
        plt.close()
        print(f"Saved aggregated confusion matrix for epoch {epoch} to {result_path}")

        # mcm = multilabel_confusion_matrix(y_true, y_pred)

        # # 可視化每個類別的混淆矩陣
        # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        # axes = axes.flatten()

        # for i, (ax, label) in enumerate(zip(axes, labels)):
        #     cm = mcm[i]
        #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not " + label, label],
        #                 yticklabels=["Not " + label, label], ax=ax)
        #     ax.set_title(f"Confusion Matrix for {label} (Epoch {epoch})")
        #     ax.set_xlabel("Predicted")
        #     ax.set_ylabel("True")

        # plt.tight_layout()
        # result_path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch}.png")
        # plt.savefig(result_path)
        # plt.close(fig)  # 關閉圖表以釋放內存
        # print(f"Saved confusion matrix for epoch {epoch} to {result_path}")

if __name__ == '__main__':
    main()  