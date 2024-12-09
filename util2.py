import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

import os
import random
import json
import pandas as pd


def read_split_data(annotations_file: str, img_dir: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(annotations_file), "annotations file: {} does not exist.".format(annotations_file)

    # 读取多标标签文件
    img_labels = []
    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            labels = parts[1:]  # 标注的多标签列表
            img_labels.append((img_name, labels))

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历所有图片
    for img_name, labels in img_labels:
        img_path = os.path.join(img_dir, img_name)
        if os.path.splitext(img_name)[-1] not in supported:
            continue  # 跳过不支持的文件

        # 按比例随机采样验证样本
        if random.random() < val_rate:
            val_images_path.append(img_path)
            val_images_label.append(labels)
        else:
            train_images_path.append(img_path)
            train_images_label.append(labels)

    print("{} images were found in the dataset.".format(len(img_labels)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label



def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
        
def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion, df, df_file_path):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累計損失
    correct_predictions = torch.zeros(1).to(device)  # 累計正確的樣本數
    total_labels = 0  # 累計的標籤數
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))  # 預測結果 (logits)

        # 使用 BCEWithLogitsLoss 計算損失
        loss = criterion(pred, labels.to(device).float())  # 標籤轉為 float
        loss.backward()
        accu_loss += loss.detach()

        # 使用 sigmoid 將 logits 轉換為二進制預測類別 (0 或 1)
        pred_classes = torch.sigmoid(pred) >= 0.5

        # 計算每個樣本的正確預測標籤數量
        correct_predictions += torch.sum(pred_classes == labels.to(device)).item()

        # 計算總標籤數，用於後續計算準確度
        total_labels += labels.numel()

        # 更新進度條
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), correct_predictions.item() / total_labels)

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = accu_loss.item() / (step + 1)
        avg_accuracy = correct_predictions.item() / total_labels

        # # 將結果保存到 DataFrame 並寫入文件
        # new_row = pd.DataFrame({"epoch": [epoch], "mode": ["train"], "loss": [avg_loss], "accuracy": [avg_accuracy]})
        # df = pd.concat([df, new_row], ignore_index=True)

        # df.to_csv(df_file_path, index=False)

        

    # 返回平均損失和準確度
    return avg_loss, avg_accuracy,df

# def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion):
#     model.train()
#     # loss_function = torch.nn.CrossEntropyLoss()
#     loss_function = criterion
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     optimizer.zero_grad()

#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]

#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()

#         loss = loss_function(pred, labels.to(device))
#         loss.backward()
#         accu_loss += loss.detach()

#         data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
#                                                                                accu_loss.item() / (step + 1),
#                                                                                accu_num.item() / sample_num)

#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)

#         optimizer.step()
#         optimizer.zero_grad()

#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num

import torchvision.transforms as transforms

def save_checkpoint(model, mode,epoch,per_best_accuracy, is_best=False, checkpoint_interval=None):
    """保存模型檔案，如果是最佳模型則標記，並依據 checkpoint_interval 定期保存"""
    if is_best:
        if mode == "per_accuracy":
            torch.save(model.state_dict(), "./weights/best_model-{}_{}_per_accuracy.pth".format(epoch, per_best_accuracy))
        else:
            torch.save(model.state_dict(), "./weights/best_model-{}_{}_match_accuracy.pth".format(epoch, per_best_accuracy))
            
    elif checkpoint_interval and epoch % checkpoint_interval == 0:
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

@torch.no_grad()
def evaluate(model, data_loader, device, epoch,criterion ,df, df_file_path, checkpoint_interval,per_best_accuracy,best_exact_match_accuracy,writer=None):
    # 使用適合多標籤分類的損失函數
    loss_function = criterion

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正確的樣本數
    accu_loss = torch.zeros(1).to(device)  # 累計損失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    label_map = {0: "fire", 1: "smoke", 2: "cloud", 3: "none"}
    # 定義反標準化轉換，假設你的標準化是 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    unnormalize = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():  # 訓練時不需要計算梯度

        exact_match_count = 0
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]  # 累計樣本數

            # 模型預測
            pred = model(images.to(device))
            
            # 將預測值轉換為二進制類別，使用 0.5 作為閾值
            pred_classes = torch.sigmoid(pred) > 0.5
            
            true_labels = labels.cpu()

            # 檢查預測是否與實際標籤一致 per
            accu_num += torch.eq(pred_classes.to(device), labels.to(device)).sum()

            # 累積所有樣本的真實標籤和預測標籤
            all_true_labels.extend(true_labels.numpy())
            all_predicted_labels.extend(pred_classes.numpy())

            # 計算損失
            loss = loss_function(pred, labels.to(device).float())  # 將標籤轉為 float
            accu_loss += loss
            

            # 計算完全匹配（Exact Match）準確率
            exact_match = torch.all(torch.eq(pred_classes.to(device), labels.to(device)), dim=1)
            exact_match_count += exact_match.sum().item()  # 完全匹配樣本數


            # 記錄圖片及其預測機率和真實標籤
            if writer is not None and step == 0:  # 每個 epoch 僅記錄一次
                img_grid = vutils.make_grid(images.cpu())  # 創建圖片網格
                writer.add_image(f'Validation Images/Epoch_{epoch}', img_grid)

                # 顯示每張圖片的預測機率
                for i in range(images.size(0)):  # 迭代 batch 中的每張圖片
                    fig, ax = plt.subplots(figsize=(6, 8))  # 增大圖片尺寸，8是為了增加空白區域

                    # 取消標準化並顯示圖像
                    img = unnormalize(images[i].cpu())  # 取消標準化
                    img = torch.clamp(img, 0, 1)  # 確保圖片數據在 [0, 1] 範圍內
                    ax.imshow(img.permute(1, 2, 0))  # (H, W, C)
                    ax.axis('off')

                    # 獲取當前圖片的預測機率
                    label_prob = torch.sigmoid(pred[i]).detach().cpu().numpy()  # 當前圖片的預測機率

                    # 獲取真實標籤（GT）
                    gt_labels = labels[i].cpu().numpy()

                    # 將標籤的預測機率分成兩行顯示
                    half = len(label_map) // 2  # 分成兩部分
                    first_half_pred = " | ".join([f"{label_map[k]}: {v:.2f}" for k, v in enumerate(label_prob[:half])])
                    second_half_pred = " | ".join([f"{label_map[k]}: {v:.2f}" for k, v in enumerate(label_prob[half:], start=half)])
                    title_str_pred = f"{first_half_pred}\n{second_half_pred}"  # 預測值使用換行符 '\n'

                    # 顯示真實標籤 (GT)
                    first_half_gt = " | ".join([f"{label_map[k]}: {int(v)}" for k, v in enumerate(gt_labels[:half])])
                    second_half_gt = " | ".join([f"{label_map[k]}: {int(v)}" for k, v in enumerate(gt_labels[half:], start=half)])
                    title_str_gt = f"{first_half_gt}\n{second_half_gt}"  # GT值使用換行符 '\n'

                    # 調整標題字體大小，這個標題將作為圖像的一部分存在 TensorBoard 中
                    ax.set_title(f"Predicted probabilities:\n{title_str_pred}\n\nGround Truth:\n{title_str_gt}", fontsize=14, pad=20)

                    # 調整圖表的佈局，確保空間充足以顯示標題
                    plt.subplots_adjust(top=0.85)  # 增加圖片上方的空間

                    # 添加圖像的預測機率和GT到 TensorBoard
                    writer.add_figure(f'Validation Predictions/Image_{i+1}/Epoch_{epoch}', fig)

                    # 關閉圖表，避免顯存浪費
                    plt.close(fig)

            # 更新進度條
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, per_acc: {:.3f} ,match_acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / (sample_num * labels.shape[1]),
                                                                                   exact_match_count / sample_num)  # 乘以標籤數計算平均準確率
        avg_loss =accu_loss.item() / (step + 1)
        per_avg_accuracy =accu_num.item() / (sample_num * labels.shape[1])


         # 計算完全匹配準確率
        exact_match_accuracy = exact_match_count / sample_num

        new_row = pd.DataFrame({"epoch": [epoch], "mode": ["test"], "loss": [avg_loss], "per-label-accuracy": [per_avg_accuracy], "exact_match_accuracy":[exact_match_accuracy]})
        new_row = new_row.dropna(how='all')  # 移除全為 NA 的列
        df = pd.concat([df, new_row], ignore_index=True)

         # 檢查是否為最佳模型並保存 Per-Label Accuracy
        is_best = per_avg_accuracy > per_best_accuracy
        if is_best:
            per_best_accuracy = per_avg_accuracy
        save_checkpoint(model,"per_accuracy", epoch,per_best_accuracy, is_best=is_best, checkpoint_interval=checkpoint_interval)

        # 檢查是否為最佳模型並保存 Per-Label Accuracy
        is_best = exact_match_accuracy > best_exact_match_accuracy
        if is_best:
            best_exact_match_accuracy = exact_match_accuracy
        save_checkpoint(model,"match_accuracy", epoch,best_exact_match_accuracy, is_best=is_best, checkpoint_interval=checkpoint_interval)
        # df.to_csv(df_file_path, index=False)

        # 計算多標籤混淆矩陣
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predicted_labels)

        mcm = multilabel_confusion_matrix(y_true, y_pred)

        # 可視化每個類別的混淆矩陣
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for i, (ax, label) in enumerate(zip(axes, labels)):
            cm = mcm[i]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not " + label, label],
                        yticklabels=["Not " + label, label], ax=ax)
            ax.set_title(f"Confusion Matrix for {label} (Epoch {epoch})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        save_dir = "./inference_results"

        plt.tight_layout()
        result_path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch}.png")
        plt.savefig(result_path)

    return   avg_loss,per_avg_accuracy,writer,df,per_best_accuracy,best_exact_match_accuracy

# def evaluate(model, data_loader, device, epoch):
#     loss_function = torch.nn.CrossEntropyLoss()

#     model.eval()

#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失

#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]

#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()

#         loss = loss_function(pred, labels.to(device))
#         accu_loss += loss

#         data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
#                                                                                accu_loss.item() / (step + 1),
#                                                                                accu_num.item() / sample_num)

#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num