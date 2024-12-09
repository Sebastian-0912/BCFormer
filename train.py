import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

from dataset import MultiLabelImageDataset
from vit_model import vit_base_patch32_224_in21k as create_model
from utils import train_one_epoch, evaluate, read_split_data

from torch.utils.data import DataLoader, random_split
# from transformers import ViTImageProcessor, ViTModel
import pandas as pd
from clip_model import CLIPImageClassifier
import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
from our_model.bfc import BFC_Classifier

def main(args):
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # print(f"device:{device}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    checkpoint_interval=10

   
    # labels_file = 'C:/Users/User/Documents/Kuan-wu Chu/final_smoke_datasets/labels.txt'
    # images_dir = 'C:/Users/User/Documents/Kuan-wu Chu/final_smoke_datasets'
    df_file_path='dataframes/results.csv'
    if os.path.exists(os.path.dirname(df_file_path)) is False:
        os.makedirs(os.path.dirname(df_file_path))

    if os.path.exists(df_file_path):
        df = pd.read_csv(df_file_path)
    else:
        df = pd.DataFrame(columns=["epoch","mode", "loss", "per-label-accuracy","exact_match_accuracy"])

    # 使用 MultiLabelImageDataset 進行多標籤分類
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.Resize(224),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
    }

    # # 初始化完整數據集
    # full_dataset = MultiLabelImageDataset(images_dir=images_dir, labels_file=labels_file)

    # # 分割數據集
    # train_size = int(0.8 * len(full_dataset))  # 80% 作為訓練集
    # val_size = len(full_dataset) - train_size   # 剩下的 20% 作為驗證集
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # # 設置不同轉換的 Dataset
    # train_dataset.dataset.transform = data_transform["train"]
    # val_dataset.dataset.transform = data_transform["val"]
    
    # 設定圖片資料夾和標籤文件路徑
    train_images_dir = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/train'   # 訓練集資料夾
    val_images_dir = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val'       # 驗證集資料夾
    train_labels_file = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/train/train_labels.txt'  # 訓練集標籤檔案
    val_labels_file = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val/val_labels.txt'       # 驗證集標籤檔案


    # 分別創建訓練和驗證數據集，並應用對應的轉換
    train_dataset = MultiLabelImageDataset(images_dir=train_images_dir, labels_file=train_labels_file, transform=data_transform["train"])
    val_dataset = MultiLabelImageDataset(images_dir=val_images_dir, labels_file=val_labels_file, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw)

    # 創建 ViT 模型
    # model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # model = CLIPImageClassifier(num_classes=args.num_classes).to(device)
    model = BFC_Classifier(num_classes=args.num_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # 載入預訓練權重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 載入權重
        # print(model.load_state_dict(weights_dict, strict=False))


        # # 刪除不需要的權重
        # del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        
        # # 檢查每個鍵是否存在於 weights_dict 中，若存在則刪除
        # for k in del_keys:
        #     if k in weights_dict:
        #         del weights_dict[k]
        
        

    # 冻结预训练模型中的部分参数
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         if "head" not in name and "pre_logits" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    # 定義優化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Cosine學習率調度
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 使用 BCEWithLogitsLoss 損失函數
    criterion = nn.BCEWithLogitsLoss()

    per_best_accuracy=0.0
    best_exact_match_accuracy=0.0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc,df = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                criterion=criterion,
                                                df=df,
                                                df_file_path=df_file_path)

        scheduler.step()

        # validate
        val_loss, val_acc,tb_writer,df,per_best_accuracy,best_exact_match_accuracy = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     criterion=criterion,
                                     df=df,
                                     df_file_path=df_file_path,
                                     checkpoint_interval=checkpoint_interval,
                                     per_best_accuracy=per_best_accuracy,
                                     
                                     best_exact_match_accuracy=best_exact_match_accuracy,
                                     writer=tb_writer
                                     )
        
        df.to_csv(df_file_path, index=False)

        # 記錄到 TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存模型
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)  # 多標籤分類的類別數
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # parser.add_argument('--data-path', type=str, default="/data/multi_label_dataset")  # 多標籤數據集根目錄
    parser.add_argument('--weights', type=str, default='model/vit-base-patch16-224-in21k.bin',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
