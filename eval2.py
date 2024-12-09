import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import pandas as pd
from dataset import MultiLabelImageDataset
from our_model.bfc import BFC_Classifier
from utils import evaluate
import numpy as np
# from transformers import ViTImageProcessor, ViTModel
import pandas as pd
from clip_model import CLIPImageClassifier
from vit_model import vit_base_patch32_224_in21k as create_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 設定 DataFrame 儲存路徑
    df_file_path = 'dataframes/vit_eval_results.csv'
    if os.path.exists(os.path.dirname(df_file_path)) is False:
        os.makedirs(os.path.dirname(df_file_path))

    # 初始化 DataFrame
    if os.path.exists(df_file_path):
        df = pd.read_csv(df_file_path)
    else:
        df = pd.DataFrame(columns=["filename", "model", "epoch", "loss", "per-label-accuracy", "exact_match_accuracy"])

    # 設定驗證資料
    val_images_dir = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val'
    val_labels_file = 'C:/Users/User/Documents/Kuan-wu Chu/dataset/val/val_labels.txt'

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_dataset = MultiLabelImageDataset(images_dir=val_images_dir, labels_file=val_labels_file, transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 損失函數
    criterion = nn.BCEWithLogitsLoss()

    # 模型檔案路徑
    model_dir = "C:/Users/User/Documents/Kuan-wu Chu/weights/vit-b"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') or f.endswith('.bin')]
    print(f"Found {len(model_files)} models in '{model_dir}'.")

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"Evaluating model: {model_file}")

        # 初始化模型
        # model = BFC_Classifier(num_classes=args.num_classes).to(device)
        # model = CLIPImageClassifier(num_classes=args.num_classes).to(device)
        model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

        model = torch.nn.DataParallel(model)

        # 載入模型權重
        weights_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(weights_dict, strict=False)

        # 驗證
        val_loss, val_acc, _, df, _, _ = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=0,
            criterion=criterion,
            df=df,
            df_file_path=df_file_path,
            checkpoint_interval=None,
            per_best_accuracy=0.0,
            best_exact_match_accuracy=0.0,
            writer=None
        )
        df['filename'] = model_file
        df.to_csv(df_file_path, index=False)
        print(f"Results saved for model: {model_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for multi-label classification')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory containing model files')

    opt = parser.parse_args()
    main(opt)
